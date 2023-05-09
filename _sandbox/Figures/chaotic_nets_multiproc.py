import argparse 
from copy import deepcopy
import logging
import multiprocessing
import numpy as np
import os
import pandas as pd
import queue
from scipy.integrate import solve_ivp
import time
import torch
import torch.multiprocessing
import traceback
from tqdm.auto import tqdm
import sys

def numpy_torch_conversion(x, use_torch, device='cpu', dtype='torch.DoubleTensor'):
    if use_torch:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.type(dtype).to(device)
    else:
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
    
    return x

def rnn(t, x, W, tau, g):
    return (1/tau)*(-x + g*W @ np.tanh(x))

def rnn_jacobian(x, W, tau, dt, N, use_torch=False, device='cpu', dtype='torch.DoubleTensor'):
    x = numpy_torch_conversion(x, use_torch, device, dtype)
    W = numpy_torch_conversion(W, use_torch, device, dtype)
    if use_torch:
        I = torch.eye(N).type(dtype).to(device)
        if len(x.shape) == 1:
            return I + (dt/tau)*(-I + (W @ torch.diag(1 - torch.tanh(x)**2)))
        else:
            return I.unsqueeze(0) + (dt/tau)*(-I.unsqueeze(0) + (W*((1 - torch.tanh(x)**2).unsqueeze(1))))
    else:
        if len(x.shape) == 1:
            return np.eye(N) + (dt/tau)*(-np.eye(N) + (W @ np.diag(1 - np.tanh(x)**2)))
        else:
            print((1 - np.tanh(x)**2)[:, np.newaxis].shape)
            return np.eye(N)[np.newaxis] + (dt/tau)*(-np.eye(N)[np.newaxis] + (W*(1 - np.tanh(x)**2)[:, np.newaxis]))

def compute_lyaps(Js, dt=1, worker_num=None, message_queue=None, verbose=False):
    T, n = Js.shape[0], Js.shape[1]
    old_Q = np.eye(n)
    lexp = np.zeros(n)
    lexp_counts = np.zeros(n)
    for t in tqdm(range(T), disable=not verbose):
        # QR-decomposition of Js[t] * old_Q
        mat_Q, mat_R = np.linalg.qr(np.dot(Js[t], old_Q))
        # force diagonal of R to be positive
        # (if QR = A then also QLL'R = A with L' = L^-1)
        sign_diag = np.sign(np.diag(mat_R))
        sign_diag[np.where(sign_diag == 0)] = 1
        sign_diag = np.diag(sign_diag)
#         print(sign_diag)
        mat_Q = np.dot(mat_Q, sign_diag)
        mat_R = np.dot(sign_diag, mat_R)
        old_Q = mat_Q
        # successively build sum for Lyapunov exponents
        diag_R = np.diag(mat_R)

#         print(diag_R)
        # filter zeros in mat_R (would lead to -infs)
        idx = np.where(diag_R > 0)
        lexp_i = np.zeros(diag_R.shape, dtype="float32")
        lexp_i[idx] = np.log(diag_R[idx])
#         lexp_i[np.where(diag_R == 0)] = np.inf
        lexp[idx] += lexp_i[idx]
        lexp_counts[idx] += 1

        # it may happen that all R-matrices contained zeros => exponent really has
        # to be -inf

        # normalize exponents over number of individual mat_Rs
#         idx = np.where(lexp_counts > 0)
        #lexp[idx] /= lexp_counts[idx]
#         lexp[np.where(lexp_counts == 0)] = np.inf

        if message_queue is not None:
            message_queue.put((worker_num, "task complete", "DEBUG"))

    return np.divide(lexp, lexp_counts)*(1/dt)

def mp_worker(worker_num, task_queue, message_queue=None):
    # until the task queue is empty, keep taking tasks from the queue and completing them
    while True:
        try:
            # pull a task from the queue
            task_params = task_queue.get_nowait()
            mp_args, run_num = task_params
            save_path = os.path.join(mp_args.RESULTS_DIR, f"RUN_{run_num}")
            
            if message_queue is not None:
                message_queue.put((worker_num, f"starting {save_path}", "DEBUG"))

            if os.path.exists(save_path):
                temp_ret = pd.read_pickle(save_path)
            else:
                temp_ret = None

            dt, tau, N, T, batch_size, num_batches, g_vals = mp_args.dt, mp_args.tau, mp_args.N, mp_args.T, mp_args.batch_size, mp_args.num_batches, mp_args.g_vals

            if message_queue is not None:
                message_queue.put((worker_num, f"RUN {run_num}: Simulating networks", "DEBUG"))

            t_span = [0, T*dt]
            t_eval = np.arange(t_span[0], t_span[1], dt)
            if temp_ret is not None:
                W = temp_ret['W']
                signals = temp_ret['signals']
            else:
                W = np.random.randn(N, N)/np.sqrt(N)
                signals = {}
                for g in g_vals:
                    x0 = np.random.randn(N)
                    sol = solve_ivp(lambda t, x: rnn(t, x, W=W, tau=tau, g=g), t_span=t_span, t_eval=t_eval, y0=x0)
                    signals[g] = sol.y.T
            
                if message_queue is not None:
                    message_queue.put((worker_num, f"RUN {run_num}: Signals are simulated!", "DEBUG"))
            
            if temp_ret is None:
                temp_ret = {'lyaps': {}}

            device = worker_num if mp_args.USE_CUDA else 'cpu'

            lyaps = {}
            for key, signal in signals.items():
                if key in temp_ret['lyaps'].keys():
                    lyaps[key] = temp_ret['lyaps'][key]
                    if message_queue is not None:
                        for i in range(num_batches + T):
                            message_queue.put((worker_num, "task complete", "DEBUG"))
                else:
                    if message_queue is not None:
                        message_queue.put((worker_num, f"RUN {run_num}: Computing Jacobians for g = {key}", "DEBUG"))
                    Js = np.zeros((signal.shape[0], N, N))
                    for batch_num in range(num_batches):
                        start_ind = batch_num*batch_size
                        end_ind = np.min([(batch_num + 1)*batch_size, signal.shape[0]])
                        batch_Js = rnn_jacobian(signal[start_ind:end_ind], W, tau, dt, N, use_torch=mp_args.USE_TORCH, device=device)
                        if mp_args.USE_CUDA:
                            batch_Js = batch_Js.cpu()
                        Js[start_ind:end_ind] = batch_Js
                        if message_queue is not None:
                            message_queue.put((worker_num, "task complete", "DEBUG"))
                    
                    if message_queue is not None:
                        message_queue.put((worker_num, f"RUN {run_num}: Computing Lyaps for g = {key}", "DEBUG"))
                    lyaps[key] = compute_lyaps(Js, dt, worker_num=worker_num, message_queue=message_queue)
                ret = dict(
                    W=W,
                    dt=dt,
                    tau=tau,
                    signals=signals,
                    lyaps=lyaps
                )
                pd.to_pickle(ret, save_path)

            task_queue.task_done()

        # handling what happens when the queue is found to be empty
        except queue.Empty:
            if task_queue.qsize()==0:
                if message_queue is not None:
                    message_queue.put((worker_num, "shutting down...", "INFO"))
            break
        # handling any other exceptions that might come up
        except:
            tb = traceback.format_exc()
            if message_queue is not None:
                message_queue.put((worker_num, tb, "ERROR"))
            task_queue.task_done()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # # Required positional argument
    parser.add_argument('--path', type=str, required=True,
                            help='Required path to the multiprocessing argument dictionary, pickled.')
    parser.add_argument('--job_num', type=str, required=False, default='', help='Job number from slurm.')
    command_line_args = parser.parse_args()
    job_num = command_line_args.job_num

    print(f"Now processing: {command_line_args.path}")
    mp_args = argparse.Namespace(**pd.read_pickle(command_line_args.path))

    # ----------------------
    # Set up logging
    # ----------------------
    if mp_args.USE_LOGGING:
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)
        os.makedirs(mp_args.LOG_DIR, exist_ok=True)
        logging.basicConfig(filename=os.path.join(mp_args.LOG_DIR,f"{mp_args.LOG_NAME}_{timestamp}" + (f"_{job_num}" if len(job_num) > 0 else job_num) + ".log"),
                                filemode='a',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt="%Y-%m-%d %H:%M:%S",
                                level=mp_args.LOG_LEVEL)
        logger = logging.getLogger('mp_log')

        logger.debug("HEY!")
        # logger.info(f"CPU count is: {os.cpu_count()}")

    # ----------------------
    # Set up multiprocessing
    # ----------------------
    if mp_args.USE_TORCH:
        mp = torch.multiprocessing
    else:
        mp = multiprocessing
    os.makedirs(mp_args.RESULTS_DIR, exist_ok=True)
    mp.set_start_method('spawn')
    task_queue = mp.Manager().JoinableQueue()
    message_queue = mp.Manager().JoinableQueue()

    num_tasks = mp_args.num_runs*len(mp_args.g_vals)*(mp_args.num_batches + mp_args.T)
    
    os.makedirs(mp_args.RESULTS_DIR, exist_ok=True)
    # existing_runs = len(os.listdir(mp_args.RESULTS_DIR))
    for i in range(mp_args.num_runs):
        task_queue.put((mp_args, i))

    num_workers = mp_args.NUM_WORKERS
    processes = []
    for worker_num in range(num_workers):
        p = mp.Process(target=mp_worker, args=(worker_num, task_queue, message_queue))
        p.start()
        processes.append(p)

    # monitor for messages from workers
    killed_workers = 0
    iterator = tqdm(total=num_tasks)
    while True:
        try:
            worker_num, message, log_level = message_queue.get_nowait()
            
            if message == 'task complete':
                iterator.update(1)
            elif message == "shutting down...":
                killed_workers += 1
            # print the message from the workr
            if mp_args.USE_LOGGING:
                logger.log(getattr(logging, log_level), f"[worker {worker_num}]: {message}")
            else:
                print(f"[{log_level}] [worker {worker_num}]: {message}")

            message_queue.task_done()
            if killed_workers == num_workers and message_queue.qsize()==0:
                break
            
            sys.stdout.flush()
        except queue.Empty:
            time.sleep(0.5)
    message_queue.join()
    iterator.close()

    for p in processes:
        p.join()
    
    os.remove(command_line_args.path)