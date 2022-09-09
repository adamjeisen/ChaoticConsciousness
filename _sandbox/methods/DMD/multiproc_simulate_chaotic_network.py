from kneed import KneeLocator
import multiprocessing as mp
from nolitsa import delay
import numpy as np
import os
import pandas as pd
import queue
import logging
import time
from tqdm.auto import tqdm
import traceback

def network_deriv(x, t, W, tau, noise=0):
    return (1/tau)*(-x + W @ np.tanh(x) + noise)

def simulation_worker(worker_name, task_queue, message_queue=None):
    if message_queue is not None:
        message_queue.put((worker_name, "starting up !!", "INFO"))
    while True:
        try:
            # if message_queue is not None:
            #     message_queue.put((worker_name, "checking in", "DEBUG"))
            W, g, time_vals, x0, noise_sigma, projection_mat, num_lags, results_dir = task_queue.get_nowait()
            N = W.shape[0]
            dt = time_vals[1]

            key = f"g_{g:.3f}"
            W_eff = W*g
            
            if message_queue is not None:
                message_queue.put((worker_name, f"Simulating g = {g:.3f}", "DEBUG"))

            signal_full = np.zeros((len(time_vals), N))
            signal_full[0] = x0
            noise_vals = np.random.randn(len(time_vals), N)*noise_sigma
            for t in range(1, len(time_vals)):
                signal_full[t] = signal_full[t - 1] + (dt*1000)*network_deriv(signal_full[t-1], time_vals[t-1], W_eff, tau_sim, noise=noise_vals[t-1])
            
        #     signals_full[key] = odeint(lambda y, t: network_deriv(y, t, W=W_eff, tau=tau, noise_sigma), x0, time_vals)
            signal = signal_full @ projection_mat

            if message_queue is not None:
                message_queue.put((worker_name, f"g = {g:.3f} simulation complete!", "DEBUG"))
                message_queue.put((worker_name, f"g = {g:.3f} computing delayed mi", "DEBUG"))

            
            delayed_mi = np.zeros((signal.shape[1], num_lags))
            for i in range(signal.shape[1]):
                for lag in range(num_lags):
                    delayed_mi[i, lag] = delay.mi(signal[lag:, i], signal[:signal.shape[0] - lag, i])
            
            mean_delayed_mi = delayed_mi.mean(axis=0)
            tau = KneeLocator(np.arange(num_lags), mean_delayed_mi, S=40, curve='convex', direction='decreasing').knee
            
            signal_subsampled = signal[np.arange(0, signal.shape[0], tau)]
            
            signal_info = dict(
                signal=signal,
                signal_full=signal_full,
                signal_subsampled=signal_subsampled,
                delayed_mi=delayed_mi,
                tau=tau,
                N=N,
                W=W,
                g=g,
                dt=dt,
                time_vals=time_vals,
                noise_sigma=noise_sigma,
                projection_mat=projection_mat
            )

            pd.to_pickle(signal_info, os.path.join(results_dir, key))
            if message_queue is not None:
                message_queue.put((worker_name, "task complete", "DEBUG"))
            # task_queue.task_done()

        except queue.Empty:
            if message_queue is not None:
                message_queue.put((worker_name, "shutting down...", "INFO"))
            break
        except:
            tb = traceback.format_exc()
            if message_queue is not None:
                message_queue.put((worker_name, tb, "ERROR"))
            # task_queue.task_done()

if __name__ == "__main__":
    # set up logging
    log_dir = "/om2/user/eisenaj/code/ChaoticConsciousness/shell_scripts"
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir,f"chaotic_multiproc_{timestamp}.log"),
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt="%Y-%m-%d %H:%M:%S",
                            level=logging.DEBUG)
    logger = logging.getLogger('multiprocess_logger')

    task_queue = mp.Manager().JoinableQueue()
    message_queue = mp.Manager().JoinableQueue()

    # -----------------------------
    # PARAMETERS
    # -----------------------------
    N = 1000
    dt = 0.001 # s
    tau_sim = 10 # ms
    noise_sigma = 0.005
    num_lags = 1000
    projection_dim = 10
    projection_mat = np.random.randn(N, projection_dim)
    total_time = 2500

    g_vals = np.hstack([np.arange(0.9, 1.6, 0.02), np.arange(1.6, 2, 0.05)])

    W = np.random.randn(N, N)/np.sqrt(N)

    time_vals = np.arange(0, total_time + dt/2, dt)
    x0 = np.random.randn(N)*0.5

    results_dir = f"/om/user/eisenaj/ChaoticConsciousness/data/chaotic_net/N_{N}_dt_{dt}_tausim_{tau_sim}_totaltime_{total_time}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # multiproc parameters
    num_workers = 15
    

    # -----------------------------
    # QUEUE TASKS
    # -----------------------------

    total_its = 0
    for g in g_vals:
        task_queue.put((W, g, time_vals, x0, noise_sigma, projection_mat, num_lags, results_dir))

    # -----------------------------
    # RUN WORKERS
    # -----------------------------

    logger.info(f"Queue Size: {task_queue.qsize()}")
    
    processes = []
    for i in range(num_workers):
        proc = mp.Process(target=simulation_worker, args=(f"worker {i}", task_queue, message_queue))
        processes.append(proc)
        proc.start()

    killed_workers = 0
    iterator = tqdm(total=total_its)
    while True:
        try:
            worker_name, message, log_level = message_queue.get_nowait()
            
            if message == 'task complete':
                iterator.update(1)
            elif message == "shutting down...":
                killed_workers += 1
                logger.log(getattr(logging, log_level), f"[{worker_name}]: {message}")
            else:
                logger.log(getattr(logging, log_level), f"[{worker_name}]: {message}")
            message_queue.task_done()
            if killed_workers == num_workers and message_queue.qsize()==0:
                break
        except queue.Empty:
            time.sleep(0.5)
    message_queue.join()
    iterator.close()
    
    # for proc in processes:
    #     proc.join()
    
    logger.info("Chaotic network simulation complete !!!")
