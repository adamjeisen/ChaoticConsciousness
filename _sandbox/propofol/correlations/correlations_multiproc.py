# pylint: disable=redefined-outer-name
import argparse
import multiprocessing as mp
import os
import numpy as np
import queue
from scipy.stats import pearsonr
import sys
import time
from tqdm.auto import tqdm

sys.path.append('../../..')
from utils import load, save

def correlations_worker(worker_name, task_queue, message_queue=None):
    # if message_queue is not None:
        # message_queue.put((worker_name, "starting up !!"))
    while True:
        try:
            file_path, results_dir = task_queue.get_nowait()
            window_data = load(file_path)

            num_units = window_data['data'].shape[1]
            window_data['correlations'] = np.zeros((num_units, num_units))
            window_data['p_vals'] = np.zeros((num_units, num_units))


            for i in range(num_units):
                for j in range(num_units):
                    window_data['correlations'][i, j], window_data['p_vals'][i, j] = pearsonr(window_data['data'][:, i], window_data['data'][:, j])

            del window_data['data']
            save(window_data, os.path.join(results_dir, os.path.basename(file_path)))

            if message_queue is not None:
                message_queue.put((worker_name, 'task complete'))

            task_queue.task_done()

        except queue.Empty:
            if message_queue is not None:
                message_queue.put((worker_name, "shutting down..."))
            # message_queue.put((worker_name, "EMPTY"))
            break
            
if __name__ == "__main__":
    task_queue = mp.Manager().JoinableQueue()
    message_queue = mp.Manager().JoinableQueue()

    parser=argparse.ArgumentParser()

    parser.add_argument('--session', '-S', help="Session name to analyze.", default=None, type=str)
    parser.add_argument('--window', '-w', help='Window length in seconds to use for analysis. Defaults to 2.5.', default=2.5, type=float)
    parser.add_argument('--stride', '-s', help='Stride length in seconds to use for analysis. Defaults to 2.5.', default=2.5, type=float)

    args = parser.parse_args()

    session = args.session
    window = args.window
    if window % 1 == 0:
        window = int(window)
        print(window)
    stride = args.stride
    if stride % 1 == 0:
        stride = int(stride)

    if session is None:
        raise ValueError("Session must be provided! Try: 'Mary-Anesthesia-20160809-01'")

    data_dir = f"/om/user/eisenaj/ChaoticConsciousness/data/propofolPuffTone/{session}_window_{window}_stride_{stride}"
    results_dir =  f"/om/user/eisenaj/ChaoticConsciousness/results/propofol/correlations/correlations_{session}_window_{window}_stride_{stride}"
    os.makedirs(results_dir, exist_ok=True)
    files = os.listdir(data_dir)

    for file in files:
        task_queue.put((os.path.join(data_dir, file), results_dir))

    print(f"Queue Size: {task_queue.qsize()}")

    processes = []
    num_workers = 31
    for i in range(num_workers):
        proc = mp.Process(target=correlations_worker, args=(f"worker {i}", task_queue, message_queue))
        processes.append(proc)
        proc.start()
    
    killed_workers = 0
    iterator = tqdm(total=len(files))
    while True:
        try:
            worker_name, message = message_queue.get_nowait()
            
            if message == 'task complete':
                iterator.update(1)
            elif message == "shutting down...":
                killed_workers += 1
            else:
                print(f"[{worker_name}]: {message}")
            message_queue.task_done()
            if killed_workers == num_workers and message_queue.qsize()==0:
                break
        except queue.Empty:
            time.sleep(0.5)
    message_queue.join()
    iterator.close()

    for proc in processes:
        proc.join()
    
    print("Multiprocessing complete...")
