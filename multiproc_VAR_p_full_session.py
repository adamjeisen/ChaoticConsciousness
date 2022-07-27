# pylint: disable=redefined-outer-name
import argparse
from email import message
import multiprocessing as mp
import os
import logging
from neural_analysis.matIO import loadmat
import numpy as np
import queue
import re
from scipy.signal import butter, lfilter
from scipy.stats import pearsonr
import sys
import time
from tqdm.auto import tqdm
import traceback

sys.path.append('../..')
from dynamical_systems_models import compute_VAR_p, predict_VAR_p
from utils import compile_folder, get_data_class, load, load_session_data, load_window_from_chunks, save

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + "_n" + str(counter) + extension
        counter += 1

    return path

def worker(worker_name, task_queue, electrode_info, bandpass_info, message_queue=None):
    if message_queue is not None:
        message_queue.put((worker_name, "starting up !!", "INFO"))
    while True:
        try:
            # if message_queue is not None:
            #     message_queue.put((worker_name, "checking in", "DEBUG"))
            area, section, window, stride, p, window_ind, T_pred, directory, N, dt, results_dir = task_queue.get_nowait()
            
            start_ind = window_ind*int(stride/dt)
            end_ind = window_ind*int(stride/dt) + int(window/dt)
            start_time = window_ind*stride
            end_time = window_ind*stride + window
            all_window_data = load_window_from_chunks(start_time, end_time + T_pred*dt, directory, N, dt)
            if bandpass_info['flag']:
                all_window_data = butter_bandpass_filter(all_window_data, bandpass_info['low'], bandpass_info['high'], 1/dt)
            test_data = all_window_data[-T_pred - p:]
            window_data = all_window_data[:-T_pred]

            window_name = f"window_start_{start_time}_end_{end_time}"

            if area == 'all':
                unit_indices = np.arange(len(electrode_info['area']))
            else:
                unit_indices = np.where(electrode_info['area'] == area)[0]

            # COMPUTE VAR(p)
            results = compute_VAR_p(window_data, p, unit_indices)

            # PREDICT
            train_prediction, train_true_vals = predict_VAR_p(window_data, results['coefs'], results['intercept'], unit_indices)
            train_mse = ((train_prediction - train_true_vals)**2).mean()
            test_prediction, test_true_vals = predict_VAR_p(test_data, results['coefs'], results['intercept'], unit_indices)
            test_mse = ((test_prediction - test_true_vals)**2).mean()
            persistence_baseline = ((all_window_data[-T_pred:] - all_window_data[-T_pred - 1:-1])**2).mean()

            # ADD TO DICTIONARY
            results['train_mse'] = train_mse
            results['test_mse'] = test_mse
            results['persistence_baseline'] = persistence_baseline

            # ADD TIMESTAMPS
            results['start_ind'] = start_ind
            results['end_ind'] = end_ind
            results['start_time'] = start_time
            results['end_time'] = end_time

            # ADD PARAMETERS
            results['window'] = window
            results['stride'] = stride
            results['p'] = p
            results['T_pred'] = T_pred
            results['area'] = area
            results['section'] = section

            save(results, uniquify(os.path.join(results_dir, area, window_name)))

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

def pick_2d_optimum(mat, thresh=0.95):
    true_min = mat.min()
    i_vals, j_vals = np.where(mat*thresh - true_min <= 0)
    selected_i = np.min(i_vals)
    selected_j = np.min(j_vals[i_vals == selected_i])
    selected_i, selected_j

    return selected_i, selected_j

if __name__ == "__main__":
    # set up logging
    log_dir = "/om2/user/eisenaj/code/ChaoticConsciousness/shell_scripts"
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir,f"full_session_multiproc_{timestamp}.log"),
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt="%Y-%m-%d %H:%M:%S",
                            level=logging.DEBUG)
    logger = logging.getLogger('multiprocess_logger')

    parser=argparse.ArgumentParser()

    parser.add_argument('--results_dir', '-R', help="Results directory to analyze. Must have done grid search.", default=None, type=str)
    # parser.add_argument('--window', '-w', help='Window length in seconds to use for analysis. Defaults to 2.5.', default=2.5, type=float)
    # parser.add_argument('--stride', '-s', help='Stride length in seconds to use for analysis. Defaults to 2.5.', default=2.5, type=float)
    # parser.add_argument('--task_type', '-t', help="The task type to run - either 'VAR', 'causality' or 'correlations'.", default='VAR', type=str)
    # parser.add_argument('--bandpass', '-b', nargs="*", help="Whether to bandpass filter the data before processing.", type=float)

    args = parser.parse_args()

    results_dir = args.results_dir

    if results_dir is None:
        raise ValueError("Results direcotry must be provided!")

    results_folder = os.path.basename(results_dir)
    session = results_folder.split('_')[0]
    if 'bandpass' in results_folder:
        bandpass_info = dict(
                flag=True,
                low = int(bandpass_vals[0]),
                high = int(bandpass_vals[1])
        )
    else:
        bandpass_info = dict(
                flag=False, 
                low=None, 
                high=None
        )

    logger.info(f"Now running VAR(p) on session {session}")

    all_data_dir = f"/om/user/eisenaj/datasets/anesthesia/mat"
    data_class = get_data_class(session, all_data_dir)

    regex = re.compile(f"{session}_lfp_chunked_.*")
    data_dir = None
    chunk_time = np.Inf
    for f in os.listdir(os.path.join(all_data_dir, data_class)):
        if regex.match(f):
            if int(f.split('_')[-1][:-1]) < chunk_time:
                data_dir = os.path.join(all_data_dir, data_class, f)
                chunk_time = int(f.split('_')[-1][:-1])
    if data_dir is None:
        raise ValueError(f"The session {session} has not been chunked. Please evaluate, or split up the data as needed.")
    else:
        logger.info(f"Data located: using chunks of size {chunk_time} seconds")

    # -----------------------------
    # PARAMETERS
    # -----------------------------
    T_pred = 25
    # thresh = 0.9

    # multiproc parameters
    num_workers = 63

    # -----------------------------
    # RESULTS AND DATA DIRECTORY
    # -----------------------------
    logger.info(f"Now running VAR(p) full session. Results will be saved to {results_dir}.")

    directory = load(os.path.join(data_dir, 'directory'))

    # -----------------------------
    # LOAD NEURAL DATA
    # -----------------------------
    variables = ['electrodeInfo', 'lfpSchema', 'sessionInfo']
    session_vars, T, N, dt = load_session_data(session, all_data_dir, variables, data_class=data_class)
    electrode_info, lfp_schema, session_info = session_vars['electrodeInfo'], session_vars['lfpSchema'], session_vars['sessionInfo']

    eyes_close = session_info['eyesClose'][-1] if isinstance(session_info['eyesClose'], np.ndarray) else session_info['eyesClose'] 
    section_times = dict( 
        pre=(0, session_info['drugStart'][0]),
        induction=(session_info['drugStart'][0], eyes_close),
        during=(eyes_close, session_info['drugEnd'][1]),
        post=(session_info['drugEnd'][1], T*dt)
    )
    sections = list(section_times.keys())

    areas = np.hstack([np.unique(electrode_info['area']), ['all']])

    # ============================================================
    # FULL SESSION PROCESSING
    # ============================================================

    task_queue = mp.Manager().JoinableQueue()
    message_queue = mp.Manager().JoinableQueue()

    analysis_dict = load(os.path.join(results_dir, 'grid_search', '_analysis'))
    selected_model_parameters = analysis_dict['selected_model_parameters']

    thresh = analysis_dict['thresh']
    full_results_dir = os.path.join(results_dir, f'results_thresh_{thresh}')

    for area in areas:
        os.makedirs(os.path.join(full_results_dir, area), exist_ok=True)

    # -----------------------------
    # QUEUE TASKS
    # -----------------------------

    logger.info(f"Queing tasks...")

    total_its = 0
    task_list = []
    for area in areas:
        for section in sections:
            window = selected_model_parameters[area][section]['window']
            p = selected_model_parameters[area][section]['p']
            stride = window
            window_inds = np.arange(int(section_times[section][0]/stride), int(np.ceil(section_times[section][1]/stride)))
            total_its += len(window_inds)
            for window_ind in window_inds:
                task_list.append((area, section, window, stride, p, window_ind, T_pred, directory, N, dt, full_results_dir))
    
    for task_info in tqdm(task_list):
        task_queue.put(task_info)

    # -----------------------------
    # RUN WORKERS
    # -----------------------------

    logger.info(f"Queue Size: {task_queue.qsize()}")
    
    processes = []
    for i in range(num_workers):
        proc = mp.Process(target=worker, args=(f"worker {i}", task_queue, electrode_info, bandpass_info, message_queue))
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
    
    for proc in processes:
        proc.join()
    
    logger.info("Multiprocessing complete !!!")

    # -----------------------------
    # COMPILE RESULTS
    # -----------------------------

    logger.info("Compiling full results...")

    for area in os.listdir(full_results_dir):
        if area != 'grid_search':
            logger.info(f"Compiling {os.path.join(full_results_dir, area)}")
            # compile results
            compile_folder(os.path.join(full_results_dir, area))

    logger.info("Compiling full results complete !!!")