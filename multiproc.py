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
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from statsmodels.tsa import stattools
from statsmodels.tsa.api import VAR
import sys
import time
from tqdm.auto import tqdm
import traceback

sys.path.append('../..')
from utils import compile_folder, get_data_class, load, load_window_from_chunks, save

def compute_VAR(window_data, unit_indices=None, PCA_dim=-1):
    if unit_indices is None:
        chunk = window_data
    else:
        chunk = window_data[:, unit_indices]
    k = chunk.shape[0]

    results = {}
    results['explained_variance'] = None
    if PCA_dim > 0:
        if PCA_dim < 2:
            raise ValueError(f"PCA dimension must be greater than 1; provided value was {PCA_dim}")
        pca = PCA(n_components=PCA_dim)
        chunk = pca.fit_transform(chunk)
        results['explained_variance'] = pca.explained_variance_ratio_
    
    model = VAR(chunk)
    VAR_results = model.fit(1)
    results['A_mat'] = VAR_results.coefs[0]
    results['A_mat_with_bias'] = VAR_results.params
    e,_ = np.linalg.eig(VAR_results.coefs[0])   
    results['eigs'] = e   
    results['criticality_inds'] = np.abs(e)

    results['sigma2_ML'] = np.linalg.norm(VAR_results.endog[1:] - (VAR_results.endog_lagged @ VAR_results.params), axis=1).sum()/(k - 2)
    results['AIC'] = k*np.log(results['sigma2_ML']) + 2
    results['sigma_norm'] = np.linalg.norm(VAR_results.sigma_u, ord=2)

    return results

def compute_causality(window_data, p=1):
    num_units = window_data.shape[1]

    results = {}
    results['causality'] = np.zeros((num_units, num_units))

    lags = [p]
    for i in range(num_units):
        for j in range(num_units):
            grangers = stattools.grangercausalitytests(window_data[:, [j, i]], lags, verbose=False)
            true_vals = grangers[p][1][0].model.data.endog
            restricted_preds = grangers[p][1][0].model.data.exog @ grangers[p][1][0].params
            unrestricted_preds = grangers[p][1][1].model.data.exog @ grangers[p][1][1].params
            restricted_error = true_vals - restricted_preds
            unrestricted_error = true_vals - unrestricted_preds
            results['causality'][i, j] = np.log(np.var(restricted_error)/np.var(unrestricted_error))
    
    return results

def compute_correlations(window_data):
    results = {}
    results['correlations'] = np.zeros((num_units, num_units))
    results['p_vals'] = np.zeros((num_units, num_units))
    num_units = window_data.shape[1]

    for i in range(num_units):
        for j in range(num_units):
            results['correlations'][i, j], results['p_vals'][i, j] = pearsonr(window_data[:, i], window_data[:, j])

    return results

def worker(worker_name, task_queue, message_queue=None, areas=None):
    if message_queue is not None:
        message_queue.put((worker_name, "starting up !!", "INFO"))
    while True:
        try:
            # if message_queue is not None:
            #     message_queue.put((worker_name, "checking in", "DEBUG"))
            window_start, window_end, directory, N, dt, results_dir, task_type = task_queue.get_nowait()
            window_data = load_window_from_chunks(window_start, window_end, directory, N, dt)
            
            window_name = f"window_start_{window_start}_end_{window_end}"

            if task_type == 'VAR':
                for area in np.unique(areas):
                    unit_indices = np.where(area == areas)[0]
                    results = compute_VAR(window_data, unit_indices)

                    results['start_time'] = window_start
                    results['start_ind'] = int(window_start/dt)
                    results['end_time'] = window_end
                    results['end_ind'] = int(window_end/dt)
                    save(results, os.path.join(results_dir, area, window_name))
                
                results = compute_VAR(window_data)
                results['start_time'] = window_start
                results['start_ind'] = int(window_start/dt)
                results['end_time'] = window_end
                results['end_ind'] = int(window_end/dt)
                save(results, os.path.join(results_dir, 'all', window_name))

            elif task_type == 'causality':
                
                results = compute_causality(window_data, p=1)
                results['start_time'] = window_start
                results['start_ind'] = int(window_start/dt)
                results['end_time'] = window_end
                results['end_ind'] = int(window_end/dt)
                save(results, os.path.join(results_dir, window_name))

            else: # task_type == 'correlations'

                results = compute_correlations(window_data)
                results['start_time'] = window_start
                results['start_ind'] = int(window_start/dt)
                results['end_time'] = window_end
                results['end_ind'] = int(window_end/dt)
                save(results, os.path.join(results_dir, window_name))

            if message_queue is not None:
                message_queue.put((worker_name, "task complete", "DEBUG"))
            task_queue.task_done()

        except queue.Empty:
            if message_queue is not None:
                message_queue.put((worker_name, "shutting down...", "INFO"))
            break
        except:
            tb = traceback.format_exc()
            if message_queue is not None:
                message_queue.put((worker_name, tb, "ERROR"))
            task_queue.task_done()
            
if __name__ == "__main__":
    # set up logging
    log_dir = "/om2/user/eisenaj/code/ChaoticConsciousness/shell_scripts"
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir,f"multiproc_{timestamp}.log"),
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt="%Y-%m-%d %H:%M:%S",
                            level=logging.DEBUG)
    logger = logging.getLogger('multiprocess_logger')

    task_queue = mp.Manager().JoinableQueue()
    message_queue = mp.Manager().JoinableQueue()

    parser=argparse.ArgumentParser()

    parser.add_argument('--session', '-S', help="Session name to analyze.", default=None, type=str)
    parser.add_argument('--window', '-w', help='Window length in seconds to use for analysis. Defaults to 2.5.', default=2.5, type=float)
    parser.add_argument('--stride', '-s', help='Stride length in seconds to use for analysis. Defaults to 2.5.', default=2.5, type=float)
    parser.add_argument('--task_type', '-t', help="The task type to run - either 'VAR', 'causality' or 'correlations'.", default='VAR', type=str)

    args = parser.parse_args()

    session = args.session

    window = args.window
    if window % 1 == 0:
        window = int(window)

    stride = args.stride
    if stride % 1 == 0:
        stride = int(stride)
    
    task_type = args.task_type

    if session is None:
        raise ValueError("Session must be provided! For instance: 'Mary-Anesthesia-20160809-01'")
    if task_type not in ['VAR', 'correlations', 'causality']:
        raise ValueError("Task type must be either 'VAR', 'causality' or 'correlations'.")

    logger.info(f"Now running {task_type} on session {session} with window = {window} and stride = {stride}")

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

    results_dir =  f"/om/user/eisenaj/ChaoticConsciousness/results/{data_class}/{task_type}/{task_type}_{session}_window_{window}_stride_{stride}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    if task_type == 'VAR':
        electrode_info = loadmat(os.path.join(all_data_dir, data_class, f"{session}.mat"), variables=['electrodeInfo'], verbose=False)
        areas = electrode_info['area']
        for area in np.unique(areas):
            os.makedirs(os.path.join(results_dir, area), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'all'), exist_ok=True)

    logger.info(f"Now running {task_type}. Results will be saved to {results_dir}.")

    directory = load(os.path.join(data_dir, 'directory'))
    lfp_schema = loadmat(os.path.join(all_data_dir, data_class, f"{session}.mat"), variables=['lfpSchema'], verbose=False)
    T = len(lfp_schema['index'][0])
    N = len(lfp_schema['index'][1])
    dt = lfp_schema['smpInterval'][0]
    num_windows = int(np.floor((T-int(window/dt))/int(stride/dt))+1)

    for i in range(num_windows):
        window_start = i*stride
        window_end = i*stride + window
        task_queue.put((window_start, window_end, directory, N, dt, results_dir, task_type))

    logger.info(f"Queue Size: {task_queue.qsize()}")
    
    processes = []
    num_workers = 31
    for i in range(num_workers):
        if task_type == 'VAR':
            proc = mp.Process(target=worker, args=(f"worker {i}", task_queue, message_queue, areas))
        else:
            proc = mp.Process(target=worker, args=(f"worker {i}", task_queue, message_queue))
        processes.append(proc)
        proc.start()

    killed_workers = 0
    iterator = tqdm(total=num_windows)
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

    logger.info("Compiling results...")

    if task_type == 'VAR':
        for folder_name in tqdm(os.listdir(results_dir)):
            logger.info(f"Compiling {os.path.join(results_dir, folder_name)}")
            # compile results
            compile_folder(os.path.join(results_dir, folder_name))
    else:
        logger.info(f"Compiling {results_dir}")
        compile_folder(results_dir)

    logger.info("Compiling complete !!!")
