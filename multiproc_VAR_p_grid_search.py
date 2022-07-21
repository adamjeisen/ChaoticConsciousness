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
from sklearn.decomposition import PCA
from statsmodels.tsa import stattools
from statsmodels.tsa.api import VAR
import sys
import time
from tqdm.auto import tqdm
import traceback

sys.path.append('../..')
from utils import compile_folder, get_data_class, load, load_session_data, load_window_from_chunks, save

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def compute_VAR_p(window_data, p=1, lamb=0, unit_indices=None, PCA_dim=-1):
    if unit_indices is None:
        chunk = window_data
    else:
        chunk = window_data[:, unit_indices]

    results = {}
    results['explained_variance'] = None
    if PCA_dim > 0:
        if PCA_dim < 2:
            raise ValueError(f"PCA dimension must be greater than 1; provided value was {PCA_dim}")
        pca = PCA(n_components=PCA_dim)
        chunk = pca.fit_transform(chunk)
        results['explained_variance'] = pca.explained_variance_ratio_
    
    window = chunk.shape[0]
    N = chunk.shape[1]

    model = VAR(chunk)
    VAR_results = model.fit(p)
    coefs = VAR_results.coefs
    results['coefs'] = VAR_results.coefs
    results['intercept'] = VAR_results.intercept

    # X_p = np.zeros((N*p + 1, window - p))
    # # Y_p = np.zeros((N*p, window - p))
    # Y = np.zeros((N, window - p))
    # for t in range(window - p):
    #     for i in range(p):
    #         X_p[i*N:(i + 1)*N, t] = chunk[t + p - 1 - i]
    #         # Y_p[i*N:(i + 1)*N, t] = chunk[t + p - i]
    #     Y[:, t] = chunk[t + p]
    # X_p[-1] = np.ones(window - p)
    # U, S, Vh = np.linalg.svd(X_p)

    # S_mat_inv = np.zeros((window - p, N*p + 1))
    # S_mat_inv[np.arange(N*p + 1), np.arange(N*p + 1)] = S/(S**2 + lamb)
    # # full_mat = Y_p[:N] @ Vh.T @ S_mat_inv @ U.t
    # full_mat = Y @ Vh.T @ S_mat_inv @ U.T
    # coefs = np.zeros((p, N, N))
    # for j in range(p):
    #     coefs[j] = full_mat[:, j*N:(j + 1)*N]
    # results['coefs'] = coefs
    # results['intercept'] = full_mat[:, -1]

    A_mat = np.zeros((N*p, N*p))
    for i in range(p):
        A_mat[0:N][:, i*N:(i+1)*N] = coefs[i]

    for i in range(p - 1):
        A_mat[(i + 1)*N:(i + 2)*N][:, i*N:(i + 1)*N] = np.eye(N)
    e = np.linalg.eigvals(A_mat)   
    results['eigs'] = e  
    results['criticality_inds'] = np.abs(e)

    try:
        results['info_criteria'] = VAR_results.info_criteria
    except:
        results['info_criteria'] = None

    return results

def predict_VAR_p(data, coefs, intercept, unit_indices=None):
    if unit_indices is None:
        chunk = data
    else:
        chunk = data[:, unit_indices]
    
    # BUILD PARAMS FOR PREDICTION
    p = coefs.shape[0]
    n = chunk.shape[1]
    
    params = np.zeros((1 + n*p, n))
    params[0] = intercept
    for i in range(p):
        params[1 + i*n:1 + (i + 1)*n] = coefs[i].T

    # LAG DATA
    lagged_data = np.zeros((chunk.shape[0] - p, chunk.shape[1]*p + 1))
    lagged_data[:, 0] = np.ones(lagged_data.shape[0])
    for i in range(p):
        lagged_data[:, i*chunk.shape[1] + 1:(i + 1)*chunk.shape[1] + 1] = chunk[p - 1 - i:chunk.shape[0] - 1 - i]
    
    # PREDICT
    prediction = lagged_data @ params
    true_vals = chunk[p:]
    
    return prediction, true_vals

def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + "_n" + str(counter) + extension
        counter += 1

    return path

def grid_worker(worker_name, task_queue, electrode_info, bandpass_info, message_queue=None):
    if message_queue is not None:
        message_queue.put((worker_name, "starting up !!", "INFO"))
    while True:
        try:
            # if message_queue is not None:
            #     message_queue.put((worker_name, "checking in", "DEBUG"))
            section, window, stride, p, window_ind, T_pred, directory, N, dt, results_dir = task_queue.get_nowait()
            
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

            for area in np.hstack([np.unique(electrode_info['area']), ['all']]):
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

                save(results, uniquify(os.path.join(results_dir, area, section, window_name)))

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
    logging.basicConfig(filename=os.path.join(log_dir,f"grid_multiproc_{timestamp}.log"),
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt="%Y-%m-%d %H:%M:%S",
                            level=logging.DEBUG)
    logger = logging.getLogger('multiprocess_logger')

    task_queue = mp.Manager().JoinableQueue()
    message_queue = mp.Manager().JoinableQueue()

    parser=argparse.ArgumentParser()

    parser.add_argument('--session', '-S', help="Session name to analyze.", default=None, type=str)
    # parser.add_argument('--window', '-w', help='Window length in seconds to use for analysis. Defaults to 2.5.', default=2.5, type=float)
    # parser.add_argument('--stride', '-s', help='Stride length in seconds to use for analysis. Defaults to 2.5.', default=2.5, type=float)
    # parser.add_argument('--task_type', '-t', help="The task type to run - either 'VAR', 'causality' or 'correlations'.", default='VAR', type=str)
    parser.add_argument('--bandpass', '-b', nargs="*", help="Whether to bandpass filter the data before processing.", type=float)

    args = parser.parse_args()

    session = args.session

    if args.bandpass is None:
        bandpass_info = dict(
            flag=False, 
            low=None, 
            high=None
        )
    else:
        bandpass_info = dict(
            flag=True,
            low=args.bandpass[0],
            high=args.bandpass[1]
        )

    if session is None:
        raise ValueError("Session must be provided! For instance: 'Mary-Anesthesia-20160809-01'")
    # if task_type not in ['VAR', 'correlations', 'causality']:
    #     raise ValueError("Task type must be either 'VAR', 'causality' or 'correlations'.")

    logger.info(f"Now running VAR(p) grid search on session {session}")

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
    num_window_samples = 5
    # num_window_samples = 1
    windows = [int(w) if w % 1 == 0 else w for w in np.arange(0.5, 10.1, 0.5)]
    # windows = [4, 5]
    max_lag = 15
    # max_lag = 10
    # max_lag = 4
    T_pred = 25
    thresh = 0.9

    # multiproc parameters
    num_workers = 63

    # -----------------------------
    # RESULTS AND DATA DIRECTORY
    # -----------------------------
    if bandpass_info['flag']:
        results_dir =  f"/om/user/eisenaj/ChaoticConsciousness/results/{data_class}/VAR_p/{session}_num_window_samples_{num_window_samples}_bandpass_{bandpass_info['low']}-{bandpass_info['high']}_{timestamp}"
    else:
        results_dir =  f"/om/user/eisenaj/ChaoticConsciousness/results/{data_class}/VAR_p/{session}_num_window_samples_{num_window_samples}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    logger.info(f"Now running VAR(p) grid search. Results will be saved to {results_dir}.")

    directory = load(os.path.join(data_dir, 'directory'))

    # -----------------------------
    # LOAD NEURAL DATA
    # -----------------------------
    variables = ['electrodeInfo', 'lfpSchema', 'sessionInfo']
    session_vars, T, N, dt = load_session_data(session, all_data_dir, variables, data_class=data_class)
    electrode_info, lfp_schema, session_info = session_vars['electrodeInfo'], session_vars['lfpSchema'], session_vars['sessionInfo']

    # ============================================================
    # GRID SEARCH
    # ============================================================

    grid_results_dir = os.path.join(results_dir, 'grid_search')

    # -----------------------------
    # QUEUE TASKS
    # -----------------------------
    eyes_close = session_info['eyesClose'][-1] if isinstance(session_info['eyesClose'], np.ndarray) else session_info['eyesClose'] 
    section_times = dict( 
        pre=(0, session_info['drugStart'][0]),
        induction=(session_info['drugStart'][0], eyes_close),
        during=(eyes_close, session_info['drugEnd'][1]),
        post=(session_info['drugEnd'][1], T*dt)
    )
    sections = list(section_times.keys())

    areas = np.hstack([np.unique(electrode_info['area']), ['all']])
    for section in sections:
        for area in areas:
            os.makedirs(os.path.join(grid_results_dir, area, section), exist_ok=True)

    total_its = 0
    lags = np.arange(1, max_lag + 1)
    for section in sections:
        for window in windows: 
            stride = window
            # choose window inds to sample
            min_ind = int(section_times[section][0]/stride)
            max_ind = int((section_times[section][1] - window - T_pred*dt)/stride)
            possible_inds = np.arange(min_ind, max_ind + 1)
            window_inds = np.random.choice(possible_inds, size=(np.min([num_window_samples, len(possible_inds)])), replace=False)
            for p in lags:
                total_its += len(window_inds)
                for window_ind in window_inds:
                    task_queue.put((section, window, stride, p, window_ind, T_pred, directory, N, dt, grid_results_dir))

    # -----------------------------
    # RUN WORKERS
    # -----------------------------

    logger.info(f"Queue Size: {task_queue.qsize()}")
    
    processes = []
    for i in range(num_workers):
        proc = mp.Process(target=grid_worker, args=(f"worker {i}", task_queue, electrode_info, bandpass_info, message_queue))
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
    
    logger.info("Grid search multiprocessing complete !!!")

    # -----------------------------
    # COMPILE RESULTS
    # -----------------------------

    logger.info("Compiling grid search results...")

    for area in os.listdir(grid_results_dir):
        for section in os.listdir(os.path.join(grid_results_dir, area)):
            logger.info(f"Compiling {os.path.join(grid_results_dir, area, section)}")
            # compile results
            compile_folder(os.path.join(grid_results_dir, area, section))

    logger.info("Grid search compiling complete !!!")

    # -----------------------------
    # PICK MODEL PARAMETERS
    # -----------------------------
    selected_model_parameters = {}
    test_mse_mats = {}
    grid_search_dfs = {}
    logger.info("Picking model parameters...")
    iterator = tqdm(total = len(areas)*len(sections))
    for area in areas:
        selected_model_parameters[area] = {}
        test_mse_mats[area] = {}
        grid_search_dfs[area] = {}
        for section in sections:
            grid_search_df = load(os.path.join(grid_results_dir, area, section))
            grid_search_dfs[area][section] = grid_search_df

            windows = np.sort(grid_search_df.window.unique())
            lags = np.sort(grid_search_df.p.unique())
            max_lag = lags.max()

            test_mse_mat = np.zeros((len(windows), len(lags)))
            for i, window in enumerate(windows):
                for j, p in enumerate(lags):
                    test_mse_mat[i, j] = grid_search_df[np.logical_and(grid_search_df.window == window, grid_search_df.p == p)].test_mse.mean()
            test_mse_mat[np.isnan(test_mse_mat)] = test_mse_mat[~np.isnan(test_mse_mat)].max()*1.2
            w_ind, p_ind = pick_2d_optimum(test_mse_mat, thresh)

            test_mse_mats[area][section] = test_mse_mat

            selected_model_parameters[area][section] = {'window': windows[w_ind], 'p': lags[p_ind]}
            iterator.update()
    iterator.close()

    analysis = dict(
        selected_model_parameters=selected_model_parameters,
        test_mse_mats=test_mse_mats,
        thresh=thresh  
    )
    
    save(analysis, os.path.join(grid_results_dir, '_analysis'))

    logger.info("Parameter picking complete!!")

    # queue the next job

    os.system(f'sbatch /om2/user/eisenaj/code/ChaoticConsciousness/shell_scripts/multiproc_VAR_p_full_session.sh {results_dir}')

    logger.info("Queued the next job!!")

    # # ============================================================
    # # FULL SESSION PROCESSING
    # # ============================================================

    # task_queue = mp.Manager().JoinableQueue()
    # message_queue = mp.Manager().JoinableQueue()

    # full_results_dir = os.path.join(results_dir, f'results_thresh_{thresh}')

    # for area in areas:
    #     os.makedirs(os.path.join(full_results_dir, area), exist_ok=True)

    # # -----------------------------
    # # QUEUE TASKS
    # # -----------------------------

    # logger.info(f"Queing tasks...")

    # total_its = 0
    # task_list = []
    # for area in areas:
    #     for section in sections:
    #         window = selected_model_parameters[area][section]['window']
    #         p = selected_model_parameters[area][section]['p']
    #         stride = window
    #         window_inds = np.arange(int(section_times[section][0]/stride), int(np.ceil(section_times[section][1]/stride)))
    #         total_its += len(window_inds)
    #         for window_ind in window_inds:
    #             task_list.append((area, section, window, stride, p, window_ind, T_pred, directory, N, dt, full_results_dir))
    
    # for task_info in tqdm(task_list):
    #     task_queue.put(task_info)

    # # -----------------------------
    # # RUN WORKERS
    # # -----------------------------

    # logger.info(f"Queue Size: {task_queue.qsize()}")
    
    # processes = []
    # for i in range(num_workers):
    #     proc = mp.Process(target=worker, args=(f"worker {i}", task_queue, bandpass_info, message_queue, areas))
    #     processes.append(proc)
    #     proc.start()

    # killed_workers = 0
    # iterator = tqdm(total=total_its)
    # while True:
    #     try:
    #         worker_name, message, log_level = message_queue.get_nowait()
            
    #         if message == 'task complete':
    #             iterator.update(1)
    #         elif message == "shutting down...":
    #             killed_workers += 1
    #             logger.log(getattr(logging, log_level), f"[{worker_name}]: {message}")
    #         else:
    #             logger.log(getattr(logging, log_level), f"[{worker_name}]: {message}")
    #         message_queue.task_done()
    #         if killed_workers == num_workers and message_queue.qsize()==0:
    #             break
    #     except queue.Empty:
    #         time.sleep(0.5)
    # message_queue.join()
    # iterator.close()
    
    # for proc in processes:
    #     proc.join()
    
    # logger.info("Multiprocessing complete !!!")

    # # -----------------------------
    # # COMPILE RESULTS
    # # -----------------------------

    # logger.info("Compiling full results...")

    # for area in os.listdir(full_results_dir):
    #     if area != 'grid_search':
    #         logger.info(f"Compiling {os.path.join(full_results_dir, area)}")
    #         # compile results
    #         compile_folder(os.path.join(full_results_dir, area))

    # logger.info("Compiling full results complete !!!")