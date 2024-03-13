from copy import deepcopy
import numpy as np
import os
import pandas as pd
from spynal.matIO import loadmat
import time
from tqdm.auto import tqdm

def get_data_class(session, all_data_dir):
    data_class = None
    for (dirpath, dirnames, filenames) in os.walk(all_data_dir):
        if f"{session}.mat" in filenames:
            data_class = os.path.basename(dirpath)
            break
    if data_class is None:
        raise ValueError(f"Neural data for session {session} could not be found in the provided folder.")

    return data_class

def compile_grid_results(session, grid_search_results_dir, areas=None, normed=False):
    if normed is False:
        norm_folder = 'NOT_NORMED'
    else:
        norm_folder = 'NORMED'

    if areas is None:
        areas = os.listdir(os.path.join(grid_search_results_dir, session, norm_folder))

    session_results = {}
    for area in areas:
        df = pd.DataFrame({'window': [], 'matrix_size': [], 'r': [], 'AICs': [], 'time_vals': [], 'file_paths': []}).set_index(['window', 'matrix_size', 'r'])
        area_folder = os.path.join(grid_search_results_dir, session, norm_folder, area)
        for f in os.listdir(area_folder):
            t = float(f.split('_')[0])
            file_path = os.path.join(area_folder, f)
            df_new = pd.DataFrame(pd.read_pickle(file_path))
            if np.isnan(df_new.AIC).sum() > 0:
                print(file_path)
            df_new = df_new.set_index(['window', 'matrix_size', 'r'])
            for i, row in df_new.iterrows():
                if i in df.index:
                    df.loc[i, 'AICs'].append(row.AIC)
                    df.loc[i, 'time_vals'].append(t)
                    df.loc[i, 'file_paths'].append(file_path)
                else:
                    df.loc[i] = {'AICs': [row.AIC], 'time_vals': [t], 'file_paths': [file_path]}

        df = df.loc[df.index.sortlevel()[0]]
        session_results[area] = df
    
    return session_results

def combine_grid_results(results_dict):
    all_results = None
    for key, results in results_dict.items():
        if all_results is None:
            all_results = deepcopy(results)
            if 'AICs' not in all_results.columns:
                all_results['AICs'] = all_results.AIC.apply(lambda x: [x])
                all_results = all_results.drop('AIC', axis='columns')
        else:
            for i, row in results.iterrows():
                if i in all_results.index:
                    if 'AICs' in row:
                        all_results.loc[i, 'AICs'].extend(row.AICs)
                    else:
                        all_results.loc[i, 'AICs'].append(row.AIC)
                    if 'time_vals' in all_results.columns:
                        all_results.loc[i, 'time_vals'].extend(row.time_vals)
                    if 'file_paths' in all_results.columns:
                        all_results.loc[i, 'file_paths'].extend(row.file_paths)
                else:
                    if 'AICs' in row:
                        all_results.loc[i] = {'AICs': row.AICs, 'time_vals': row.time_vals, 'file_paths': row.file_paths}
                    else:
                        all_results.loc[i] = {'AICs': [row.AIC], 'time_vals': row.time_vals, 'file_paths': row.file_paths}
#     full_length_inds = all_results.AICs.apply(lambda x: len(x)) == all_results.AICs.apply(lambda x: len(x)).max()
#     window, matrix_size, r = all_results.index[full_length_inds][all_results[full_length_inds].AICs.apply(lambda x: np.mean(x)).argmin()]
    
#     all_results = all_results.drop(all_results[all_results.index.get_level_values('matrix_size') < all_results.index.get_level_values('r')].index, inplace=False)
#     window, matrix_size, r = all_results.index[all_results.AICs.apply(lambda x: np.mean(x)).argmin()]
    
    while True:
        opt_index = all_results.index[all_results.AICs.apply(lambda x: np.mean(x)).argmin()]
        in_all_dfs = True
        for key, result in results_dict.items():
            if opt_index not in result.index:
                in_all_dfs = False
                break

        if in_all_dfs:
            break
        else:
            all_results = all_results.drop(opt_index, inplace=False)
    
    window, matrix_size, r = opt_index

    return window, matrix_size, r, all_results

def get_chosen_params(session, stability_results_dir, grid_search_results_dir, normed=False):
    chosen_params_dir = os.path.join(stability_results_dir, 'chosen_params')
    os.makedirs(chosen_params_dir, exist_ok=True)
    chosen_params_filepath = os.path.join(chosen_params_dir, session)
    if os.path.exists(chosen_params_filepath):
        chosen_params = pd.read_pickle(chosen_params_filepath)
    else:
        session_grid_results = compile_grid_results(session, grid_search_results_dir, normed=normed)
        chosen_params = {}
        for area in session_grid_results.keys():
            window, matrix_size, r, all_results = combine_grid_results({area: session_grid_results[area]})
            chosen_params[area] = dict(
                window=window,
                matrix_size=matrix_size,
                r=r
            )
        pd.to_pickle(chosen_params, chosen_params_filepath)
    
    return chosen_params

def get_stability_run_list(session, stability_results_dir, grid_search_results_dir, all_data_dir, normed=False, T_pred=None, stride=None):
    stability_run_list_dir = os.path.join(stability_results_dir, 'stability_run_lists')
    os.makedirs(stability_run_list_dir, exist_ok=True)
    stability_run_list_file = os.path.join(stability_run_list_dir, session) 

    if os.path.exists(stability_run_list_file):
        stability_run_list = pd.read_pickle(stability_run_list_file)

    # MAKE THE LIST
    else:
        chosen_params = get_chosen_params(session, stability_results_dir, grid_search_results_dir, normed=normed)

        # GET SESSION INFO
        data_class = get_data_class(session, all_data_dir)

        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        variables = ['electrodeInfo', 'lfpSchema']
        session_vars, T, N, dt = load_session_data(session, all_data_dir, variables, data_class=data_class, verbose=False)
        electrode_info, lfp_schema = session_vars['electrodeInfo'], session_vars['lfpSchema']
        areas = np.unique(electrode_info['area'])
        areas = np.concatenate((areas, ('all',)))

        directory_path = os.path.join(all_data_dir, data_class, session + '_lfp_chunked_20s', 'directory')

        stability_run_list = {}
        for area in areas:
            stability_run_list[area] = []
            window = chosen_params[area]['window']
            if stride is None:
                stride = window
            if T_pred is None:
                T_pred = window
        
            if area == 'all':
                unit_indices = np.arange(len(electrode_info['area']))
            else:
                unit_indices = np.where(electrode_info['area'] == area)[0]
            
            num_windows = int(np.floor((T - (window + T_pred))/stride)) + 1
            window_start_times = np.arange(num_windows)*dt*stride
        
            for window_start in window_start_times:
                stability_run_list[area].append(dict(
                    session=session,
                    area=area,
                    window_start=window_start,
                    window_end=window_start + window*dt,
                    test_window_start=window_start + window*dt,
                    test_window_end=window_start + (T_pred + window)*dt,
                    dimension_inds=unit_indices,
                    directory_path=directory_path
                ))
        
                stability_run_list[area][-1] = stability_run_list[area][-1] | chosen_params[area]

        pd.to_pickle(stability_run_list, stability_run_list_file)
    
    return stability_run_list

def save_lfp_chunks(session, chunk_time_s=4*60):
    all_data_dir = f"/om/user/eisenaj/datasets/anesthesia/mat"
    data_class = get_data_class(session, all_data_dir)
    
    filename = os.path.join(all_data_dir, data_class, f'{session}.mat')
    print("Loading data ...")
    start = time.process_time()
    lfp, lfp_schema = loadmat(filename, variables=['lfp', 'lfpSchema'], verbose=False)
    dt = lfp_schema['smpInterval'][0]
    fs = 1/dt
    print(f"Data loaded (took {time.process_time() - start:.2f} seconds)")
    
    save_dir = os.path.join(all_data_dir, data_class, f"{session}_lfp_chunked_{chunk_time_s}s")
    os.makedirs(save_dir, exist_ok=True)
    
    chunk_width = int(chunk_time_s*fs)
    num_chunks = int(np.ceil(lfp.shape[0]/chunk_width))
    directory = []
    for i in tqdm(range(num_chunks)):
        start_ind = i*chunk_width
        end_ind = np.min([(i+1)*chunk_width, lfp.shape[0]])
        chunk = lfp[start_ind:end_ind]
        filepath = os.path.join(save_dir, f"chunk_{i}")
        if os.path.exists(filepath):
            print(f"Chunk at {filepath} already exists")
        else:
            pd.to_pickle(chunk, filepath)
            directory.append(dict(
                start_ind=start_ind,
                end_ind=end_ind,
                filepath=filepath,
                start_time=start_ind*dt,
                end_time=end_ind*dt
            ))
    
    directory = pd.DataFrame(directory)
    
    pd.to_pickle(directory, os.path.join(save_dir, "directory"))
#         print(f"Chunk: {start_ind/(1000*60)} min to {end_ind/(1000*60)} ([{start_ind}, {end_ind}])")

def load_window_from_chunks(window_start, window_end, directory, dimension_inds=None):
    dt = directory.end_time.iloc[0]/directory.end_ind.iloc[0]
    fs = 1/dt
    window_start = int(window_start*fs)
    window_end = int(window_end*fs)
    
    start_time_bool = directory.start_ind <= window_start
    start_row = np.argmin(start_time_bool) - 1 if np.sum(start_time_bool) < len(directory) else len(directory) - 1
    end_time_bool = directory.end_ind > window_end
    end_row = np.argmax(end_time_bool) if np.sum(end_time_bool) > 0 else len(directory) - 1
    
    window_data = None
    
    pos_in_window = 0
    for row_ind in range(start_row, end_row + 1):
        row = directory.iloc[row_ind]
        chunk = pd.read_pickle(row.filepath)
        if dimension_inds is None:
            dimension_inds = np.arange(chunk.shape[1])
        if window_data is None:
            window_data = np.zeros((window_end - window_start, len(dimension_inds)))
                
        if row.start_ind <= window_start:
            start_in_chunk = window_start - row.start_ind
        else:
            start_in_chunk = 0

        if row.end_ind <= window_end:
            end_in_chunk = chunk.shape[0]
        else:
            end_in_chunk = window_end - row.start_ind

        window_data[pos_in_window:pos_in_window + end_in_chunk - start_in_chunk] = chunk[start_in_chunk:end_in_chunk, dimension_inds]
        pos_in_window += end_in_chunk - start_in_chunk
                
    return window_data

def load_session_data(session, all_data_dir, variables, data_class=None, verbose=True):   
    if data_class is None:
        data_class = get_data_class(session, all_data_dir)
    
    filename = os.path.join(all_data_dir, data_class, f'{session}.mat')

    start = time.process_time()
    if 'lfpSchema' not in variables:
        variables.append('lfpSchema')

    if verbose:
        print(f"Loading data: {variables}...")
    start = time.process_time()
    session_vars = {}
    for arg in variables:
        session_vars[arg] = loadmat(filename, variables=[arg], verbose=verbose)
    if verbose:
        print(f"Data loaded (took {time.process_time() - start:.2f} seconds)")

    if 'electrodeInfo' in variables:
        if session in ['MrJones-Anesthesia-20160201-01', 'MrJones-Anesthesia-20160206-01', 'MrJones-Anesthesia-20160210-01']:
            session_vars['electrodeInfo']['area'] = np.delete(session_vars['electrodeInfo']['area'], np.where(np.arange(len(session_vars['electrodeInfo']['area'])) == 60))
            session_vars['electrodeInfo']['channel'] = np.delete(session_vars['electrodeInfo']['channel'], np.where(np.arange(len(session_vars['electrodeInfo']['channel'])) == 60))
            session_vars['electrodeInfo']['NSP'] = np.delete(session_vars['electrodeInfo']['NSP'], np.where(np.arange(len(session_vars['electrodeInfo']['NSP'])) == 60))
        elif data_class == 'leverOddball':
            session_vars['electrodeInfo']['area'] = np.array([f"{area}-{h[0].upper()}" for area, h in zip(session_vars['electrodeInfo']['area'], session_vars['electrodeInfo']['hemisphere'])])
    T = len(session_vars['lfpSchema']['index'][0])
    N = len(session_vars['lfpSchema']['index'][1])
    dt = session_vars['lfpSchema']['smpInterval'][0]

    return session_vars, T, N, dt