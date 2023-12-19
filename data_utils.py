from copy import deepcopy
import numpy as np
import os
import pandas as pd
from spynal.matIO import loadmat
import time

def get_data_class(session, all_data_dir):
    data_class = None
    for (dirpath, dirnames, filenames) in os.walk(all_data_dir):
        if f"{session}.mat" in filenames:
            data_class = os.path.basename(dirpath)
            break
    if data_class is None:
        raise ValueError(f"Neural data for session {session} could not be found in the provided folder.")

    return data_class

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

def save_lfp_chunks(session, chunk_time_s=4*60):
    all_data_dir = f"/om/user/eisenaj/datasets/anesthesia/mat"
    data_class = get_data_class(session, all_data_dir)
    
    filename = os.path.join(all_data_dir, data_class, f'{session}.mat')
    print("Loading data ...")
    start = time.process_time()
    lfp, lfp_schema = loadmat(filename, variables=['lfp', 'lfpSchema'], verbose=False)
    dt = lfp_schema['smpInterval'][0]
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
            print(f"Chunk at {file_path} already exists")
        else:
            save(chunk, filepath)
            directory.append(dict(
                start_ind=start_ind,
                end_ind=end_ind,
                filepath=filepath,
                start_time=start_ind*dt,
                end_time=end_ind*dt
            ))
    
    directory = pd.DataFrame(directory)
    
    save(directory, os.path.join(save_dir, "directory"))
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