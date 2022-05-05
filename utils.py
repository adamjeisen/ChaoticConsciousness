from datetime import datetime
import h5py
from neural_analysis.matIO import loadmat
import numpy as np
import os
import pandas as pd
import pickle
import re
import scipy
import shutil
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import time
from tqdm.auto import tqdm

# ===========================
# HDF5 UTILS
# ===========================

# convert an HDF5 object reference or dataset to a string
def to_string(obj, f = None):
    if not isinstance(obj, h5py.Dataset):
        if f is None:
            raise ValueError("If obj is a reference, you must pass the filesystem as argument \`f\`")
        obj = f[obj]
    return ''.join([chr(int(i)) for i in obj[()]])

# get the duration of the test
def get_test_duration(f, units='s'):
    test_duration = f[f['lfpSchema']['index'][0, 0]][-1, 0]
    if units == 'ms':
        test_duration *= 1000
    return test_duration

# get the sample interval (units per sample)
def get_sample_interval(f, units='s'):
    smpInterval = f['lfpSchema']['smpInterval'][0][0]
    if units == 'ms':
        smpInterval *= 1000
    return smpInterval

# get, for each phase of the experiment, the range of times encompassed by that phase
def get_phase_ranges(f, units='s'):
    phase_ranges = dict(
        experiment=np.array([[0, get_test_duration(f)]]),
        initial_phase=np.array([[0, f['sessionInfo']['drugStart'][0, 0]]]),
        loading_phase=np.array([[f['sessionInfo']['drugStart'][0, 0], f['sessionInfo']['drugEnd'][0, 0]]]),
        maintenance_phase=np.array([[f['sessionInfo']['drugStart'][1, 0], f['sessionInfo']['drugEnd'][1, 0]]]),
        unconscious_phase=np.array([[f['sessionInfo']['eyesClose'][-1, 0], f['sessionInfo']['eyesOpen'][-1, 0]]]),
        conscious_phase=np.array(
            [[0, f['sessionInfo']['eyesClose'][-1, 0]], [f['sessionInfo']['eyesOpen'][-1, 0], get_test_duration(f)]]),
        post_anesthesia_phase=np.array([[f['sessionInfo']['drugEnd'][1, 0], get_test_duration(f)]])
        #     loading_while_conscious_phase=np.array([f['sessionInfo']['drugStart'][0, 0], f['sessionInfo']['eyesClose'][-1, 0]])
    )

    if units == 'ms':
        for key in phase_ranges.keys():
            phase_ranges[key] *= 1000

    return phase_ranges

# get all channel names
def get_analog_chnl_names(f):
    ain_index = []
    for item in f[f['ainSchema']['index'][1, 0]][:, 0]:
        ain_index.append(to_string(item, f).split('.')[1])

    return ain_index

# get the analog chnl corresponding to the channel name
def get_analog_chnl(chnl, f):
    ain_index = get_analog_chnl_names(f)
    # chnl_index = np.argmax(np.array(ain_index) == f"{to_string(f['sessionInfo']['session'], f)}.{chnl}")
    chnl_index = np.argmax(np.array(ain_index) == chnl)

    return f['ain'][chnl_index]

# get the number of units
def get_num_units(f):
    return f['unitInfo']['area'].shape[1]

# get the spike times, area and hemisphere for a specific unit
def get_unit_spikes(unit_index, f):
    spike_times = f[f['spikeTimes'][unit_index, 0]][:, 0]
    area = to_string(f[f['unitInfo']['area'][0, unit_index]])
    hemisphere = to_string(f[f['unitInfo']['hemisphere'][0, unit_index]])

    return dict(
        spike_times=spike_times,
        area=area,
        hemisphere=hemisphere
    )

# ===========================
# DATA UTILS
# ===========================

# window, stride in ms
# spike_times, test_duration in s
# returns firing_rate in Hz, bin_locations in s
# requires spike times to be sorted in increasing order!!
def spike_times_to_firing_rate(spike_times, window=50, stride=10, test_duration=None, progress_bar=False):
    if stride is None:
        stride = window
    if stride <= 0:
        stride = window

    window = window * 0.001
    stride = stride * 0.001

    if test_duration is None:
        test_duration = spike_times[-1]

    num_time_bins = int(np.floor((test_duration - window) / stride) + 1)

    firing_rate = np.zeros(num_time_bins)
    bin_location = np.zeros(num_time_bins)
    if progress_bar:
        iterator = tqdm(total=num_time_bins)
    for b in range(num_time_bins):
        start = np.argmax(spike_times >= stride * b)
        end = np.argmin(spike_times < stride * b + window)
        if end == 0 and start != 0:
            end = len(spike_times)
        firing_rate[b] = (end - start) / window
        bin_location[b] = stride * b + window / 2
        if progress_bar:
            iterator.update()

    if progress_bar:
        iterator.close()

    return firing_rate, bin_location


# spike times, time_range are in s
# length, sample_interval are in ms
def get_spike_triggered_average(spike_times, stimulus, length=150, sample_interval=1, time_range=None):
    sta_window = int(length / sample_interval)
    relevant_spikes = spike_times[spike_times >= length * 0.001]
    if time_range is not None:
        indices = np.array([False] * len(relevant_spikes))
        for segment in time_range:
            indices = np.logical_or(indices,
                                    np.logical_and(relevant_spikes >= segment[0], relevant_spikes < segment[1]))
        relevant_spikes = relevant_spikes[indices]

    sta = np.zeros(sta_window)
    for spike_t in relevant_spikes:
        spike_t_step = int(spike_t * 1000 / sample_interval)
        sta += stimulus[spike_t_step - sta_window:spike_t_step]

    return sta / len(relevant_spikes), len(relevant_spikes)

def get_binary_stimuli(f):
    sample_interval = get_sample_interval(f, 's')
    airPuff_binary = np.zeros(int(get_test_duration(f) / sample_interval))
    puff_on_times = f['trialInfo']['cpt_puffOn'][:, 0][~np.isnan(f['trialInfo']['cpt_puffOn'][:, 0])]
    puff_off_times = f['trialInfo']['cpt_puffOff'][:, 0][~np.isnan(f['trialInfo']['cpt_puffOff'][:, 0])]
    for puffOn in puff_on_times:
        puffOff = puff_off_times[np.argmax(puff_off_times > puffOn)]
        airPuff_binary[int(puffOn / sample_interval):int(puffOff / sample_interval)] = 1

    audio_binary = np.zeros(int(get_test_duration(f) / sample_interval))
    tone_on_times = f['trialInfo']['cpt_toneOn'][:, 0][~np.isnan(f['trialInfo']['cpt_toneOn'][:, 0])]
    tone_off_times = f['trialInfo']['cpt_toneOff'][:, 0][~np.isnan(f['trialInfo']['cpt_toneOff'][:, 0])]
    for toneOn in tone_on_times:
        toneOff = tone_off_times[np.argmax(tone_off_times > toneOn)]
        audio_binary[int(toneOn / sample_interval):int(toneOff / sample_interval)] = 1

    return airPuff_binary, audio_binary

# def get_stimuli_start_and
# sys.path.append('../')ff(stim) == -1).astype(np.int)
#         stimuli[stim_name + ' (From Start)'] = stim_begin
#         stimuli[stim_name + ' (From End)'] = stim_end

#     # both and separate
#     tone_only_begin = np.zeros(len(stim))
#     tone_only_end = np.zeros(len(stim))
#     puff_only_begin = np.zeros(len(stim))
#     puff_only_end = np.zeros(len(stim))
#     both_begin = np.zeros(len(stim))
#     both_end = np.zeros(len(stim))
#     count = 0
#     for t, (puffOn, toneOn) in enumerate(zip(f['trialInfo']['cpt_puffOn'][:, 0], f['trialInfo']['cpt_toneOn'][:, 0])):
#         if np.isnan(puffOn):
#             tone_only_begin[int(toneOn * 1000)] = 1
#             tone_only_end[int(f['trialInfo']['cpt_toneOff'][t, 0] * 1000)] = 1
#         elif np.isnan(toneOn):
#             puff_only_begin[int(puffOn * 1000)] = 1
#             puff_only_end[int(f['trialInfo']['cpt_puffOff'][t, 0] * 1000)] = 1
#         else:  # both are on
#             count += 1
#             both_begin[int(np.min([toneOn, puffOn]) * 1000)] = 1
#             both_end[int(np.max([f['trialInfo']['cpt_toneOff'][t, 0], f['trialInfo']['cpt_puffOff'][t, 0]]) * 1000)] = 1
#     stimuli['Tone Only (From Start)'] = tone_only_begin
#     stimuli['Tone Only (From End)'] = tone_only_end
#     stimuli['Puff Only (From Start)'] = puff_only_begin
#     stimuli['Puff Only (From End)'] = puff_only_end
#     stimuli['Tone and Puff (From Start)'] = both_begin
#     stimuli['Tone and Puff (From End)'] = both_end

#     return stimuli

# ===========================
# DATA UTILS
# ===========================

def save(obj, filepath):
    with open(filepath, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load(filepath):
    with open(filepath, 'rb') as handle:
        obj = pickle.load(handle)

    return obj

def compile_folder(folder_to_compile):
    files = os.listdir(folder_to_compile)

    all_window_data = []
    for file in tqdm(files):
        window_data = load(os.path.join(folder_to_compile, file))
        all_window_data.append(window_data) 

    results_df = pd.DataFrame(all_window_data)

    results_df = results_df.sort_values('start_ind')
    results_df = results_df.reset_index(drop=True)

    shutil.rmtree(folder_to_compile)

    save(results_df, folder_to_compile)
    
    return results_df

def get_data_class(session, all_data_dir):
    data_class = None
    for (dirpath, dirnames, filenames) in os.walk(all_data_dir): 
        if f"{session}.mat" in filenames:
            data_class = os.path.basename(dirpath)
            break
    if data_class is None:
        raise ValueError(f"Neural data for session {session} could not be found in the provided folder.")

    return data_class

def get_result_path(results_dir, session, window, stride=None, bandpass_info=None):
    if stride is None:
        stride = window
    
    if bandpass_info is None:
        file_template = f"{os.path.basename(results_dir)}_{session}_window_{window}_stride_{stride}_" + "[a-zA-Z]{3}-" 
    else:
        if bandpass_info['flag']:
            file_template = f"{os.path.basename(results_dir)}_{session}_window_{window}_stride_{stride}_bandpass"
        else:
            file_template = f"{os.path.basename(results_dir)}_{session}_window_{window}_stride_{stride}_" + "[a-zA-Z]{3}-"
    regex = re.compile(file_template)

    matching_files = []
    for file in os.listdir(results_dir):
        if regex.match(file):
            matching_files.append(file)
    
    # pick the most recent result, if there are multiple
    if len(matching_files) == 1:
        filename = matching_files[0]
    else:
        filename = None
        date = datetime(1996, 11, 8)
        for test_file in matching_files:
            test_date = datetime.strptime('_'.join(test_file.split('_')[-2:]), '%b-%d-%Y_%H%M')
            if test_date > date:
                filename = test_file
                date = test_date
    if filename is None:
        raise ValueError(f"File starting with {file_template} not found.")
    
    return os.path.join(results_dir, filename)

def save_lfp_chunks(session, all_data_dir, chunk_time_s=4*60):
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
    
    chunk_width = int(chunk_time_s/dt)
    num_chunks = int(np.ceil(lfp.shape[0]/chunk_width))
    directory = []
    for i in tqdm(range(num_chunks)):
        start_ind = i*chunk_width
        end_ind = np.min([(i+1)*chunk_width, lfp.shape[0]])
        chunk = lfp[start_ind:end_ind]
        filepath = os.path.join(save_dir, f"chunk_{i}")
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

def load_window_from_chunks(window_start, window_end, directory, N, dt):
    window_start = int(window_start/dt)
    window_end = int(window_end/dt)

    start_time_bool = directory.start_ind <= window_start
    start_row = np.argmin(start_time_bool) - 1 if np.sum(start_time_bool) < len(directory) else len(directory) - 1
    end_time_bool = directory.end_ind > window_end
    end_row = np.argmax(end_time_bool) if np.sum(end_time_bool) > 0 else len(directory) - 1

    window_data = np.zeros((window_end - window_start, N))

    pos_in_window = 0
    for row_ind in range(start_row, end_row + 1):
        row = directory.iloc[row_ind]
        chunk = load(row.filepath)

        if row.start_ind <= window_start:
            start_in_chunk = window_start - row.start_ind
        else:
            start_in_chunk = 0

        if row.end_ind <= window_end:
            end_in_chunk = chunk.shape[0]
        else:
            end_in_chunk = window_end - row.start_ind

        # print("----------------------")
        # print(window_start, window_end)
        # print(row.start_time, row.end_time)
        # print(start_in_chunk, end_in_chunk)
        # print(pos_in_window, pos_in_window + end_in_chunk - start_in_chunk)

        window_data[pos_in_window:pos_in_window + end_in_chunk - start_in_chunk] = chunk[start_in_chunk:end_in_chunk]
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
        session_vars[arg] = loadmat(filename, variables=[arg], verbose=False)
    if verbose:
        print(f"Data loaded (took {time.process_time() - start:.2f} seconds)")

    if 'electrodeInfo' in variables:
        if session in ['MrJones-Anesthesia-20160201-01', 'MrJones-Anesthesia-20160206-01', 'MrJones-Anesthesia-20160210-01']:
            session_vars['electrodeInfo']['area'] = np.delete(session_vars['electrodeInfo']['area'], np.where(np.arange(len(session_vars['electrodeInfo']['area'])) == 60))
        elif data_class == 'leverOddball':
            session_vars['electrodeInfo']['area'] = np.array([f"{area}-{h[0].upper()}" for area, h in zip(session_vars['electrodeInfo']['area'], session_vars['electrodeInfo']['hemisphere'])])
    T = len(session_vars['lfpSchema']['index'][0])
    N = len(session_vars['lfpSchema']['index'][1])
    dt = session_vars['lfpSchema']['smpInterval'][0]

    return session_vars, T, N, dt

def run_window_selection(session, pred_steps=10, pct_of_value=0.95, return_data=False, bandpass_info=None, verbose=True):
    all_data_dir = f"/om/user/eisenaj/datasets/anesthesia/mat"
    data_class = get_data_class(session, all_data_dir)

    results_dir = os.path.join(f'/om/user/eisenaj/ChaoticConsciousness/results/{data_class}/VAR')

    # -----------------------------------------------------
    # check if session has already computed VAR_results with selected windows
    # (note that this does not check if more windows have been added since the last computation)
    # -----------------------------------------------------
    window_selection_dir = os.path.join(results_dir, 'window_selection')
    os.makedirs(window_selection_dir, exist_ok=True)

    if bandpass_info is None:
        regex = re.compile(f"VAR_{session}_selected_windows_phases_{pred_steps}_steps")
    else:
        if bandpass_info['flag']:
            regex = re.compile(f"VAR_{session}_selected_windows_bandpass_phases_{pred_steps}_steps")
        else:
            regex = re.compile(f"VAR_{session}_selected_windows_phases_{pred_steps}_steps")

    VAR_results_path = None
    for file_name in os.listdir(window_selection_dir):
        if regex.match(file_name):
            VAR_results_path = os.path.join(window_selection_dir, file_name)
            if verbose:
                print(f"VAR results with selected windows found for session {session}.")
            break
    
    # -----------------------------------------------------
    # check if session has already computed selected windows
    # (note that this does not check if more windows have been added since the last computation)
    # -----------------------------------------------------
    if bandpass_info is None:
        regex = re.compile(f"{session}_selected_windows_phases_{pred_steps}_steps")
    else:
        if bandpass_info['flag']:
            regex = re.compile(f"{session}_selected_windows_bandpass_phases_{pred_steps}_steps")
        else:
            regex = re.compile(f"{session}_selected_windows_phases_{pred_steps}_steps")
    

    window_selection_info_path = None
    for file_name in os.listdir(window_selection_dir):
        if regex.match(file_name):
            window_selection_info_path = os.path.join(window_selection_dir, file_name)
            if verbose:
                print(f"Window selection info found for session {session}.")
            break

    if VAR_results_path is not None:
        
        if window_selection_info_path is None:
            if verbose:
                print("Even though results were found, couldn't find window selection info.")
            if return_data:
                return load(VAR_results_path), None
        else:
            if return_data:
                return load(VAR_results_path), load(window_selection_info_path)

    else: # ONLY RUNS THIS IF VAR RESULTS NOT FOUND:

        # -----------------------------------------------------
        # make slice function
        # -----------------------------------------------------
        filename = os.path.join(all_data_dir, data_class, f'{session}.mat')
        session_info = loadmat(filename, variables=['sessionInfo'], verbose=False)
        if data_class == 'propofolPuffTone':
            
            # slice_funcs = dict(
            #     pre=lambda window: slice(0, int(session_info['drugStart'][0]/window)),
            #     during=lambda window: slice(int(session_info['drugStart'][0]/window), int(session_info['drugEnd'][1]/window)),
            #     post=lambda window: slice(int(session_info['drugEnd'][1]/window),-1)
            # )

            eyes_close = session_info['eyesClose'][1] if isinstance(session_info['eyesClose'], np.ndarray) else session_info['eyesClose']
            slice_funcs = dict(
                pre=lambda window: slice(0, int(session_info['drugStart'][0]/window)),
                induction=lambda window: slice(int(session_info['drugStart'][0]/window), int(eyes_close/window)),
                during=lambda window: slice(int(eyes_close/window), int(session_info['drugEnd'][1]/window)),
                post=lambda window: slice(int(session_info['drugEnd'][1]/window),-1)
            )
        elif data_class == 'leverOddball':
            slice_funcs = dict(
                whole=lambda window: slice(0, -1)
            )
        # -----------------------------------------------------
        # RUN WINDOW SELECTION
        # -----------------------------------------------------
        if window_selection_info_path is None:
            if verbose:
                print(f"Session {session}: selected windows by phase with {pred_steps}-step prediction not found. Running now.")
                
            variables = ['electrodeInfo', 'lfp', 'lfpSchema']
            session_vars, T, N, dt = load_session_data(session, all_data_dir, variables, data_class=data_class, verbose=verbose)
            electrode_info, lfp, lfp_schema = session_vars['electrodeInfo'], session_vars['lfp'], session_vars['lfpSchema']

            # find completed windows 
            # TODO: account for the specific bandpass frequencies
            windows = []
            if bandpass_info is None:
                regex = re.compile(f"VAR_{session}_window_" + ".{1,3}_stride_.{1,3}_[a-zA-Z]{3}-")
            else:
                if bandpass_info['flag']:
                    regex = re.compile(f"VAR_{session}_window_" + ".{1,3}_stride_.{1,3}_bandpass")
                else:
                    regex = re.compile(f"VAR_{session}_window_" + ".{1,3}_stride_.{1,3}_[a-zA-Z]{3}-")

            for file_name in os.listdir(results_dir):
                if regex.match(file_name):
                    if bandpass_info is None:
                        windows.append(float(file_name.split('_')[-3]))
                    else:
                        if bandpass_info['flag']:
                            windows.append(float(file_name.split('_')[-7]))
                        else:
                            windows.append(float(file_name.split('_')[-3]))
            windows.sort()
            windows = [int(w) if w % 1 == 0 else w for w in np.unique(windows)]

            # -----------------------------------------------------
            # compute forward predictions
            # (note that this assumes window and stride are the same)
            # -----------------------------------------------------

            T_pred = pred_steps

            predictions = {}
            true_vals = {}
            step_mse = {}
            for area in np.unique(electrode_info['area']):
                predictions[area] = {}
                true_vals[area] = {}
                step_mse[area] = {}
            predictions['all'] = {}
            true_vals['all'] = {}
            step_mse['all'] = {}

            for window in windows:
                stride = window
                if verbose:
                    print(f"Now computing window = {window}")
                VAR_results_dir = get_result_path(results_dir, session, window, stride, bandpass_info=bandpass_info)
                VAR_results = {}
                for file_name in tqdm(os.listdir(VAR_results_dir)):
                    try:
                        VAR_results[file_name] = load(os.path.join(VAR_results_dir, file_name))
                    except IsADirectoryError:
                        if verbose:
                            print(f"Need to compile {os.path.join(VAR_results_dir, file_name)}")
                        # compile results
                        VAR_results[file_name] = compile_folder(os.path.join(VAR_results_dir, file_name))
                
                for area in VAR_results.keys():
                    if area == 'all':
                        unit_indices = np.arange(len(electrode_info['area']))
                    else:
                        unit_indices = np.where(electrode_info['area'] == area)[0]
                    
                    predictions[area][window] = np.zeros((len(VAR_results[area]) - 1, T_pred, len(unit_indices)))
                    true_vals[area][window] = np.zeros(predictions[area][window].shape)

                    for i in tqdm(range(predictions[area][window].shape[0])):
                        row = VAR_results[area].iloc[i]
                        start_step = int(stride*i/dt)
                        # x0 = lfp[start_step + int(window/dt) - 1, unit_indices]

                        for t in range(T_pred):
                            predictions[area][window][i, t] = np.hstack([[1], lfp[start_step + int(window/dt) - 1 + t, unit_indices]]) @ row.A_mat_with_bias
                            # if t == 0:
                            #     predictions[area][window][i, t] = np.hstack([[1], x0]) @ row.A_mat_with_bias
                            # else:
                            #     predictions[area][window][i, t] = np.hstack([[1], predictions[area][window][i, t - 1]]) @ row.A_mat_with_bias

                        true_vals[area][window][i] = lfp[start_step + int(window/dt):start_step + int(window/dt) + T_pred, unit_indices]

                    step_mse[area][window] = ((predictions[area][window] - true_vals[area][window])**2).mean(axis=2)
            
            # -----------------------------------------------------
            # pick and save selected_windows
            # -----------------------------------------------------
            selected_windows = {}
            window_mses = {}

            phases = slice_funcs.keys()

            for phase in phases:
                slice_func = slice_funcs[phase]
                selected_windows[phase] = {}
                window_mses[phase] = {}
                for area in step_mse.keys():
                    window_mses[phase][area] = [step_mse[area][window][slice_func(window), :].mean() for window in windows]

                    asymptotic_value = np.array(window_mses[phase][area]).min()
                    asymptotic_ind = np.argmin(window_mses[phase][area])
                    for i in range(len(window_mses[phase][area])):
                        if window_mses[phase][area][i]*pct_of_value <= asymptotic_value or i == asymptotic_ind:
                            selected_windows[phase][area] = windows[i]
                            break
            
            window_selection_info = dict(
                selected_windows=selected_windows,
                predictions=predictions,
                true_vals=true_vals,
                step_mse=step_mse,
                window_mses=window_mses
            )

            save_file_path = os.path.join(window_selection_dir, f"{session}_selected_windows_phases_{pred_steps}_steps")
            if bandpass_info is not None:
                if bandpass_info['flag']:
                    save_file_path = os.path.join(window_selection_dir, f"{session}_selected_windows_bandpass_phases_{pred_steps}_steps")
            save(window_selection_info, save_file_path)
        else:
            window_selection_info = load(window_selection_info_path)
        
        # -----------------------------------------------------
        # LOAD AND COMPILE DATA
        # -----------------------------------------------------

        selected_windows = window_selection_info['selected_windows']

        window_info = {}
        for phase in selected_windows.keys():
            for area, window in selected_windows[phase].items():
                window = int(window) if window % 1 == 0 else window
                if window not in window_info.keys():
                    window_info[window] = []
                window_info[window].append((area, phase))

        columns = ['explained_variance', 'A_mat', 'A_mat_with_bias', 'eigs',
                    'criticality_inds', 'sigma2_ML', 'AIC', 'sigma_norm', 'start_time',
                        'start_ind', 'end_time', 'end_ind']
                
        VAR_results = {}
        key = list(selected_windows.keys())[0]
        for area in selected_windows[key].keys():
            VAR_results[area] = {col: [] for col in columns}


        for window in window_info.keys():
            stride = window
            areas_to_load = np.unique([entry[0] for entry in window_info[window]])
            VAR_results_dir = get_result_path(results_dir, session, window, stride)
            
            temp_results = {}
            for area in areas_to_load:
                if verbose:
                    print(f"Now attempting to load area {area} with window {window}")
                try:
                    temp_results[area] = load(os.path.join(VAR_results_dir, area))
                except IsADirectoryError:
                    if verbose:
                        print(f"Need to compile {os.path.join(VAR_results_dir, area)}")
                    # compile results
                    temp_results[area] = compile_folder(os.path.join(VAR_results_dir, area))
            
            for (area, phase) in window_info[window]:
                for key in columns:
                    VAR_results[area][key].extend(temp_results[area][key].iloc[slice_funcs[phase](window)])
                VAR_results

        for area in VAR_results.keys():
            VAR_results[area] = pd.DataFrame(VAR_results[area]).sort_values('start_time').reset_index(drop=True)
            VAR_results[area]['window'] = VAR_results[area].apply(lambda row: row.end_time - row.start_time, axis=1)
        
        save_file_path = os.path.join(window_selection_dir, f"VAR_{session}_selected_windows_phases_{pred_steps}_steps")
        if bandpass_info is not None:
            if bandpass_info['flag']:
                save_file_path = os.path.join(window_selection_dir, f"VAR_{session}_selected_windows_bandpass_phases_{pred_steps}_steps")

        save(VAR_results, save_file_path)
        
        if return_data:
            return VAR_results, window_selection_info

def compute_summary_statistics(df, session_info):
    ret_info = {}
    
    # -----------------
    # MEAN STATS AND MANN WHITNEY U TEST
    # -----------------
    
    if isinstance(session_info['eyesClose'], float):
        eyes_close = session_info['eyesClose']
    else:
        eyes_close = session_info['eyesClose'][-1]

    anesthesia_start_ind = np.argmax(df['start_time'] > session_info['drugStart'][0])
    LOC_ind = np.argmax(df['start_time'] > eyes_close)
    anesthesia_end_ind = np.argmax(df['start_time'] > session_info['drugEnd'][1])
    
    wake_inds = np.hstack(df[:anesthesia_start_ind].criticality_inds.to_numpy())
    anesthesia_inds = np.hstack(df[LOC_ind:anesthesia_end_ind].criticality_inds.to_numpy())

    ret_info['wake_mean'] = wake_inds.mean()
    ret_info['wake_sd'] = wake_inds.std()
    ret_info['wake_se'] = ret_info['wake_sd']/np.sqrt(len(wake_inds))

    ret_info['anesthesia_mean'] = anesthesia_inds.mean()
    ret_info['anesthesia_sd'] = anesthesia_inds.std()
    ret_info['anesthesia_se'] = ret_info['anesthesia_sd']/np.sqrt(len(anesthesia_inds))
    
    mannwhitney_ret = scipy.stats.mannwhitneyu(wake_inds, anesthesia_inds, alternative='less')
    ret_info['mannwhitney_stat'] = mannwhitney_ret.statistic
    ret_info['mannwhitney_p'] = mannwhitney_ret.pvalue
    
    # plt.figure(figsize=(12, 5))
    # plt.hist(wake_inds, label='wakeful', density=True, alpha=0.7)
    # plt.axvline(wake_inds.mean(), c='C0', linestyle='--', label='wakeful mean')
    # plt.hist(anesthesia_inds, label='anesthesia', density=True, alpha=0.7)
    # plt.axvline(anesthesia_inds.mean(), c='C1', linestyle='--', label='anesthesia mean')
    # plt.legend(fontsize=13)
    # plt.ylabel("Density", fontsize=14)
    # plt.xlabel("Criticality Index", fontsize=14)
    # plt.tick_params(labelsize=13)
    # plt.show()
    
    # -----------------
    # DESTABILIZATION RATE
    # -----------------
    
    destab_inds = df[anesthesia_start_ind:LOC_ind].criticality_inds.apply(lambda x: x.mean()).to_numpy()

    time_vals = df[anesthesia_start_ind:LOC_ind].start_time.to_numpy().reshape(-1, 1)
    lr_fit = LinearRegression().fit(time_vals, destab_inds)
    beta = lr_fit.coef_[0] # s^{-1}
    ret_info['beta'] = beta
    
    preds = lr_fit.predict(time_vals)
    sigma_2 = np.mean((preds - destab_inds)**2)
    var_beta = sigma_2/((time_vals - time_vals.mean())**2).sum()
    beta_SD = np.sqrt(var_beta)
    ret_info['beta_SE'] = beta_SD
    CI = scipy.stats.t.ppf(0.975, len(destab_inds) - 2)*beta_SD
    ret_info['CI_low'], ret_info['CI_high'] = beta - CI, beta + CI
    ret_info['r2_score'] = r2_score(destab_inds, lr_fit.predict(time_vals))
    
    # plt.plot(time_vals/60, destab_inds, label='true values')
    # plt.plot(time_vals/60, preds, label=f'regression: rate = {beta:.2e}' + r'$\pm$' + f'{CI:.2e}' + r' $s^{-1}$', linestyle='--')
    # plt.xlabel("Time in Session (min)", fontsize=14)
    # plt.ylabel("Mean Criticality Index", fontsize=14)
    # plt.tick_params(labelsize=13)
    # plt.legend(fontsize=11)
    # plt.show()
    
    return ret_info