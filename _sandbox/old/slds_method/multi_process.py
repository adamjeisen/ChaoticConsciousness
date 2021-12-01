from matplotlib import colors
import matplotlib.pyplot as plt
import multiprocessing
from neural_analysis.matIO import loadmat
import numpy as np
import os
import pandas as pd
import sys
import time
from tqdm.auto import tqdm

sys.path.append('../..')
sys.path.append('/om2/user/eisenaj/code/ChaoticConsciousness')
from utils import save

def slds_eigs_worker(param_tuple, savefile=True, verbose=False):

    (start_time, start_step, data, transitions, emissions_dim, n_disc_states, latent_dim, data_dir) = param_tuple
    
    results = slds_compute_eigs(data, transitions, emissions_dim, n_disc_states, latent_dim, verbose)
    results['start_time'] = start_time
    results['start_step'] = start_step

    if savefile:
        save(results, os.path.join(data_dir, f"start_time_{start_time}"))
    else:
        return results

def main():

    # ============================================
    # LOAD DATA
    # ============================================

    local = True

    if local:
        # filename = '../../__data__/Mary-Anesthesia-20160809-01.mat'
        filename = r'/home/adameisen/millerdata/common/datasets/anesthesia/mat/propofolPuffTone/Mary-Anesthesia-20160809-01.mat'
        # filename = r'/home/adameisen/common/datasets/anesthesia/mat/propofolWakeUp/Mary-Anesthesia-20170203-02.mat'
    else:
        filename = r'/om/user/eisenaj/ChaoticConsciousness/data/propofolPuffTone/Mary-Anesthesia-20160809-01.mat'
    print("Loading data ...")
    start = time.process_time()
    # electrode_info, lfp, lfp_schema, session_info, spike_times, unit_info = loadmat(filename, variables=['electrodeInfo', 'lfp', 'lfpSchema', 'sessionInfo', 'spikeTimes', 'unitInfo'], verbose=False)
    electrode_info, lfp, lfp_schema, session_info, unit_info = loadmat(filename, variables=['electrodeInfo', 'lfp', 'lfpSchema', 'sessionInfo', 'unitInfo'], verbose=False)
    # spike_times = spike_times[0]
    dt = lfp_schema['smpInterval'][0]
    T = lfp.shape[0]
    print(f"Data loaded (took {time.process_time() - start:.2f} seconds)")

    # ============================================
    # PARAMETERS
    # ============================================

    # --------
    # User-guided SLDS parameters
    # --------
    latent_dim = 2 # number of latent dimensions
    # transitions = "standard" # transition class
    transitions = "standard"
    stride = 5 # s
    duration = 5 # s
    # stride = 2000
    # duration = 0.2

    length = int(duration/dt)
    start_times = np.arange(0, lfp.shape[0]*dt - duration + 0.1, stride).astype(int)

    # areas = ['vlPFC', 'FEF', 'CPB', '7b']
    # areas = np.unique(electrode_info['area'])
    areas = ['vlPFC']
    unit_indices = np.arange(lfp.shape[1])[pd.Series(electrode_info['area']).isin(areas)]
    var_names = [f"unit_{unit_num} {electrode_info['area'][unit_num]}" for unit_num in unit_indices]


    # --------
    # Process parameters
    # --------
    multi_process = True

    # --------
    # Set the parameters of the SLDS
    # --------
    emissions_dim = len(unit_indices)      # number of observed dimensions
    
    # --------
    # Data Directory
    # --------
    
    if local:
        data_dir = "../../__data__/slds/"
    else:
        data_dir = "/om/user/eisenaj/ChaoticConsciousness/results"
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    data_dir = os.path.join(data_dir, f"slds_big_run_latent_{latent_dim}_duration_{duration}_stride_{stride}_{timestamp}")
    os.makedirs(data_dir, exist_ok=True)

    run_params = dict(
        duration=duration,
        dt=dt,
        stride=stride,
        length=length,
        var_names=var_names,
        transitions=transitions,
        emissions_dim=emissions_dim,
        latent_dim=latent_dim,
        unit_indices=unit_indices,
    )
    save(run_params, os.path.join(data_dir, f'run_params'))

    # ============================================
    # RUN
    # ============================================

    # --------
    # Make Param List
    # --------

    anesthesia_bounds = [session_info['drugStart'][0], session_info['drugEnd'][1]]
    param_list = []
    for start_time in start_times:

        # --------
        # Set Disc States for Each Segment
        # --------
        piece_bounds = [start_time, start_time + duration]
        if piece_bounds[1] <= anesthesia_bounds[0] or piece_bounds[0] >= anesthesia_bounds[1]:
            # WAKEFUL
            n_disc_states = 2
        elif piece_bounds[1] > anesthesia_bounds[0] and piece_bounds[1] <= anesthesia_bounds[1]:
            if piece_bounds[0] < anesthesia_bounds[0]:
                # TRANSITION TO ANESTHESIA
                n_disc_states = 2
            else: # piece_bounds[0] >= anesthesia_bounds
                # FULL ANESTHESIA
                n_disc_states = 2
        else: # piece_bounds[0] > anesthesia_bounds[1] and piece_bounds[1] > anesthesia_bounds[1]
            # TRANSITION OUT OF ANESTHESIA
            n_disc_states = 2

        start_step = int(start_time/dt)
        data = lfp[start_step:start_step + length, unit_indices]
        param_list.append((start_time, start_step, data, transitions, emissions_dim, n_disc_states, latent_dim, data_dir))

    # --------
    # Process
    # --------

    if multi_process:
        PROCESSES = os.cpu_count() - 2
        with multiprocessing.Pool(PROCESSES) as pool:
            list(tqdm(pool.imap(slds_eigs_worker, param_list), total=len(param_list)))

    else:
        results = []
        for param_tuple in tqdm(param_list):
            slds_eigs_worker(param_tuple)

    # save(results, os.path.join(data_dir, f'slds_big_run_full_brain_{timestamp}_{duration}s_duration_{stride}s_stride'))

if __name__ == '__main__':
    main()
