from matplotlib import colors
import matplotlib.pyplot as plt
import multiprocessing
from neural_analysis.matIO import loadmat
import numpy as np
import os
import pandas as pd
import ssm
from ssm.util import find_permutation
import sys
import time
from tqdm.auto import tqdm

sys.path.append('../..')
from nld_utils import simulate_lorenz
from up_down import get_up_down
from utils import get_sample_interval, load, save

def slds_eigs_worker(param_tuple):

    (start_time, start_step, data, transitions, emissions_dim, n_disc_states, latent_dim, data_dir) = param_tuple
    
    # Create the model and initialize its parameters
    slds = ssm.SLDS(emissions_dim, n_disc_states, latent_dim, transitions=transitions, emissions="gaussian_orthog", verbose=False)

    # Fit the model using Laplace-EM with a structured variational posterior
    q_lem_elbos, q_lem = slds.fit(data, method="laplace_em",
                                   variational_posterior="structured_meanfield",
                                   num_iters=10, alpha=0.0, verbose=False)
     
    criticality_inds = np.zeros((n_disc_states, latent_dim))
    eigs = np.zeros((n_disc_states, latent_dim), dtype='complex')
    for i in range(n_disc_states):
        eigs[i] = np.linalg.eig(slds.dynamics._As[i])[0]
        criticality_inds[i] = np.abs(eigs[i])
        criticality_inds[i].sort()
        criticality_inds[i] = criticality_inds[i][::-1]
    
    q_lem_x = q_lem.mean_continuous_states[0]
    preds = slds.smooth(q_lem_x, data)
    mse = ((preds - data)**2).mean()
    
    disc_states = slds.most_likely_states(q_lem_x, data)

    results = dict(
        start_time=start_time,
        n_disc_states=n_disc_states,
        start_step=start_step,
        slds=slds,
        q_lem_elbos=q_lem_elbos,
        q_lem=q_lem,
        eigs=eigs,
        criticality_inds=criticality_inds,
        mse=mse,
        disc_states=disc_states
    )

    save(results, os.path.join(data_dir, f"start_time_{start_time}"))

def main():

    # ============================================
    # LOAD DATA
    # ============================================

    local = False

    if local:
        # filename = '../../__data__/Mary-Anesthesia-20160809-01.mat'
        filename = r'/home/adameisen/millerdata/common/datasets/anesthesia/mat/propofolPuffTone/Mary-Anesthesia-20160809-01.mat'
        # filename = r'/home/adameisen/common/datasets/anesthesia/mat/propofolWakeUp/Mary-Anesthesia-20170203-02.mat'
    else:
        filename = r'/om/user/eisenaj/ChaoticConsciousness/data/propofolPuffTone/Mary-Anesthesia-20160809-01.mat'
    print("Loading data ...")
    start = time.process_time()
    electrode_info, lfp, lfp_schema, session_info, spike_times, unit_info = loadmat(filename, variables=['electrodeInfo', 'lfp', 'lfpSchema', 'sessionInfo', 'spikeTimes', 'unitInfo'], verbose=False)
    spike_times = spike_times[0]
    dt = lfp_schema['smpInterval'][0]
    T = lfp.shape[0]
    print(f"Data loaded (took {time.process_time() - start:.2f} seconds)")

    # ============================================
    # PARAMETERS
    # ============================================

    # --------
    # User-guided SLDS parameters
    # --------
    n_disc_states = 2      # number of discrete states
    latent_dim = 10 # number of latent dimensions
    # transitions = "standard" # transition class
    transitions = "recurrent_only"
    # stride = 10*60 # s
    # duration = 10*60 # s
    stride = 2000
    duration = 0.2

    length = int(duration/dt)
    start_times = np.arange(0, lfp.shape[0]*dt - duration + 0.1, stride).astype(int)

    # --------
    # Process parameters
    # --------
    multi_process = True

    # --------
    # Set the parameters of the SLDS
    # --------
    emissions_dim = lfp.shape[1]      # number of observed dimensions
    
    # areas = ['vlPFC', 'FEF', 'CPB', '7b']
    areas = np.unique(electrode_info['area'])
    unit_indices = np.arange(lfp.shape[1])[pd.Series(electrode_info['area']).isin(areas)]
    var_names = [f"unit_{unit_num} {electrode_info['area'][unit_num]}" for unit_num in unit_indices]

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
        stride=stride,
        length=length,
        var_names=var_names,
        transitions=transitions,
        emissions_dim=emissions_dim,
        n_disc_states=n_disc_states,
        latent_dim=latent_dim,
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
        PROCESSES = os.cpu_count() - 1
        with multiprocessing.Pool(PROCESSES) as pool:
            list(tqdm(pool.imap(slds_eigs_worker, param_list), total=len(param_list)))

    else:
        results = []
        for param_tuple in tqdm(param_list):
            slds_eigs_worker(param_tuple)

    # save(results, os.path.join(data_dir, f'slds_big_run_full_brain_{timestamp}_{duration}s_duration_{stride}s_stride'))

if __name__ == '__main__':
    main()