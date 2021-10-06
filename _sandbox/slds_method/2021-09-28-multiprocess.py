import h5py
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

def slds_eigs_worker(start_time):
    # ------------------
    # get signal
    # ------------------
    start_step = int(start_time/dt)
    length = int(duration/dt)
    data = lfp[start_step:start_step + length, unit_indices]
    var_names = [f"unit_{unit_num} {electrode_info['area'][unit_num]}" for unit_num in unit_indices]
    time_vals = np.arange(start_time, start_time+duration, dt)

    # Create the model and initialize its parameters
    slds = ssm.SLDS(emissions_dim, n_disc_states, latent_dim, emissions="gaussian_orthog", verbose=2)

    # Fit the model using Laplace-EM with a structured variational posterior
    q_lem_elbos, q_lem = slds.fit(data, method="laplace_em",
                                   variational_posterior="structured_meanfield",
                                   num_iters=10, alpha=0.0, verbose=2)
     
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

    return dict(
        start_time=start_time,
        duration=duration,
        start_step=start_step,
        length=length,
        unit_indices=unit_indices,
        areas=areas,
        slds=slds,
        q_lem_elbos=q_lem_elbos,
        q_lem=q_lem,
        eigs=eigs,
        criticality_inds=criticality_inds,
        mse=mse,
        disc_states=disc_states
    )

if __name__ == '__main__':
    # filename = '../../__data__/Mary-Anesthesia-20160809-01.mat'
    filename = r'/home/adameisen/millerdata/common/datasets/anesthesia/mat/propofolPuffTone/Mary-Anesthesia-20160809-01.mat'
    # filename = r'/home/adameisen/common/datasets/anesthesia/mat/propofolWakeUp/Mary-Anesthesia-20170203-02.mat'
    print("Loading data ...")
    start = time.process_time()
    electrode_info, lfp, lfp_schema, session_info, spike_times, unit_info = loadmat(filename, variables=['electrodeInfo', 'lfp', 'lfpSchema', 'sessionInfo', 'spikeTimes', 'unitInfo'], verbose=False)
    spike_times = spike_times[0]
    dt = lfp_schema['smpInterval'][0]
    T = lfp.shape[0]
    print(f"Data loaded (took {time.process_time() - start:.2f} seconds)")

    # Set the parameters of the SLDS
    n_disc_states = 2      # number of discrete states
    latent_dim = 2 # number of latent dimensions
    emissions_dim = lfp.shape[1]      # number of observed dimensions

    # areas = ['vlPFC', 'FEF', 'CPB', '7b']
    areas = np.unique(electrode_info['area'])
    unit_indices = np.arange(lfp.shape[1])[pd.Series(electrode_info['area']).isin(areas)]
    stride = 5*60 # s
    duration = 10*60 # s

    start_times = np.arange(0, lfp.shape[0]*dt - duration + 0.1, stride).astype(int)

    PROCESSES = os.cpu_count() - 1
    with multiprocessing.Pool(PROCESSES) as pool:
        results = list(tqdm(pool.imap(slds_eigs_worker, start_times), total=len(start_times)))

    data_dir = "../../__data__/slds/"
    os.makedirs(data_dir, exist_ok=True)
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    save(results, os.path.join(data_dir, f'slds_big_run_full_brain_{timestamp}_{duration}s_duration_{stride}s_stride'))