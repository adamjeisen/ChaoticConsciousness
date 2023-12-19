import hydra
import logging
import numpy as np
from omegaconf import DictConfig
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
import sys

from data_utils import get_data_class, load_session_data, load_window_from_chunks

log = logging.getLogger("LTSA logger")

def perform_dim_reduction(signal, pca_dims, delay_p, delay_tau, standardize, subsample, ltsa_n_components, n_neighbors_pct, device='cuda'):
    if pca_dims >= 0:
        log.info("performing PCA")
        signal = PCA(n_components=pca_dims).fit_transform(signal)
        log.info("finished PCA")
        
    # delay
    sys.path.append('/om2/user/eisenaj/code/DeLASE')
    from dmd import embed_signal_torch
    signal = embed_signal_torch(signal, delay_p, delay_tau)
    
    # standardize
    if standardize:
        signal = (signal - signal.mean())/signal.std()
    
    # subsample
    signal = signal[np.arange(0, signal.shape[0], subsample)]
    
    # manifold embed
    log.info("performing LTSA")
    n_neighbors = int((n_neighbors_pct/100) * signal.shape[0])
    log.info(f"signal shape = {signal.shape}")
    embed = LocallyLinearEmbedding(method="ltsa", n_components=ltsa_n_components, n_neighbors=n_neighbors)
    embedded_signal = embed.fit_transform(signal)
    log.info("LTSA complete")
    
    return embedded_signal

@hydra.main(config_path='/om2/user/eisenaj/code/ChaoticConsciousness/conf', config_name='config', version_base='1.3')
def main(cfg: DictConfig):
    
    session = cfg.ltsa_dim_reduction.params.session
    pca_dims = cfg.ltsa_dim_reduction.params.pca_dims
    delay_p = cfg.ltsa_dim_reduction.params.delay_p
    delay_tau = cfg.ltsa_dim_reduction.params.delay_tau
    standardize = cfg.ltsa_dim_reduction.params.standardize
    subsample = cfg.ltsa_dim_reduction.params.subsample
    ltsa_n_components = cfg.ltsa_dim_reduction.params.ltsa_n_components
    n_neighbors_pct = cfg.ltsa_dim_reduction.params.n_neighbors_pct

    start_time = cfg.ltsa_dim_reduction.params.start_time # s
    window = cfg.ltsa_dim_reduction.params.window # s

    log.info(f"Process ID {os.getpid()} executing task")
    log.info(f"session = {session}")
    log.info(f"pca_dims = {pca_dims}")
    log.info(f"delay_p = {delay_p}")
    log.info(f"delay_tau = {delay_tau}")
    log.info(f"standardize = {standardize}")
    log.info(f"subsample = {subsample}")
    log.info(f"n_neighbors_pct = {n_neighbors_pct}")
    log.info(f"window = {window}")
    log.info(f"start_time = {start_time:.3f}")

    # -------------------------------------------------
    # LOAD SESSION DATA
    # -------------------------------------------------

    all_data_dir = '/scratch2/weka/millerlab/eisenaj/datasets/anesthesia/mat'
    data_class = get_data_class(session, all_data_dir)

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    variables = ['electrodeInfo', 'lfpSchema', 'sessionInfo', 'spikeTimes', 'trialInfo', 'unitInfo']
    session_vars, T, N, dt = load_session_data(session, all_data_dir, variables, data_class=data_class, verbose=False)
    electrode_info, lfp_schema, session_info, spike_times, trial_info, unit_info = session_vars['electrodeInfo'], session_vars['lfpSchema'], session_vars['sessionInfo'], session_vars['spikeTimes'], session_vars['trialInfo'], session_vars['unitInfo']

    eyes_open = session_info['eyesOpen'][-1] if isinstance(session_info['eyesOpen'], np.ndarray) else session_info['eyesOpen']
    eyes_close = session_info['eyesClose'][-1] if isinstance(session_info['eyesClose'], np.ndarray) else session_info['eyesClose']

    section_times = dict( 
        wake=(0, session_info['drugStart'][0]),
        induction=(session_info['drugStart'][0], eyes_close),
        anesthesia=(eyes_close, session_info['drugEnd'][1]),
        recovery=(session_info['drugEnd'][1], T*dt)
    )
    sections = list(section_times.keys())

    tone_on = trial_info['cpt_toneOn'][~np.isnan(trial_info['cpt_toneOn'])]
    tone_off = trial_info['cpt_toneOff'][~np.isnan(trial_info['cpt_toneOff'])]
    puff_on = trial_info['cpt_puffOn'][~np.isnan(trial_info['cpt_puffOn'])]

    log.info("finished loading session data")

    # -------------------------------------------------
    # PREP LFP DATA
    # -------------------------------------------------

    dir_ = f"/scratch2/weka/millerlab/eisenaj/datasets/anesthesia/mat/propofolPuffTone/{session}_lfp_chunked_20s"
    directory = pd.read_pickle(os.path.join(dir_, "directory"))

    # window = 15 # s
    # window = 30 # s
    # start_wake = 2500 # s
    # start_anesthesia = session_info['drugStart'][1] - 500 # s

    areas_to_include = ['vlPFC', 'FEF', '7b', 'CPB']
    electrode_indices = np.array([False]*len(electrode_info['area']))
    for area in areas_to_include:
        electrode_indices = np.logical_or(electrode_indices, electrode_info['area'] == area)
    electrode_indices = np.where(electrode_indices)[0]

    # # ANALYZE TIMES
    # if 'Mary' in session:
    #     start_wake = (548.37086667 - 1.37)
    # else:
    # #     shift_w = (2576.19616667 - 1.37) - start_wake
    #     start_wake = (2672.6031 - 1.37)
        
    # #     shift_w = (559.23773333 - 1.37) - start_wake
    # # shift_w = (tone_on[60] - 1.37) - start_wake
    # log.info(f"start_wake = {start_wake}")
    # log.info(f"tone times = {tone_on[(tone_on >= start_wake) & (tone_on <= start_wake + window)]}")
    # log.info(f"puff times = {puff_on[(puff_on >= start_wake) & (puff_on <= start_wake + window)]}")

    # if 'Mary' in session:
    #     start_anesthesia = (1969.6065 - 1.37)
    # else:
    #     start_anesthesia = (5022.42016667 - 1.37)

    # # shift_a = (tone_on[373] - 1.37) - start_anesthesia
    # log.info(f"start_anesthesia = {start_anesthesia}")
    # log.info(f"tone times = {tone_on[(tone_on >= start_anesthesia) & (tone_on <= start_anesthesia + window)]}")
    # log.info(f"puff times = {puff_on[(puff_on >= start_anesthesia) & (puff_on <= start_anesthesia + window)]}")

    # log.info("finished prepping LFP data")

    # -------------------------------------------------
    # ANALYSIS
    # -------------------------------------------------

    # SAVE TO CUSTOM FOLDER
    save_dir = f'/scratch2/weka/millerlab/eisenaj/ChaoticConsciousness/ltsa_dim_reduction_multirun/{session}'
    hyper_param_folder = f"pca_dims={pca_dims}_delay_p={delay_p}_delay_tau={delay_tau}_standardize={standardize}_subsample={subsample}_ltsa_n_components={ltsa_n_components}_n_neighbors_pct={n_neighbors_pct}_window={window}"
    save_dir = os.path.join(save_dir, hyper_param_folder)
    
    # # SAVE TO HYDRA FOLDER
    # save_dir = os.path.join(os.getcwd(), 'saved_files')

    os.makedirs(save_dir, exist_ok=True)

    # start_times = [start_wake, start_anesthesia]

    # for t in start_times:
    log.info(f"now computing time {start_time:.3f}")
    save_path = os.path.join(save_dir, f"{start_time:.3f}.pkl") 
    if not os.path.exists(save_path):
        signal = load_window_from_chunks(start_time, start_time + window, directory, electrode_indices)
        embedded_signal = perform_dim_reduction(signal, pca_dims, delay_p, delay_tau, standardize, subsample, ltsa_n_components, n_neighbors_pct)
        pd.to_pickle(embedded_signal, save_path)
    else:
        log.info(f"time {start_time:.3f} already computed!") 

        # signal = load_window_from_chunks(t, t + window, directory, electrode_indices)
        # embedded_signal = perform_dim_reduction(signal, pca_dims, delay_p, delay_tau, standardize, subsample, n_neighbors_pct)
        # pd.to_pickle(embedded_signal, os.path.join(save_dir, f"{t:.3f}.pkl"))

if __name__ == '__main__':
    main()