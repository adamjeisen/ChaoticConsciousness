import hydra
import logging
import numpy as np
from omegaconf import DictConfig
import os
import pandas as pd
import sys

from data_utils import get_data_class, load_session_data, load_window_from_chunks



log = logging.getLogger("Hyperparameter logger")

def perform_dim_reduction(train_signal, test_signal, matrix_size, rank, dt, N_time_bins=None, max_freq=500, max_unstable_freq=125, device='cpu'):
        
    # delay
    sys.path.append('/om2/user/eisenaj/code/DeLASE')
    from delase import DeLASE

    if N_time_bins is None:
        # N_time_bins = np.max([int(np.ceil(train_signal.shape[0]/matrix_size)), 20])
        pass
    
    delase_model = DeLASE(
                    train_signal, 
                    matrix_size=matrix_size,
                    rank=rank,
                    dt=dt,
                    N_time_bins=N_time_bins,
                    max_freq=max_freq,
                    max_unstable_freq=max_unstable_freq,
                    device=device
                )      

    delase_model.fit()

    from performance_metrics import compute_AIC
    aic_val = compute_AIC(delase_model, test_signal)
    
    return dict(
        stability_params=delase_model.stability_params,
        stability_freqs=delase_model.stability_freqs,
        aic_val=aic_val
    )

@hydra.main(config_path='/om2/user/eisenaj/code/ChaoticConsciousness/conf', config_name='config', version_base='1.3')
def main(cfg: DictConfig):
    
    session = cfg.hyperparameter_testing.params.session
    matrix_size = cfg.hyperparameter_testing.params.matrix_size
    rank = cfg.hyperparameter_testing.params.rank
    window = cfg.hyperparameter_testing.params.window # s
    T_pred = cfg.hyperparameter_testing.params.T_pred # s
    start_time = cfg.hyperparameter_testing.params.start_time # s

    log.info(f"Process ID {os.getpid()} executing task")
    log.info(f"session = {session}")
    log.info(f"matrix_size = {matrix_size}")
    log.info(f"rank = {rank}")
    log.info(f"window = {window}")
    log.info(f"T_pred = {T_pred}")
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

    if rank > int(np.ceil(matrix_size/N))*N:
        log.info(f"Rank {rank} is greater than needed matrix size {N*int(np.ceil(matrix_size/N))} therefore skipping!")
        sys.exit()
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

    areas_to_include = ['vlPFC', 'FEF', '7b', 'CPB']
    electrode_indices = np.array([False]*len(electrode_info['area']))
    for area in areas_to_include:
        electrode_indices = np.logical_or(electrode_indices, electrode_info['area'] == area)
    electrode_indices = np.where(electrode_indices)[0]

    # -------------------------------------------------
    # ANALYSIS
    # -------------------------------------------------

    # SAVE TO CUSTOM FOLDER
    save_dir = f'/scratch2/weka/millerlab/eisenaj/ChaoticConsciousness/hyperparameter_testing/{session}'
    hyper_param_folder = f"matrix_size={matrix_size}_rank={rank}_window={window}_T_pred={T_pred}"
    save_dir = os.path.join(save_dir, hyper_param_folder)
    
    # # SAVE TO HYDRA FOLDER
    os.makedirs(save_dir, exist_ok=True)

    log.info(f"now computing time {start_time:.3f}")
    save_path = os.path.join(save_dir, f"{start_time:.3f}.pkl") 
    if not os.path.exists(save_path):
        signal = load_window_from_chunks(start_time, start_time + window + T_pred, directory, electrode_indices)
        train_signal = signal[:int(window/dt)]
        test_signal = signal[int(window/dt):]
        ret_dict = perform_dim_reduction(train_signal, test_signal, matrix_size, rank, dt)
        pd.to_pickle(ret_dict, save_path)
    else:
        log.info(f"time {start_time:.3f} already computed!") 

if __name__ == '__main__':
    main()