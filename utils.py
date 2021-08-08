import h5py
from neural_analysis.matIO import loadmat
import numpy as np
import pickle
from tqdm.notebook import tqdm

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

def get_stimuli_start_and_end_flags(f):
    airPuff_binary, audio_binary = get_binary_stimuli(f)
    stimuli = {}
    for stim_name, stim in [('AirPuff Binary', airPuff_binary), ('Audio Binary', audio_binary)]:
        stim_begin = (np.diff(stim) == 1).astype(np.int)
        stim_end = (np.diff(stim) == -1).astype(np.int)
        stimuli[stim_name + ' (From Start)'] = stim_begin
        stimuli[stim_name + ' (From End)'] = stim_end

    # both and separate
    tone_only_begin = np.zeros(len(stim))
    tone_only_end = np.zeros(len(stim))
    puff_only_begin = np.zeros(len(stim))
    puff_only_end = np.zeros(len(stim))
    both_begin = np.zeros(len(stim))
    both_end = np.zeros(len(stim))
    count = 0
    for t, (puffOn, toneOn) in enumerate(zip(f['trialInfo']['cpt_puffOn'][:, 0], f['trialInfo']['cpt_toneOn'][:, 0])):
        if np.isnan(puffOn):
            tone_only_begin[int(toneOn * 1000)] = 1
            tone_only_end[int(f['trialInfo']['cpt_toneOff'][t, 0] * 1000)] = 1
        elif np.isnan(toneOn):
            puff_only_begin[int(puffOn * 1000)] = 1
            puff_only_end[int(f['trialInfo']['cpt_puffOff'][t, 0] * 1000)] = 1
        else:  # both are on
            count += 1
            both_begin[int(np.min([toneOn, puffOn]) * 1000)] = 1
            both_end[int(np.max([f['trialInfo']['cpt_toneOff'][t, 0], f['trialInfo']['cpt_puffOff'][t, 0]]) * 1000)] = 1
    stimuli['Tone Only (From Start)'] = tone_only_begin
    stimuli['Tone Only (From End)'] = tone_only_end
    stimuli['Puff Only (From Start)'] = puff_only_begin
    stimuli['Puff Only (From End)'] = puff_only_end
    stimuli['Tone and Puff (From Start)'] = both_begin
    stimuli['Tone and Puff (From End)'] = both_end

    return stimuli

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