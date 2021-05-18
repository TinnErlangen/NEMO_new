import mne
from mne import io
from mne.datasets import sample
from mne.event import make_fixed_length_events
from mne.beamformer import tf_dics
import numpy as np

print(__doc__)

## LOAD FILES FROM SAMPLE DATASET

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
subjects_dir = data_path + '/subjects'
label_name = 'Aud-lh'
fname_label = data_path + '/MEG/sample/labels/%s.label' % label_name

## PREP EPOCHS LIKE IN TF_DICS EXAMPLE

raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.info['bads'] = ['MEG 2443']  # 1 bad MEG channel
# Pick a selection of magnetometer channels. A subset of all channels was used
# to speed up the example. For a solution based on all MEG channels use
# meg=True, selection=None and add mag=4e-12 to the reject dictionary.
left_temporal_channels = mne.read_selection('Left-temporal')
picks = mne.pick_types(raw.info, meg='mag', eeg=False, eog=False,
                       stim=False, exclude='bads',
                       selection=left_temporal_channels)
raw.pick_channels([raw.ch_names[pick] for pick in picks])
reject = dict(mag=4e-12)
# Re-normalize our empty-room projectors, which should be fine after subselection
raw.info.normalize_proj()
# Setting time windows.
tmin, tmax, tstep = -0.5, 0.75, 0.05
# Read epochs
event_id = 1
events = mne.read_events(event_fname)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    baseline=None, preload=True, proj=True, reject=reject)
# Read forward operator
forward = mne.read_forward_solution(fname_fwd)
# Read label
label = mne.read_label(fname_label)


# CWT_MORLET exemplary PARAMETERS for TF-DICS beamformer to re-produce bug/error

frequencies = [[4.,5.,6.,7.,8.,9.,10.,11.,12.],[12.,13.,14.,15.,16.,17.,18.,19.,20.,21.,22.,23.,24.,25.,26.,27.,28.,29.,30.]]
win_lengths = [0.3, 0.2]
# use this line for cwt_cycles as list of float
cwt_n_cycles = [5.,7.]
# use this line to test single value
# cwt_n_cycles = 7.


# TF_DICS CALCULATION

stcs = tf_dics(epochs, forward, noise_csds = None, tmin=tmin, tmax=tmax, tstep=tstep, win_lengths=win_lengths, subtract_evoked=False, mode='cwt_morlet', frequencies=frequencies,
               reg = 0.05, label=label, pick_ori='max-power', inversion='single', depth=1.0)
