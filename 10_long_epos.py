#step 2 C again - creating long epochs for experiment blocks including all pics and the long tone for each trial

import mne
import numpy as np

# define file locations
preproc_dir = "D:/NEMO_analyses_new/preproc/"
proc_dir = "D:/NEMO_analyses_new/proc/"
# pass subject and run lists
subjs_all = ["NEM_10","NEM_11","NEM_12","NEM_14","NEM_15","NEM_16",
        "NEM_17","NEM_18","NEM_19","NEM_20","NEM_21","NEM_22",
        "NEM_23","NEM_24","NEM_26","NEM_27","NEM_28",
        "NEM_29","NEM_30","NEM_31","NEM_32","NEM_33","NEM_34",
        "NEM_35","NEM_36","NEM_37"]
excluded = ["NEM_19","NEM_21","NEM_30","NEM_32","NEM_33","NEM_37"]
subjs = ["NEM_10","NEM_11","NEM_12","NEM_14","NEM_15",
         "NEM_16","NEM_17","NEM_18","NEM_20","NEM_22",
         "NEM_23","NEM_24","NEM_26","NEM_27","NEM_28",
         "NEM_29","NEM_31","NEM_34","NEM_35","NEM_36"]
# subjs = ["NEM_10"]
runs = ["3","4"] # runs 3,4 = blocks A+B of experiment

# dictionaries with conditions/triggers
event_id_orig = {'negative/pic1': 10, 'positive/pic1': 20, 'negative/pic2': 30,
                 'positive/pic2': 40, 'negative/pic3': 50, 'positive/pic3': 60,
                 'negative/r1/part1': 110, 'negative/r1/part2': 111, 'negative/r1/part3': 112, 'negative/r1/part4': 113,
                 'positive/r1/part1': 120, 'positive/r1/part2': 121, 'positive/r1/part3': 122, 'positive/r1/part4': 123,
                 'negative/r2/part1': 130, 'negative/r2/part2': 131, 'negative/r2/part3': 132, 'negative/r2/part4': 133,
                 'positive/r2/part1': 140, 'positive/r2/part2': 141, 'positive/r2/part3': 142, 'positive/r2/part4': 143,
                 'negative/s1/part1': 150, 'negative/s1/part2': 151, 'negative/s1/part3': 152, 'negative/s1/part4': 153,
                 'positive/s1/part1': 160, 'positive/s1/part2': 161, 'positive/s1/part3': 162, 'positive/s1/part4': 163,
                 'negative/s2/part1': 170, 'negative/s2/part2': 171, 'negative/s2/part3': 172, 'negative/s2/part4': 173,
                 'positive/s2/part1': 180, 'positive/s2/part2': 181, 'positive/s2/part3': 182, 'positive/s2/part4': 183}
# only the picture starts are needed here
event_id = {'negative/pic1': 10, 'positive/pic1': 20}

# set parameters
baseline = None
tmin = -0.2
tmax = 11.8

#
for sub in subjs:
    #loading raw data and original events per run, cut epochs and save
    raw_3 = mne.io.Raw('{}{}_3_ica-raw.fif'.format(preproc_dir,sub))
    events_3 = list(np.load('{}nc_{}_3_events.npy'.format(preproc_dir,sub)))
    epochs_3 = mne.Epochs(raw_3,events_3,event_id=event_id,baseline=baseline,picks=['meg'],tmin=tmin,tmax=tmax,preload=True)
    epochs_3.save('{}{}_3_long-epo.fif'.format(preproc_dir,sub),overwrite=True)
    print(epochs_3)
    raw_4 = mne.io.Raw('{}{}_4_ica-raw.fif'.format(preproc_dir,sub))
    events_4 = list(np.load('{}nc_{}_4_events.npy'.format(preproc_dir,sub)))
    epochs_4 = mne.Epochs(raw_4,events_4,event_id=event_id,baseline=baseline,picks=['meg'],tmin=tmin,tmax=tmax,preload=True)
    epochs_4.save('{}{}_4_long-epo.fif'.format(preproc_dir,sub),overwrite=True)
    print(epochs_4)
    # equalize bads and append all experiment epochs, save
    bads = epochs_3.info['bads'] + epochs_4.info['bads']
    epochs_3.info['bads'] = bads
    epochs_4.info['bads'] = bads
    print(epochs_4.info['bads'])
    # override head_position data to append sensor data
    epochs_4.info['dev_head_t'] = epochs_3.info['dev_head_t']
    epo_all = mne.concatenate_epochs([epochs_3,epochs_4])
    epo_all.save("{}{}_long-epo.fif".format(proc_dir,sub),overwrite=True)
