#step 2 C - creating epochs for experiment blocks

import mne
import numpy as np

# define file locations
proc_dir = "D:/NEMO_analyses_new/preproc/"
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
event_id = {'negative/pics': 70, 'positive/pics': 80,
            'negative/r1/part1': 110, 'negative/r1/part2': 111, 'negative/r1/part3': 112, 'negative/r1/part4': 113,
            'positive/r1/part1': 120, 'positive/r1/part2': 121, 'positive/r1/part3': 122, 'positive/r1/part4': 123,
            'negative/r2/part1': 130, 'negative/r2/part2': 131, 'negative/r2/part3': 132, 'negative/r2/part4': 133,
            'positive/r2/part1': 140, 'positive/r2/part2': 141, 'positive/r2/part3': 142, 'positive/r2/part4': 143,
            'negative/s1/part1': 150, 'negative/s1/part2': 151, 'negative/s1/part3': 152, 'negative/s1/part4': 153,
            'positive/s1/part1': 160, 'positive/s1/part2': 161, 'positive/s1/part3': 162, 'positive/s1/part4': 163,
            'negative/s2/part1': 170, 'negative/s2/part2': 171, 'negative/s2/part3': 172, 'negative/s2/part4': 173,
            'positive/s2/part1': 180, 'positive/s2/part2': 181, 'positive/s2/part3': 182, 'positive/s2/part4': 183}
trig_id = {v: k for k,v in event_id.items()}   # this reverses the dictionary and will be useful later

mini_epochs_num = 4
mini_epochs_len = 2

for sub in subjs:
    for run in runs:
        #loading raw data and original events
        raw = mne.io.Raw('{}{}_{}_ica-raw.fif'.format(proc_dir,sub,run))
        events = list(np.load('{}nc_{}_{}_events.npy'.format(proc_dir,sub,run)))
        #creating new event list with slices/starting time points for epochs
        new_events = []
        for e in events:
            if e[2] > 100:
                if e[2] in [110,130,150,170]:
                    new_events.append(np.array([e[0]-mini_epochs_len*raw.info["sfreq"], 0, 70]))
                if e[2] in [120,140,160,180]:
                    new_events.append(np.array([e[0]-mini_epochs_len*raw.info["sfreq"], 0, 80]))
                for me in range(mini_epochs_num):
                    new_events.append(np.array(
                    [e[0]+me*mini_epochs_len*raw.info["sfreq"], 0, e[2]+me]))
        new_events = np.array(new_events).astype(int)
        #check if events alright
        print(new_events[:30,:])
        print(len(new_events))
        print(np.unique(new_events[:,2]))

        # #creating Epoch object from new event list
        epochs = mne.Epochs(raw,new_events,event_id=event_id,baseline=None,picks=['meg'],tmin=0,tmax=mini_epochs_len,preload=True)
        #check epochs and labels
        print(epochs.event_id)
        print(epochs.events[:12])
        print(epochs[1:3])
        print(epochs['negative/r1'])
        print(epochs['part3'])
        print(epochs.drop_log)
        print(len(epochs.drop_log))
        # saving to epoch file
        epochs.save('{}{}_{}-epo.fif'.format(proc_dir,sub,run),overwrite=True)

        # look at them (optional check)
        # epochs.plot(n_epochs=10,n_channels=90,picks=['meg'],scalings=dict(mag=2e-12),event_id=trig_id)
        # epochs.plot_psd(fmax=95,bandwidth=1,average=False)
        # epochs.plot_psd_topomap()
