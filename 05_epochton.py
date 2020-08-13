#step 2 B - creating epochs for tone baseline block

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
subjs = ["NEM_10"]
runs = ["2"] # run 2 = tone baseline

#dictionary with conditions/triggers
event_id = {'tonbas/r1/part1': 191,'tonbas/r2/part1': 192, 'tonbas/s1/part1': 193, 'tonbas/s2/part1': 194,
            'tonbas/r1/part2': 291,'tonbas/r2/part2': 292, 'tonbas/s1/part2': 293, 'tonbas/s2/part2': 294,
            'tonbas/r1/part3': 391,'tonbas/r2/part3': 392, 'tonbas/s1/part3': 393, 'tonbas/s2/part3': 394,
            'tonbas/r1/part4': 491,'tonbas/r2/part4': 492, 'tonbas/s1/part4': 493, 'tonbas/s2/part4': 494,
            'tonbas/r1/part5': 591,'tonbas/r2/part5': 592, 'tonbas/s1/part5': 593, 'tonbas/s2/part5': 594,
            'tonrat/r1/part1': 691,'tonrat/r2/part1': 692, 'tonrat/s1/part1': 693, 'tonrat/s2/part1': 694,
            'tonrat/r1/part2': 791,'tonrat/r2/part2': 792, 'tonrat/s1/part2': 793, 'tonrat/s2/part2': 794,
            'tonrat/r1/part3': 891,'tonrat/r2/part3': 892, 'tonrat/s1/part3': 893, 'tonrat/s2/part3': 894,
            'tonrat/r1/part4': 991,'tonrat/r2/part4': 992, 'tonrat/s1/part4': 993, 'tonrat/s2/part4': 994}
trig_id = {v: k for k,v in event_id.items()}   # this reverses the dictionary and will be useful later

mini_epochs_num = (5,4)
mini_epochs_len = 2

for sub in subjs:
    for run in runs:
        #loading raw data and original events
        raw = mne.io.Raw('{}nc_{}_{}_m-raw.fif'.format(proc_dir,sub,run))
        events = list(np.load('{}nc_{}_{}_events.npy'.format(proc_dir,sub,run)))
        #creating new event list with slices/starting time points for epochs
        new_events = []
        for e in events[:16]:
            for me in range(mini_epochs_num[0]):
                new_events.append(np.array(
                [e[0]+me*mini_epochs_len*raw.info["sfreq"], 0, e[2]+me*100]))
        for e in events[16:]:
            for me in range(mini_epochs_num[1]):
                new_events.append(np.array(
                [e[0]+me*mini_epochs_len*raw.info["sfreq"], 0, e[2]+500+me*100]))
        new_events = np.array(new_events).astype(int)

        #check if events alright
        print(new_events[:25,:])
        print(len(new_events))
        print(np.unique(new_events[:,2]))

        #creating Epoch object from new event list
        epochs = mne.Epochs(raw,new_events,event_id=event_id,baseline=None,tmin=0,tmax=mini_epochs_len,preload=True)
        #check epochs and labels
        print(epochs.event_id)
        print(epochs.events[:12])
        print(epochs[1:3])
        print(epochs['tonbas/s1'])
        print(epochs['part4'])
        print(epochs.drop_log)
        print(len(epochs.drop_log))
        #saving to epoch file
        epochs.save('{}{}_{}-epo.fif'.format(proc_dir,sub,run))

        #look at them (optional check)
        # epochs.plot(n_epochs=8,n_channels=32)
        # epochs.plot_psd(fmax=50,average=False)
        # epochs.plot_psd_topomap()
