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

for sub in subjs:
    # load and prepare the MEG data
    rest = mne.read_epochs("{dir}{sub}_1-epo.fif".format(dir=proc_dir,sub=sub))
    ton = mne.read_epochs("{dir}{sub}_2-epo.fif".format(dir=proc_dir,sub=sub))
    epo_a = mne.read_epochs("{dir}{sub}_3-epo.fif".format(dir=proc_dir,sub=sub))
    epo_b = mne.read_epochs("{dir}{sub}_4-epo.fif".format(dir=proc_dir,sub=sub))
    # read bad channels and append to common list
    bads = rest.info['bads']
    for i in ton.info['bads']:
        if i not in bads:
            bads.append(i)
    for i in epo_a.info['bads']:
        if i not in bads:
            bads.append(i)
    for i in epo_b.info['bads']:
        if i not in bads:
            bads.append(i)
    # apply bads to all and save
    rest.info['bads'] = bads
    ton.info['bads'] = bads
    epo_a.info['bads'] = bads
    epo_b.info['bads'] = bads
    print(bads)
    rest.save("{dir}{sub}_1-epo.fif".format(dir=proc_dir,sub=sub),overwrite=True)
    ton.save("{dir}{sub}_2-epo.fif".format(dir=proc_dir,sub=sub),overwrite=True)
    epo_a.save("{dir}{sub}_3-epo.fif".format(dir=proc_dir,sub=sub),overwrite=True)
    epo_b.save("{dir}{sub}_4-epo.fif".format(dir=proc_dir,sub=sub),overwrite=True)
