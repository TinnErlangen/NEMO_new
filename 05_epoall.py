## now append all epoch files per subject, to have all conditions together and to compute CSDs

import mne
import matplotlib.pyplot as plt
import numpy as np

plt.ion() #this keeps plots interactive

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


event_id = {'rest': 220,
            'tonbas/r1/part1': 191,'tonbas/r2/part1': 192, 'tonbas/s1/part1': 193, 'tonbas/s2/part1': 194,
            'tonbas/r1/part2': 291,'tonbas/r2/part2': 292, 'tonbas/s1/part2': 293, 'tonbas/s2/part2': 294,
            'tonbas/r1/part3': 391,'tonbas/r2/part3': 392, 'tonbas/s1/part3': 393, 'tonbas/s2/part3': 394,
            'tonbas/r1/part4': 491,'tonbas/r2/part4': 492, 'tonbas/s1/part4': 493, 'tonbas/s2/part4': 494,
            'tonbas/r1/part5': 591,'tonbas/r2/part5': 592, 'tonbas/s1/part5': 593, 'tonbas/s2/part5': 594,
            'tonrat/r1/part1': 691,'tonrat/r2/part1': 692, 'tonrat/s1/part1': 693, 'tonrat/s2/part1': 694,
            'tonrat/r1/part2': 791,'tonrat/r2/part2': 792, 'tonrat/s1/part2': 793, 'tonrat/s2/part2': 794,
            'tonrat/r1/part3': 891,'tonrat/r2/part3': 892, 'tonrat/s1/part3': 893, 'tonrat/s2/part3': 894,
            'tonrat/r1/part4': 991,'tonrat/r2/part4': 992, 'tonrat/s1/part4': 993, 'tonrat/s2/part4': 994,
            'negative/pics': 70, 'positive/pics': 80,
            'negative/r1/part1': 110, 'negative/r1/part2': 111, 'negative/r1/part3': 112, 'negative/r1/part4': 113,
            'positive/r1/part1': 120, 'positive/r1/part2': 121, 'positive/r1/part3': 122, 'positive/r1/part4': 123,
            'negative/r2/part1': 130, 'negative/r2/part2': 131, 'negative/r2/part3': 132, 'negative/r2/part4': 133,
            'positive/r2/part1': 140, 'positive/r2/part2': 141, 'positive/r2/part3': 142, 'positive/r2/part4': 143,
            'negative/s1/part1': 150, 'negative/s1/part2': 151, 'negative/s1/part3': 152, 'negative/s1/part4': 153,
            'positive/s1/part1': 160, 'positive/s1/part2': 161, 'positive/s1/part3': 162, 'positive/s1/part4': 163,
            'negative/s2/part1': 170, 'negative/s2/part2': 171, 'negative/s2/part3': 172, 'negative/s2/part4': 173,
            'positive/s2/part1': 180, 'positive/s2/part2': 181, 'positive/s2/part3': 182, 'positive/s2/part4': 183}
trig_id = {v: k for k,v in event_id.items()}   # this reverses the dictionary and will be useful later


for sub in subjs:
    # load and prepare the MEG data
    rest = mne.read_epochs("{}{}_1-epo.fif".format(preproc_dir,sub))
    ton = mne.read_epochs("{}{}_2-epo.fif".format(preproc_dir,sub))
    epo_a = mne.read_epochs("{}{}_3-epo.fif".format(preproc_dir,sub))
    epo_b = mne.read_epochs("{}{}_4-epo.fif".format(preproc_dir,sub))
    # override head_position data to append sensor data (just for calculating CSD !)
    rest.info['dev_head_t'] = epo_a.info['dev_head_t']
    ton.info['dev_head_t'] = epo_a.info['dev_head_t']
    epo_b.info['dev_head_t'] = epo_a.info['dev_head_t']
    epo_all = mne.concatenate_epochs([rest,ton,epo_a,epo_b])
    epo_all.save("{}{}-epo.fif".format(proc_dir,sub),overwrite=True)
