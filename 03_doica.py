import mne
import matplotlib.pyplot as plt
import numpy as np

plt.ion() #this keeps plots interactive

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
runs = ["1","2","3","4"]
# runs = ["4"]

for sub in subjs:
    for run in runs:
        #load the annotated epoch file
        mraw = mne.io.Raw('{dir}nc_{sub}_{run}_m-raw.fif'.format(dir=proc_dir,sub=sub,run=run))
        mraw.info["bads"] += ["MRyA","MRyaA"] # add broken reference channels to bad channels list

        #define the ica for reference channels and fit it onto raw file
        icaref = mne.preprocessing.ICA(n_components=8,max_iter=10000,method="picard",allow_ref_meg=True) #parameters for ica on reference channels
        picks = mne.pick_types(mraw.info,meg=False,ref_meg=True)
        icaref.fit(mraw,picks=picks)
        #save the reference ica result in its own file
        icaref.save('{dir}{sub}_{run}_ref-ica.fif'.format(dir=proc_dir,sub=sub,run=run))

        #define the ica for MEG channels and fit it onto raw file
        icameg = mne.preprocessing.ICA(n_components=100,max_iter=10000,method="picard") #parameters for ica on MEG channels
        picks = mne.pick_types(mraw.info,meg=True,ref_meg=False)
        icameg.fit(mraw,picks=picks)
        #save the MEG ica result in its own file
        icameg.save('{dir}{sub}_{run}_meg-ica.fif'.format(dir=proc_dir,sub=sub,run=run))

        #define the combined ica for MEG and reference channels and fit it onto raw file
        icaall = mne.preprocessing.ICA(n_components=100,max_iter=10000,method="picard",allow_ref_meg=True) #parameters for ica on reference channels
        picks = mne.pick_types(mraw.info,meg=True,ref_meg=True)
        icaall.fit(mraw,picks=picks)
        icaall.save('{dir}{sub}_{run}-ica.fif'.format(dir=proc_dir,sub=sub,run=run))
