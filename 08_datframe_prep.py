## load the trial power values of all subjects, and create a big Dataframe for the LMM analysis ##

import mne
import numpy as np
import pandas as pd
from io import StringIO

# set directories to load files
behav_dir = "D:/NEMO_analyses/behav/"
proc_dir = "D:/NEMO_analyses_new/proc/"

# subjects list
subjs = ["NEM_10","NEM_11","NEM_12","NEM_14","NEM_15",
         "NEM_16","NEM_17","NEM_18","NEM_20","NEM_22",
         "NEM_23","NEM_24","NEM_26","NEM_27","NEM_28",
         "NEM_29","NEM_31","NEM_34","NEM_35","NEM_36"]

# overview of labels/regions of interest to choose from
aparc_limb_labels = ["bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "cuneus", "entorhinal", "fusiform", "inferiorparietal", "inferiortemporal", "isthmuscingulate",
                     "lateraloccipital", "lateralorbitofrontal", "lingual", "medialorbitofrontal", "middletemporal", "parahippocampal", "paracentral", "parsopercularis", "parsorbitalis",
                     "parstriangularis", "pericalcarine", "postcentral", "posteriorcingulate", "precentral", "precuneus", "rostralanteriorcingulate", "rostralmiddlefrontal",
                     "superiorfrontal", "superiorparietal", "superiortemporal", "supramarginal", "frontalpole", "temporalpole", "transversetemporal", "insula",
                     "Thalamus-Proper", "Caudate", "Putamen", "Pallidum", "Hippocampus", "Amygdala"]
rois_emo = ["Amygdala","insula","parsorbitalis","medialorbitofrontal","lateralorbitofrontal"]
rois_aud = ["Thalamus-Proper","transversetemporal","superiortemporal","bankssts","supramarginal"]
rois_vis = ["lateraloccipital","cuneus","pericalcarine","lingual"]
rois_att = ["rostralanteriorcingulate","caudalanteriorcingulate","posteriorcingulate","superiorparietal"]
rois_mem = ["Hippocampus","parahippocampal","fusiform","inferiortemporal","temporalpole"]
rois_mot = ["precentral"]

# set columns construction variables: Cond, ROI, Freq
conds = ["Pic","Ton_Part1","Ton_Part2","Ton_Part3","Ton_Part4"]
hems = ["-lh","-rh"]
rois_pic = rois_vis + rois_emo + rois_aud
rois_ton = rois_vis + rois_emo + rois_aud
freqs = ["theta","alpha","beta_low","beta_high","gamma","gamma_high"]
start_cols = ["Subject","Psych_ges","Angst_ges","ER_ges","Trial_O","Trial_N","Cond","PicVal","PicArs"]
mid_cols = ["Ton"]
end_cols = ["TonLaut","TonAng","MoodVal","MoodArs"]

# get dicts here with psych variables for fixed vars
Psych_ges = [70,58,19,19,25,23,3,43,52,29,37,25,56,17,16,49,26,28,46,45]
Angst_ges = [15,9,9,7,7,7,1,5,8,6,8,2,14,5,0,6,7,4,14,8]
ER_ges = [60,56,65,77,87,77,93,55,65,86,67,81,83,75,70,71,62,78,60,69]

# start with building superDataframe
columns = start_cols
for cond in conds[0]:
    for roi in rois_pic:
        for hem in hems:
            for freq in freqs:
                columns.append("{c}_{r}{h}_{f}".format(c=cond,r=roi,h=hem,f=freq))
columns += mid_cols
for cond in conds[1:]:
    for roi in rois_ton:
        for hem in hems:
            for freq in freqs:
                columns.append("{c}_{r}{h}_{f}".format(c=cond,r=roi,h=hem,f=freq))
columns += end_cols

df_NEM = pd.DataFrame(columns=columns)

# now loop through the subjects and fill in the data
for sub_ix,sub in enumerate(subjs):
    # load the 3 required .txt files into dataframes: rat, res, trial_power
    rat = pd.read_table("{}rat_{}.txt".format(behav_dir,sub))
    res = pd.read_table("{}res_{}.txt".format(behav_dir,sub))
    trip = pd.read_table("{}{}_trial_roi_power.txt".format(proc_dir,sub))




# this loads one... figure out how to loop best...
nem_10 = pd.read_table("{}NEM_10_trial_roi_power.txt".format(work_dir))
nem_10.insert(0,'Subject','NEM_10',allow_duplicates=False)  # add a subject column
# next subject ...
nem_11 = pd.read_table("{}NEM_11_trial_roi_power.txt".format(work_dir))
nem_11.insert(0,'Subject','NEM_11',allow_duplicates=False)
# combine them ..
comb = pd.concat([nem_10,nem_11])   # s.o. try to fit this in a good loop
print (comb.shape)  # make sure, the shape is correct; get dataframe length in rows

comb.columns  # gives column names

### task: now we gotta rearrange the DF so that the elements in each frequency line are sorted into new columns
for i in range(comb.shape[0]):
    freq = comb.iloc[i].at['Freq']
    ## given that the superDataframe (NEM) exists with columns 'bankssts_lh_alpha' etc. ...
    for col in comb.columns[5:]:
        NEM.iloc[0].at['{}_{}'.format(col,freq)]
