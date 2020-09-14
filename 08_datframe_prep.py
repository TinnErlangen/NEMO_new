## load the trial power values of all subjects, and create a big Dataframe for the LMM analysis ##

import mne
import numpy as np
import pandas as pd
from io import StringIO

# set directories and load files --- correct this
work_dir = "C:/Users/kimca/Desktop/NEMO_files_aktuell/"
behav_dir = # for loading rat and res
proc_dir = # for trial power

# set columns construction variables: Cond, ROI, Freq
conds = ["Pic","Ton_Part1","Ton_Part2","Ton_Part3","Ton_Part4"]
ROIs = []
rois_emo =
rois_aud =
rois_vis =
rois_att =
freqs = ["theta","alpha","beta_low","beta_high","gamma","gamma_high"]
start_cols = ["Subject","Psych_ges","Angst_ges","ER_ges","Trial_O","Trial_N","Cond","PicVal","PicArs"]
mid_cols = ["Ton"]
end_cols = ["TonLaut","TonAng","MoodVal","MoodArs"]

# get dicts here with psych variables for fixed vars
Psych_ges =
Angst_ges =
ER_ges =

# start with building superDataframe
df_NEM = pd.DataFrame(columns=cols)

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
