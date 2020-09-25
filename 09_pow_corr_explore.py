## look at power correlations over time parts, rois and frequencies ##

import mne
import numpy as np
import pandas as pd
from io import StringIO
import feather
from scipy import stats

# set directories to load files
proc_dir = "D:/NEMO_analyses_new/proc/"

# building blocks
times = ['Pic','Ton_Part1',"Ton_Part2","Ton_Part3","Ton_Part4"]
rois_v1 = ["lateraloccipital"]
rois_v_dorsal = ["cuneus","pericalcarine","superiorparietal","inferiorparietal"]
rois_v_ventral = ["lingual","fusiform","inferiortemporal"]
rois_aud = ["bankssts","supramarginal","superiortemporal","transversetemporal"]
rois = rois_v1 + rois_v_dorsal + rois_v_ventral + rois_aud
hems = ["-lh","-rh"]
freqs = ["theta","alpha","beta_low","beta_high","gamma","gamma_high"]
behav_cols = ["Subject","Cond","Ton","PicVal","PicArs","TonLaut","TonAng"]

# load the full NEMO dataframe
NEMO = pd.read_csv("{}NEMO_complete.csv".format(proc_dir))

# # choose variables for the correlation dataframe & build it
# NEM = pd.DataFrame(df_all.loc[:,["Subject","Cond","Ton","PicVal","PicArs","TonLaut","TonAng"]])
# for time in times_conn:
#     for roi in rois:
#         for hem in hems:
#             for freq in freqs:
#                 NEM['{}_{}{}_{}'.format(time,roi,hem,freq)] = NEMO['{}_{}{}_{}'.format(time,roi,hem,freq)]

# print out correlations of interest
# set variables of interest
times_comp = ['Ton_Part1','Ton_Part1']
hem = "-rh"
roi_1 = ["Amygdala"]
freqs_1 = ["theta","alpha","beta_low","beta_high","gamma","gamma_high"]
rois_2 = ["transversetemporal"]
freqs_2 = ["theta","alpha","beta_low","beta_high","gamma","gamma_high"]
# loop, calc and print
for r in roi_1:
    for f in freqs_1:
        for roi in rois_2:
            for freq in freqs_2:
                print("Correlation {}_{}{}_{}".format(times_comp[0],r,hem,f)+" & {}_{}{}_{}".format(times_comp[1],roi,hem,freq))
                A = NEMO['{}_{}{}_{}'.format(times_comp[0],r,hem,f)]
                B = NEMO['{}_{}{}_{}'.format(times_comp[1],roi,hem,freq)]
                print(A.corr(B))
# same for negative/positive trials only
neg = NEMO.query('Cond == "negative"')
pos = NEMO.query('Cond == "positive"')
for r in roi_1:
    for f in freqs_1:
        for roi in rois_2:
            for freq in freqs_2:
                print("NEGATIVE - Correlation {}_{}{}_{}".format(times_comp[0],r,hem,f)+" & {}_{}{}_{}".format(times_comp[1],roi,hem,freq))
                A = neg['{}_{}{}_{}'.format(times_comp[0],r,hem,f)]
                B = neg['{}_{}{}_{}'.format(times_comp[1],roi,hem,freq)]
                print(A.corr(B))
for r in roi_1:
    for f in freqs_1:
        for roi in rois_2:
            for freq in freqs_2:
                print("POSITIVE - Correlation {}_{}{}_{}".format(times_comp[0],r,hem,f)+" & {}_{}{}_{}".format(times_comp[1],roi,hem,freq))
                A = pos['{}_{}{}_{}'.format(times_comp[0],r,hem,f)]
                B = pos['{}_{}{}_{}'.format(times_comp[1],roi,hem,freq)]
                print(A.corr(B))
