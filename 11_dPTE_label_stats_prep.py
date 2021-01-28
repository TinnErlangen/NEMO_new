## Load the single epochs per subjects results of dPTE Directed Connectivity for 10 Label Connections and build a Dateframe ##
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()

# set directories
proc_dir = "/home/cora/hdd/MEG/NEMO_analyses_new/proc/"  # on workstation D:/
save_dir = "/home/cora/hdd/MEG/NEMO_analyses_new/proc/dPTE/"
            # subjects list
subjs = ["NEM_10","NEM_11","NEM_12","NEM_14","NEM_15",
         "NEM_16","NEM_17","NEM_18","NEM_20","NEM_22",
         "NEM_23","NEM_24","NEM_26","NEM_27","NEM_28",
         "NEM_29","NEM_31","NEM_34","NEM_35","NEM_36"]
# subjs = ["NEM_10","NEM_11"]
excluded = ["NEM_30","NEM_32","NEM_33","NEM_37","NEM_19","NEM_21"]

# parameters from dPTE calc
freqs_a = [9.,10.,11.,12.,13.,14.]
freqs_b = [14.,15.,16.,17.,18.,19.,20.,21.]
cycles_a = 7
cycles_b = 8
# time period of interest (during tones)
tmin = 3.6
tmax = 6.6
# labels of interest
lois = ['lateraloccipital-rh','inferiortemporal-rh','inferiorparietal-rh','supramarginal-rh','transversetemporal-rh']
labels_limb = ['Left-Thalamus-Proper','Left-Caudate','Left-Putamen','Left-Pallidum','Left-Hippocampus','Left-Amygdala',
              'Right-Thalamus-Proper','Right-Caudate','Right-Putamen','Right-Pallidum','Right-Hippocampus','Right-Amygdala']

# build an empty dataframe
columns = ["Subject","Emo","Connection","dPTE"]
emos = ["neg","pos"]
conxs = ["LO-IP","LO-IT","LO-SM","LO-TT","IP-IT",
         "IP-SM","IP-TT","IT-SM","IT-TT","SM-TT"]
dPTE_ix = {"LO-IP":[0,2],"LO-IT":[1,2],"LO-SM":[2,3],"LO-TT":[2,4],"IP-IT":[0,1],
           "IP-SM":[0,3],"IP-TT":[0,4],"IT-SM":[1,3],"IT-TT":[1,4],"SM-TT":[3,4]}
df_NEM_dPTE_alpha = pd.DataFrame(columns=columns)
df_NEM_dPTE_beta = pd.DataFrame(columns=columns)

# loop through subjects to load dPTE epochs data, and sort them into the dataframes
for sub in subjs:
    # keep track
    print("Processing subject:   {}".format(sub))
    # load the dPTE results
    dPTE_neg_a = np.load("{}{}_dPTE_epo_alpha_neg.npy".format(save_dir,sub))
    dPTE_pos_a = np.load("{}{}_dPTE_epo_alpha_pos.npy".format(save_dir,sub))
    dPTE_neg_b = np.load("{}{}_dPTE_epo_beta_neg.npy".format(save_dir,sub))
    dPTE_pos_b = np.load("{}{}_dPTE_epo_beta_pos.npy".format(save_dir,sub))
    # keep track
    print("Data loaded.")
    # create empty subject x condition dataframes for collecting values
    df_neg_a = pd.DataFrame(columns=columns,index=range(dPTE_neg_a.shape[0]*len(conxs)))
    df_pos_a = pd.DataFrame(columns=columns,index=range(dPTE_pos_a.shape[0]*len(conxs)))
    df_neg_b = pd.DataFrame(columns=columns,index=range(dPTE_neg_b.shape[0]*len(conxs)))
    df_pos_b = pd.DataFrame(columns=columns,index=range(dPTE_pos_b.shape[0]*len(conxs)))
    # for each cond, fill in values row by row
    # notice: there are 10 rows for each trial, each specifying the dPTE for one connection -- analyses will later select by conx ...
    # Negative-Alpha
    for tr_ix in range(dPTE_neg_a.shape[0]):
        for c_ix,conx in enumerate(conxs):
            df_neg_a.iloc[c_ix+(tr_ix*len(conxs))].at['Subject'] = sub
            df_neg_a.iloc[c_ix+(tr_ix*len(conxs))].at['Emo'] = 'neg'
            df_neg_a.iloc[c_ix+(tr_ix*len(conxs))].at['Connection'] = conx
            # grab the dPTE values for specified connections
            # flip sign for re-ordered label connections (->other direction)
            if conx in ["LO-IP","LO-IT"]:
                df_neg_a.iloc[c_ix+(tr_ix*len(conxs))].at['dPTE'] = ((dPTE_neg_a[tr_ix][dPTE_ix[conx][0]][dPTE_ix[conx][1]]) - 0.5) *-1
            else:
                df_neg_a.iloc[c_ix+(tr_ix*len(conxs))].at['dPTE'] = (dPTE_neg_a[tr_ix][dPTE_ix[conx][0]][dPTE_ix[conx][1]]) - 0.5
    # append the subject x condition dataframe to the group dataframe
    df_NEM_dPTE_alpha = pd.concat([df_NEM_dPTE_alpha,df_neg_a])
    # Positive-Alpha
    for tr_ix in range(dPTE_pos_a.shape[0]):
        for c_ix,conx in enumerate(conxs):
            df_pos_a.iloc[c_ix+(tr_ix*len(conxs))].at['Subject'] = sub
            df_pos_a.iloc[c_ix+(tr_ix*len(conxs))].at['Emo'] = 'pos'
            df_pos_a.iloc[c_ix+(tr_ix*len(conxs))].at['Connection'] = conx
            # grab the dPTE values for specified connections
            # flip sign for re-ordered label connections (->other direction)
            if conx in ["LO-IP","LO-IT"]:
                df_pos_a.iloc[c_ix+(tr_ix*len(conxs))].at['dPTE'] = ((dPTE_pos_a[tr_ix][dPTE_ix[conx][0]][dPTE_ix[conx][1]]) - 0.5) *-1
            else:
                df_pos_a.iloc[c_ix+(tr_ix*len(conxs))].at['dPTE'] = (dPTE_pos_a[tr_ix][dPTE_ix[conx][0]][dPTE_ix[conx][1]]) - 0.5
    # append the subject x condition dataframe to the group dataframe
    df_NEM_dPTE_alpha = pd.concat([df_NEM_dPTE_alpha,df_pos_a])
    # Negative-Beta
    for tr_ix in range(dPTE_neg_b.shape[0]):
        for c_ix,conx in enumerate(conxs):
            df_neg_b.iloc[c_ix+(tr_ix*len(conxs))].at['Subject'] = sub
            df_neg_b.iloc[c_ix+(tr_ix*len(conxs))].at['Emo'] = 'neg'
            df_neg_b.iloc[c_ix+(tr_ix*len(conxs))].at['Connection'] = conx
            # grab the dPTE values for specified connections
            # flip sign for re-ordered label connections (->other direction)
            if conx in ["LO-IP","LO-IT"]:
                df_neg_b.iloc[c_ix+(tr_ix*len(conxs))].at['dPTE'] = ((dPTE_neg_b[tr_ix][dPTE_ix[conx][0]][dPTE_ix[conx][1]]) - 0.5) *-1
            else:
                df_neg_b.iloc[c_ix+(tr_ix*len(conxs))].at['dPTE'] = (dPTE_neg_b[tr_ix][dPTE_ix[conx][0]][dPTE_ix[conx][1]]) - 0.5
    # append the subject x condition dataframe to the group dataframe
    df_NEM_dPTE_beta = pd.concat([df_NEM_dPTE_beta,df_neg_b])
    # Positive-Beta
    for tr_ix in range(dPTE_pos_b.shape[0]):
        for c_ix,conx in enumerate(conxs):
            df_pos_b.iloc[c_ix+(tr_ix*len(conxs))].at['Subject'] = sub
            df_pos_b.iloc[c_ix+(tr_ix*len(conxs))].at['Emo'] = 'pos'
            df_pos_b.iloc[c_ix+(tr_ix*len(conxs))].at['Connection'] = conx
            # grab the dPTE values for specified connections
            # flip sign for re-ordered label connections (->other direction)
            if conx in ["LO-IP","LO-IT"]:
                df_pos_b.iloc[c_ix+(tr_ix*len(conxs))].at['dPTE'] = ((dPTE_pos_b[tr_ix][dPTE_ix[conx][0]][dPTE_ix[conx][1]]) - 0.5) *-1
            else:
                df_pos_b.iloc[c_ix+(tr_ix*len(conxs))].at['dPTE'] = (dPTE_pos_b[tr_ix][dPTE_ix[conx][0]][dPTE_ix[conx][1]]) - 0.5
    # append the subject x condition dataframe to the group dataframe
    df_NEM_dPTE_beta = pd.concat([df_NEM_dPTE_beta,df_pos_b])

# when all subject data are collected, save the dataframe (feather format is fast and readable in R)
df_NEM_dPTE_alpha.index = list(range(df_NEM_dPTE_alpha.shape[0]))  # fix the index for saving
# df_NEM_dPTE_alpha.to_feather("{}NEMO_dPTE_alpha.feather".format(save_dir))
df_NEM_dPTE_alpha.to_csv("{}NEMO_dPTE_alpha.csv".format(save_dir), index=False)

df_NEM_dPTE_beta.index = list(range(df_NEM_dPTE_beta.shape[0]))  # fix the index for saving
# df_NEM_dPTE_beta.to_feather("{}NEMO_dPTE_beta.feather".format(save_dir))
df_NEM_dPTE_beta.to_csv("{}NEMO_dPTE_beta.csv".format(save_dir), index=False)
