## Calculate single trial alpha/beta power in selected labels to make stats with dPTE##
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()
import statsmodels.api as sm
import statsmodels.formula.api as smf

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

# load the dataframe containing alpha dPTE and alpha/beta power values per trial
df_NEM_a_dPTE_power = pd.read_csv("{}NEMO_a_dPTE_ab_power.csv".format(save_dir))

# ## THIS WAS DONE TO PREP THE BEHAV DATAFRAME, DONE.
# # load the NEMO dataframe (from 08_datframe_prep) containing behavioral data per trial (and old tone_part power vals)
# df_NEM_behav = df_NEM_behav = pd.read_csv("{}NEMO.csv".format(proc_dir))
# # reduce it to the relevant behavioral variables only, and save
# df_new = df_NEM_behav[['Subject','Trial_O','Trial_N','Cond','Ton','PicVal','PicArs','TonLaut','TonAng','MoodVal','MoodArs']]
# df_new.to_csv('{}NEMO_behav_trials.csv'.format(proc_dir),index=False)
#
# # step 1: collect a list of indices to drop/delete from df_new (i.e. the dropped epochs with bad MEG data)
# # step 2: re-name Cond into pos/neg like in df_NEM_a_dPTE_power
# # step 3: join the two dataframes, sorting neg and pos trials in order ... (can this be done in one "join" command?)
#
# # step 1: collect a list of indices to drop/delete from df_new (i.e. the dropped epochs with bad MEG data)
# # build a container for the row indices to drop from df_new
# drop_ix = []
# # and a container for each subject's bad/dropped epoch indices
# sub_bad_ix = {}
#
# # get the dropped epoch indices for each subject, calculate the df indices, and append to drop_ix
# for sub_ix,sub in enumerate(subjs):
#     epo = mne.read_epochs("{}{}_long-epo.fif".format(proc_dir,sub))
#     # trial 32 in both these subjects is to exlude (NEM_17:repeated test trials at block start; NEM_28:('NO_DATA') in drop log cause no baseline before trigger)
#     if sub == "NEM_17":
#         epo_s = list(epo.drop_log)
#         del epo_s[(32*4):(32*4+4)]
#         epo.drop_log = tuple(epo_s)
#     if sub == "NEM_28":
#         epo_s = list(epo.drop_log)
#         epo_s[(32*4)] = ('BAD_',)
#         epo.drop_log = tuple(epo_s)
#     bad_count = epo.drop_log.count(('BAD_',))
#     bad_ix = []
#     start = 0
#     for c in range(bad_count):
#         ix = epo.drop_log.index(('BAD_',), start)
#         bad_ix.append(ix)
#         start = ix + 1
#     bad_ix = [int(x/4) for x in bad_ix]
#     sub_bad_ix[sub] = bad_ix
#     for bix in bad_ix:
#         drop_ix.append(sub_ix*64 + bix)
#
# # drop the rows with no MEG data from the behav dataframe & save it
# df_new_clean = df_new.drop(drop_ix,axis=0)
# df_new_clean = df_new_clean.replace(to_replace={'positive':'pos','negative':'neg'})
# df_new_clean = df_new_clean.rename(columns={'Cond':'Emo'})
# df_new_clean.to_csv("{}NEM_behav_epodrop.csv".format(proc_dir))

# NOW DO THE SORTING AND MERGING
df_NEM_cp = pd.read_csv("{}NEMO_a_dPTE_ab_power.csv".format(save_dir))
df_NEM_b_orig = pd.read_csv("{}NEM_behav_epodrop.csv".format(proc_dir))
df_NEM_b_sorted = pd.DataFrame(columns=df_NEM_b_orig.columns)

for sub_ix,sub in enumerate(subjs):
    df_sub = df_NEM_b_orig[df_NEM_b_orig.Subject == sub]
    df_sub_neg = df_sub[df_sub.Emo == "neg"]
    df_sub_pos = df_sub[df_sub.Emo == "pos"]
    df_NEM_b_sorted = pd.concat([df_NEM_b_sorted,df_sub_neg])
    df_NEM_b_sorted = pd.concat([df_NEM_b_sorted,df_sub_pos])

# fix the index of sorted df for merging
df_NEM_b_sorted.index = list(range(df_NEM_b_sorted.shape[0]))
# merge the two dataframes (dPTE/power (cp) and behav(b)_sorted)
NEM_all = df_NEM_cp.join(df_NEM_b_sorted, on=None, how='left', lsuffix='', rsuffix='_p', sort=False)
# NEM_all = NEM_all.drop(columns=['Unnamed: 0','Subject_p','Emo_p'])
NEM_all.to_csv("{}NEMO_cpb_stats.csv".format(save_dir), index=False)

# NOW DO STATS
# ...do this per hand to find best model for each behav variable
res_TonLaut_0 = smf.mixedlm('TonLaut ~ 1', data=NEM_all, groups=NEM_all['Subject']).fit(reml=False)
res_TonLaut_emo = smf.mixedlm('TonLaut ~ Emo', data=NEM_all, groups=NEM_all['Subject']).fit(reml=False)
print(res_TonLaut_0.aic)
print(res_TonLaut_emo.summary())
print(res_TonLaut_emo.params)
print(res_TonLaut_emo.aic)
print(np.exp((res_emo.aic - res_0.aic)/2))
res_TonAng_picval_ton_x_TT_b = smf.mixedlm('TonAng ~ PicVal + Ton * TT_b_pow', data=NEM_all, groups=NEM_all['Subject']).fit(reml=False)
# and so on ...
