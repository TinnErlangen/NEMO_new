# calculate and examine Linear Mixed Models on dPTE trial-based data #

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()
import statsmodels.api as sm
import statsmodels.formula.api as smf

# set directory
save_dir = "/home/cora/hdd/MEG/NEMO_analyses_new/proc/dPTE/"
# load the dPTE dataframe
df_NEM_dPTE_alpha = pd.read_csv("{}NEMO_dPTE_alpha.csv".format(save_dir))
df_NEM_dPTE_beta = pd.read_csv("{}NEMO_dPTE_beta.csv".format(save_dir))

# the dPTE label connections to look at
columns = ["Subject","Emo","Connection","dPTE"]
emos = ["neg","pos"]
conxs = ["LO-IP","LO-IT","LO-SM","LO-TT","IP-IT",
         "IP-SM","IP-TT","IT-SM","IT-TT","SM-TT"]

# separate LMMs for each connection

# setup LMM
# for Alpha band
print("Doing stats on ALPHA dPTE connections")
for conx in conxs:
    print("Calculating LMM for connection:  {}".format(conx))
    df = df_NEM_dPTE_alpha[df_NEM_dPTE_alpha.Connection == conx].infer_objects()
    try:
        res_0 = smf.mixedlm('dPTE ~ 1', data=df, groups=df['Subject']).fit(reml=False)
        print("Null model:")
        print(res_0.params)
        print(res_0.summary())
        print("AIC: {}".format(res_0.aic))
    except:
        print("Null Model could not converge...")
        continue
    try:
        res_emo = smf.mixedlm('dPTE ~ Emo', data=df, groups=df['Subject']).fit(reml=False)
    except:
        print("Emotion Model could not converge...")
        continue
    print("Emotion model:")
    print(res_emo.params)
    print(res_emo.summary())
    print("AIC: {}".format(res_emo.aic))
    if res_emo.aic < res_0.aic:
        print("Is AIC_Emo sign. better than 0_AIC? -- est. p-value: {}".format(np.exp((res_emo.aic - res_0.aic)/2)))

# for Beta band
print("Doing stats on BETA dPTE connections")
for conx in conxs:
    print("Calculating LMM for connection:  {}".format(conx))
    df = df_NEM_dPTE_beta[df_NEM_dPTE_beta.Connection == conx].infer_objects()
    try:
        res_0 = smf.mixedlm('dPTE ~ 1', data=df, groups=df['Subject']).fit(reml=False)
        print("Null model:")
        print(res_0.params)
        print(res_0.summary())
        print("AIC: {}".format(res_0.aic))
    except:
        print("Null Model could not converge...")
        continue
    try:
        res_emo = smf.mixedlm('dPTE ~ Emo', data=df, groups=df['Subject']).fit(reml=False)
    except:
        print("Emotion Model could not converge...")
        continue
    print("Emotion model:")
    print(res_emo.params)
    print(res_emo.summary())
    print("AIC: {}".format(res_emo.aic))
    if res_emo.aic < res_0.aic:
        print("Is AIC_Emo sign. better than 0_AIC? -- est. p-value: {}".format(np.exp((res_emo.aic - res_0.aic)/2)))

# full LMM models with connection as factor, and an rm_ANOVA for comparison

# ALPHA models
# full interaction model
res_int_a = smf.mixedlm('dPTE ~ Emo * Connection', data=dfa, groups=dfa['Subject']).fit(reml=False)
print(res_int_a.summary())
print(res_int_a.aic)
# null model
res_0_a = smf.mixedlm('dPTE ~ 1', data=dfa, groups=dfa['Subject']).fit(reml=False)
print(res_0_a.summary())
print(res_0_a.aic)
# single factor 'emotion'
res_emo_a = smf.mixedlm('dPTE ~ Emo', data=dfa, groups=dfa['Subject']).fit(reml=False)
print(res_emo_a.summary())
print(res_emo_a.aic)
# single factor 'connection'
res_con_a = smf.mixedlm('dPTE ~ Connection', data=dfa, groups=dfa['Subject']).fit(reml=False)
print(res_con_a.summary())
print(res_con_a.aic)
# two independent factors model
res_2f_a = smf.mixedlm('dPTE ~ Emo + Connection', data=dfa, groups=dfa['Subject']).fit(reml=False)
print(res_2f_a.summary())
print(res_2f_a.aic)
# rm ANOVA & Tukey HSD post-hoc tests
anova_a = statsmodels.stats.anova.AnovaRM(df_NEM_dPTE_alpha,depvar="dPTE",subject="Subject",within=["Emo","Connection"],aggregate_func='mean')
anova_a_res = anova_a.fit()
print(anova_a_res.summary())
res_tukey_emo_a = statsmodels.stats.multicomp.pairwise_tukeyhsd(df_NEM_dPTE_alpha.dPTE, groups=df_NEM_dPTE_alpha.Emo, alpha=0.05)
print(res_tukey_emo_a.summary())
res_tukey_con_a = statsmodels.stats.multicomp.pairwise_tukeyhsd(df_NEM_dPTE_alpha.dPTE, groups=df_NEM_dPTE_alpha.Connection, alpha=0.05)
print(res_tukey_con_a.summary())
# AIC diff significance test for best 2 models
