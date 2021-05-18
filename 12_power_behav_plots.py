import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# plt.ion()
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns

# Directories
save_dir = "/home/cora/hdd/MEG/NEMO_analyses_new/proc/dPTE/"
p_save_dir = "/home/cora/hdd/MEG/NEMO_analyses_new/stats/"

# Dataframe (power vals were log10 transformed)
NEM_all = pd.read_csv("{}NEMO_cpb_stats_log.csv".format(save_dir))  #_log for log10 data
pow_var = ["LO_a_pow","LO_b_pow","IP_a_pow","IP_b_pow","IT_a_pow","IT_b_pow","SM_a_pow","SM_b_pow","TT_a_pow","TT_b_pow"]
beh_var = ["TonAng","TonLaut"]

# Create Scatterplots for the Power main effects / interactions

for d_var in beh_var:
    for var in pow_var:
        fig1 = sns.lmplot(x=var,y=d_var,data=NEM_all, hue="Emo", hue_order=["pos","neg"])
        fig1.savefig("{p}{d}_{v}_x_Emo_lmplot".format(p=p_save_dir, d=d_var, v=var))
        fig2 = sns.lmplot(x=var,y=d_var,data=NEM_all, hue="Ton", hue_order=["r2","r1","s1","s2"])
        fig2.savefig("{p}{d}_{v}_x_Ton_lmplot".format(p=p_save_dir, d=d_var, v=var))

# Create Grid Scatterplots for the Power emotion effects per tone

for d_var in beh_var:
    for var in pow_var:
        fig = sns.lmplot(x=var,y=d_var,data=NEM_all, col="Ton", hue="Emo", col_order=["r2","r1","s1","s2"], hue_order=["pos","neg"])
        fig.savefig("{p}{d}_{v}_Ton_x_Emo_grid".format(p=p_save_dir, d=d_var, v=var))

# Plot boxplots / bargraphs of each Label Power, comparing Emo by Tones
for var in pow_var:
    # sns.catplot(x="Ton", y=var, hue="Emo", kind="bar", data=NEM_all, order=["r2","r1","s1","s2"], hue_order=["pos","neg"])
    # sns.catplot(x="Ton", y=var, hue="Emo", kind="violin", inner="stick",
    #             split=True, palette="pastel", data=NEM_all,
    #             order=["r2","r1","s1","s2"], hue_order=["pos","neg"])
    fig = sns.catplot(x="Ton", y=var, hue="Emo", kind="boxen", data=NEM_all, order=["r2","r1","s1","s2"], hue_order=["pos","neg"])
    fig.savefig("{p}Emo_Ton_Boxen_{v}".format(p=p_save_dir, v=var))
