import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns

save_dir = "/home/cora/hdd/MEG/NEMO_analyses_new/proc/dPTE/"

# Dataframe
NEM_all = pd.read_csv("{}NEMO_cpb_stats_log.csv".format(save_dir))  #_log for log10 data

# Variables
beh_var = ["TonLaut","TonAng"]
bas_var1 = ["C(Ton, Treatment('r2'))"]
bas_var2 = ["C(Emo, Treatment('pos'))","PicVal","PicArs"]
pow_var = ["LO_a_pow","LO_b_pow","IP_a_pow","IP_b_pow","IT_a_pow","IT_b_pow","SM_a_pow","SM_b_pow","TT_a_pow","TT_b_pow"]

# Power Network relations

# Grounding round: which basic variables inform the power in a label?
for d_var in pow_var:
    print("\n\n{}  ---  Checking Base Variables\n\n".format(d_var))
    # get the null model AIC
    model = "{} ~ 1".format(d_var)
    res_0 = smf.mixedlm('{}'.format(model), data=NEM_all, groups=NEM_all['Subject']).fit(reml=False)
    print("Null model AIC =  ", res_0.aic)
    # first do tone
    print("\n--  Testing Ton  --\n")
    model = "{} ~ {}".format(d_var, bas_var1[0])
    res_ton = smf.mixedlm('{}'.format(model), data=NEM_all, groups=NEM_all['Subject']).fit(reml=False)
    print(res_ton.summary())
    print("Ton model AIC =  ", res_ton.aic)
    print("AIC Diff. =  ", res_ton.aic - res_0.aic)
    print("Evidence ratio =  ", np.exp((res_ton.aic - res_0.aic)/2))
    # then the different emotion variables
    for e_var in bas_var2:
        print("\n--  Testing {}  --\n".format(e_var))
        model = "{} ~ {}".format(d_var, e_var)
        res_emo = smf.mixedlm('{}'.format(model), data=NEM_all, groups=NEM_all['Subject']).fit(reml=False)
        print(res_emo.summary())
        print("{} model AIC =  ".format(e_var), res_emo.aic)
        print("AIC Diff. =  ", res_emo.aic - res_0.aic)
        print("Evidence ratio =  ", np.exp((res_emo.aic - res_0.aic)/2))
    # then emo x ton interaction
    print("\n--  Testing Emo-Ton Interaction  --\n")
    model = "{} ~ Emo * Ton".format(d_var)
    res_int = smf.mixedlm('{}'.format(model), data=NEM_all, groups=NEM_all['Subject']).fit(reml=False)
    print(res_int.summary())
    print("Emo-Ton interaction model AIC =  ", res_int.aic)
    print("AIC Diff. =  ", res_int.aic - res_0.aic)
    print("Evidence ratio =  ", np.exp((res_int.aic - res_0.aic)/2))

# Topic 2: do power variables relate to rating outcomes?
for b_var in beh_var:
    print("\n\n\n\n{}  ---  Checking Power Variables\n\n".format(b_var))
    # get the null model AIC
    model = "{} ~ 1".format(b_var)
    res_0 = smf.mixedlm('{}'.format(model), data=NEM_all, groups=NEM_all['Subject']).fit(reml=False)
    print("Null model AIC =  ", res_0.aic)
    for p_var in pow_var:
        print("\n--  Testing {}  --\n".format(p_var))
        model = "{} ~ {}".format(b_var, p_var)
        res_pow = smf.mixedlm('{}'.format(model), data=NEM_all, groups=NEM_all['Subject']).fit(reml=False)
        print(res_pow.summary())
        print("{} model AIC =  ".format(p_var), res_pow.aic)
        print("AIC Diff. =  ", res_pow.aic - res_0.aic)
        print("Evidence ratio =  ", np.exp((res_pow.aic - res_0.aic)/2))

# Power relations check:
for d_var in pow_var:
    print("\n\n\n\n{}  ---  Checking Power Relations\n\n".format(d_var))
    # get the null model AIC
    model = "{} ~ 1".format(d_var)
    res_0 = smf.mixedlm('{}'.format(model), data=NEM_all, groups=NEM_all['Subject']).fit(reml=False)
    print("Null model AIC =  ", res_0.aic)
    for p_var in pow_var:
        if p_var == d_var:
            continue
        print("\n--  Testing {}  --\n".format(p_var))
        model = "{} ~ {}".format(d_var, p_var)
        res_pow = smf.mixedlm('{}'.format(model), data=NEM_all, groups=NEM_all['Subject']).fit(reml=False)
        print(res_pow.summary())
        print("{} model AIC =  ".format(p_var), res_pow.aic)
        print("AIC Diff. =  ", res_pow.aic - res_0.aic)
        print("Evidence ratio =  ", np.exp((res_pow.aic - res_0.aic)/2))
