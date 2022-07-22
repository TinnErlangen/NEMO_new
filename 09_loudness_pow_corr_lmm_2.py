# try LMM network stats on loudness correlation cluster ROIs

import numpy as np
import mne
import pandas as pd
import random
from scipy import stats
import matplotlib.pyplot as plt
plt.ion()
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf



# read in power data from NEMO dataframe
pdir = "/home/cora/hdd/MEG/NEMO_analyses_new/proc/"
df = pd.read_csv(pdir+"NEMO_complete.csv")

# select needed columns, i.e. variables, and build new DF
base_cols = [col for col in df.columns if "-" not in col]
PH_cols = [col for col in df.columns if (("parahipp" in col) and ("Ton" in col) and ("lh" in col) and ("alpha" in col))]
TT_cols = [col for col in df.columns if (("transversetemporal" in col) and ("Ton" in col) and ("lh" in col) and ("alpha" in col))]
ST_cols = [col for col in df.columns if (("superiortemporal" in col) and ("Ton" in col) and ("lh" in col) and ("alpha" in col))]
MT_cols = [col for col in df.columns if (("middletemporal" in col) and ("Ton" in col) and ("lh" in col) and ("alpha" in col))]
BST_cols = [col for col in df.columns if (("banks" in col) and ("Ton" in col) and ("lh" in col) and ("alpha" in col))]
IP_cols = [col for col in df.columns if (("inferiorparietal" in col) and ("Ton" in col) and ("lh" in col) and ("alpha" in col))]
SM_cols = [col for col in df.columns if (("supramarginal" in col) and ("Ton" in col) and ("lh" in col) and (("alpha" in col) or ("beta_high" in col)))]

laut_columns = base_cols + PH_cols + TT_cols + ST_cols + MT_cols + BST_cols + IP_cols + SM_cols
laut_dict = {col.replace("-","_"): df[col] for col in laut_columns}
laut_columns = [col.replace("-","_") for col in laut_columns]
df_laut = pd.DataFrame(laut_dict)
# remove rows with missing values
df_laut.dropna(inplace=True)

# now the STATS
# Variable Lists
beh_vars = ["TonLaut","TonAng"]
ton = "C(Ton, Treatment('r2'))"
emo = "C(Cond, Treatment('positive'))"
P1_pow_vars = [var for var in laut_columns if "Part1" in var]
P2_pow_vars = [var for var in laut_columns if "Part2" in var]
P3_pow_vars = [var for var in laut_columns if "Part3" in var]
P4_pow_vars = [var for var in laut_columns if "Part4" in var]
pow_vars = P1_pow_vars + P2_pow_vars + P3_pow_vars + P4_pow_vars


# Stat prep
d_var = "TonLaut"
obsvals = df_laut[d_var]
# aicd_thresh = 5
def aic_pval(a,b):
    return np.exp((a - b)/2)   # calculates the evidence ratio for 2 aic values
def pseudo_r2(m, y):    # calculates the pseudo r2 from model summary(m) and observed vals (y)
    fitvals = m.fittedvalues
    r , v = stats.pearsonr(fitvals, y)
    return r**2

# Building Otimal Model from Power Variables
p_vars = P4_pow_vars.copy()     # here to choose which part or all 

# Null model
print("ANALYSES FOR {}".format(d_var))
model = "{dv} ~ 1".format(dv=d_var)
res_0 = smf.mixedlm('{}'.format(model), data=df_laut, groups=df_laut['Subject']).fit(reml=False)
print("Null model AIC =  ", res_0.aic)
print("Null model PseudoR2 =  ", pseudo_r2(res_0,obsvals))
null_aic = res_0.aic
last_aic = res_0.aic    # for deltas and comparisons

# Experimental variables
print("Testing Experiment Variables..")
print("TON")
model = "{dv} ~ {t}".format(dv=d_var, t=ton)
res_ton = smf.mixedlm('{}'.format(model), data=df_laut, groups=df_laut['Subject']).fit(reml=False)
ton_aic = res_ton.aic
print("Ton model results -- AIC = ", ton_aic, ", AIC_delta = ", null_aic - ton_aic, ", AIC_p = ", aic_pval(ton_aic,null_aic))
print("EMO")
model = "{dv} ~ {e}".format(dv=d_var, e=emo)
res_emo = smf.mixedlm('{}'.format(model), data=df_laut, groups=df_laut['Subject']).fit(reml=False)
emo_aic = res_emo.aic
print("Emo model results -- AIC = ", emo_aic, ", AIC_delta = ", null_aic - emo_aic, ", AIC_p = ", aic_pval(emo_aic,null_aic))
# get current best model AIC
last_aic = np.array([ton_aic,emo_aic]).min()
# combine factors
print("EMO & TON")
model = "{dv} ~ {t} + {e}".format(dv=d_var, t=ton, e=emo)
res_comb = smf.mixedlm('{}'.format(model), data=df_laut, groups=df_laut['Subject']).fit(reml=False)
comb_aic = res_comb.aic
print("Emo & Ton 2 factor model results -- AIC = ", comb_aic, ", AIC_delta = ", last_aic - comb_aic, ", AIC_p = ", aic_pval(comb_aic,last_aic))
# short-cut for TonLaut, where we know Ton is the only main effect from the experiment:
last_aic = ton_aic

# Finding the Optimal Model with Power Variables
print("Finding the Optimal Model with Power Variables on TON ..")
aic_p = 0
ix = 0
aic_dict = {}
aicd_dict = {}
# First Best Variable
ix = 1
print("Iteration/Variable {}".format(ix))
model_bef = "{dv} ~ {t}".format(dv=d_var, t=ton)   # combined ton for TonLaut
for p_var in p_vars:
    model_now = "{mb} * {pv}".format(mb=model_bef, pv=p_var)
    print(model_now)
    res = smf.mixedlm('{}'.format(model_now), data=df_laut, groups=df_laut['Subject']).fit(reml=False)
    aic_dict[p_var] = res.aic
    aicd_dict[p_var] = last_aic - res.aic
aic_dict_t = {v: k for k,v in aic_dict.items()}
print("AICs: ", aic_dict)
print("AIC Deltas: ", aicd_dict)
best_aic = np.array(list(aic_dict.values())).min()
best_var = aic_dict_t[best_aic]
print("Best Variable is: ", best_var)
aic_p = aic_pval(best_aic, last_aic)
if aic_p < 0.05 :
    model_1st = "{mb} * {bv}".format(mb=model_bef, bv=best_var)
    model_left = "{mb} * ({bv}".format(mb=model_bef, bv=best_var)
    p_vars.remove(best_var)
    last_aic = best_aic
    print("Current best model: ", model_1st, ", AIC = ", best_aic, ", AIC_delta = ", aicd_dict[best_var], ", AIC_p = ", aic_p)
    # now Iterate until Best Model
    while aic_p < 0.05 :
        ix = ix + 1
        aic_dict = {}
        aicd_dict = {}
        print("Iteration/Variable {}".format(ix))
        for p_var in p_vars:
            model_now = "{ml} + {pv})".format(ml=model_left, pv=p_var)
            print(model_now)
            res = smf.mixedlm('{}'.format(model_now), data=df_laut, groups=df_laut['Subject']).fit(reml=False)
            aic_dict[p_var] = res.aic
            aicd_dict[p_var] = last_aic - res.aic
        aic_dict_t = {v: k for k,v in aic_dict.items()}
        print("AICs: ", aic_dict)
        print("AIC Deltas: ", aicd_dict)
        best_aic = np.array(list(aic_dict.values())).min()
        best_var = aic_dict_t[best_aic]
        print("Best Variable is: ", best_var)
        aic_p = aic_pval(best_aic, last_aic)
        if aic_p < 0.05 :
            model_bef = "{ml} + {bv})".format(ml=model_left, bv=best_var)
            model_left = "{ml} + {bv}".format(ml=model_left, bv=best_var)
            p_vars.remove(best_var)
            last_aic = best_aic
            print("Current best model: ", model_bef, ", AIC = ", best_aic, ", AIC_delta = ", aicd_dict[best_var], ", AIC_p = ", aic_p)


print("Optimization for {dv} complete.".format(dv=d_var))
print("Optimal model is: ", model_bef)
res_opt = smf.mixedlm('{}'.format(model_bef), data=df_laut, groups=df_laut['Subject']).fit(reml=False)
print(res_opt.summary())
print("Opt Model PseudoR2 = ", pseudo_r2(res_opt,obsvals))
