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

# now calc new columns/variables with power mean over parts per trial
df_laut["PH_alpha"] = (df_laut["Ton_Part1_parahippocampal_lh_alpha"] + df_laut["Ton_Part2_parahippocampal_lh_alpha"] +
                       df_laut["Ton_Part3_parahippocampal_lh_alpha"] + df_laut["Ton_Part4_parahippocampal_lh_alpha"]) / 4
df_laut["TT_alpha"] = (df_laut["Ton_Part1_transversetemporal_lh_alpha"] + df_laut["Ton_Part2_transversetemporal_lh_alpha"] +
                       df_laut["Ton_Part3_transversetemporal_lh_alpha"] + df_laut["Ton_Part4_transversetemporal_lh_alpha"]) / 4
df_laut["ST_alpha"] = (df_laut["Ton_Part1_superiortemporal_lh_alpha"] + df_laut["Ton_Part2_superiortemporal_lh_alpha"] +
                       df_laut["Ton_Part3_superiortemporal_lh_alpha"] + df_laut["Ton_Part4_superiortemporal_lh_alpha"]) / 4
df_laut["MT_alpha"] = (df_laut["Ton_Part1_middletemporal_lh_alpha"] + df_laut["Ton_Part2_middletemporal_lh_alpha"] +
                       df_laut["Ton_Part3_middletemporal_lh_alpha"] + df_laut["Ton_Part4_middletemporal_lh_alpha"]) / 4
df_laut["BST_alpha"] = (df_laut["Ton_Part1_bankssts_lh_alpha"] + df_laut["Ton_Part2_bankssts_lh_alpha"] +
                       df_laut["Ton_Part3_bankssts_lh_alpha"] + df_laut["Ton_Part4_bankssts_lh_alpha"]) / 4
df_laut["IP_alpha"] = (df_laut["Ton_Part1_inferiorparietal_lh_alpha"] + df_laut["Ton_Part2_inferiorparietal_lh_alpha"] +
                       df_laut["Ton_Part3_inferiorparietal_lh_alpha"] + df_laut["Ton_Part4_inferiorparietal_lh_alpha"]) / 4
df_laut["SM_alpha"] = (df_laut["Ton_Part1_supramarginal_lh_alpha"] + df_laut["Ton_Part2_supramarginal_lh_alpha"] +
                       df_laut["Ton_Part3_supramarginal_lh_alpha"] + df_laut["Ton_Part4_supramarginal_lh_alpha"]) / 4
df_laut["SM_beta"] = (df_laut["Ton_Part1_supramarginal_lh_beta_high"] + df_laut["Ton_Part2_supramarginal_lh_beta_high"] +
                       df_laut["Ton_Part3_supramarginal_lh_beta_high"] + df_laut["Ton_Part4_supramarginal_lh_beta_high"]) / 4

# now the STATS
# Variable Lists
beh_vars = ["TonLaut"]
ton = "C(Ton, Treatment('r2'))"
emo = "C(Cond, Treatment('positive'))"
pow_vars = ["PH_alpha","TT_alpha","ST_alpha","MT_alpha","BST_alpha","IP_alpha","SM_alpha","SM_beta"]


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
p_vars = pow_vars.copy()     # here to choose which part or all

# Null model
print("ANALYSES FOR {}".format(d_var))
model = "{dv} ~ 1".format(dv=d_var)
res_0 = smf.mixedlm('{}'.format(model), data=df_laut, groups=df_laut['Subject']).fit(reml=False)
print("Null model AIC =  ", res_0.aic)
print("Null model PseudoR2 =  ", pseudo_r2(res_0,obsvals))
null_aic = res_0.aic
last_aic = res_0.aic    # for deltas and comparisons


# Finding the Optimal Model with Power Variables
print("Finding the Optimal Model with Power Variables  ..")
aic_p = 0
ix = 0
aic_dict = {}
aicd_dict = {}
# First Best Variable
ix = 1
print("Iteration/Variable {}".format(ix))
model_bef = "{dv} ~ ".format(dv=d_var)
for p_var in p_vars:
    model_now = "{mb} {pv}".format(mb=model_bef, pv=p_var)
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
    model_bef = "{mb} {bv}".format(mb=model_bef, bv=best_var)
    p_vars.remove(best_var)
    last_aic = best_aic
    print("Current best model: ", model_bef, ", AIC = ", best_aic, ", AIC_delta = ", aicd_dict[best_var], ", AIC_p = ", aic_p)
    # now Iterate until Best Model
    while aic_p < 0.05 :
        ix = ix + 1
        aic_dict = {}
        aicd_dict = {}
        print("Iteration/Variable {}".format(ix))
        for p_var in p_vars:
            model_now = "{mb} + {pv}".format(mb=model_bef, pv=p_var)
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
            model_bef = "{mb} + {bv}".format(mb=model_bef, bv=best_var)
            p_vars.remove(best_var)
            last_aic = best_aic
            print("Current best model: ", model_bef, ", AIC = ", best_aic, ", AIC_delta = ", aicd_dict[best_var], ", AIC_p = ", aic_p)


print("Optimization for {dv} complete.".format(dv=d_var))
print("Optimal model is: ", model_bef)
res_opt = smf.mixedlm('{}'.format(model_bef), data=df_laut, groups=df_laut['Subject']).fit(reml=False)
print(res_opt.summary())
print("Opt Model PseudoR2 = ", pseudo_r2(res_opt,obsvals))
