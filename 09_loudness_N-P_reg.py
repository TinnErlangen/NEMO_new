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

# make a new DF with the mean power vars only
pow_vars = ["PH_alpha","TT_alpha","ST_alpha","MT_alpha","BST_alpha","IP_alpha","SM_alpha","SM_beta"]

LOUDD_dict = {c: df_laut[c] for c in df_laut.columns if c in base_cols or c in pow_vars}
LOUDD = pd.DataFrame(LOUDD_dict)

# then calculate the N-P diff Dataframe (per Subjects)
N_behav = pd.read_csv('{}NEMO_behav.csv'.format(pdir))
LOUD = N_behav.copy()

for sub in LOUD.Subjects:
    subdat = df_laut.loc[df_laut['Subject'] == sub]
    for p_var in pow_vars:
        LOUD.loc[LOUD.Subjects == sub, p_var+'_Diff'] = subdat[subdat['Cond']=='negative'][p_var].mean() - subdat[subdat['Cond']=='positive'][p_var].mean()



# Stat prep
d_var = "Ton_Laut"
# d_var = "Ton_Ang"
obsvals = LOUD[d_var]
# aicd_thresh = 5
def aic_pval(a,b):
    return np.exp((a - b)/2)   # calculates the evidence ratio for 2 aic values

vars = ['Psycho_ges','Angst_ges','ER_ges','Pic_Val','Pic_Ars',"PH_alpha_Diff","TT_alpha_Diff","ST_alpha_Diff","MT_alpha_Diff","BST_alpha_Diff","IP_alpha_Diff","SM_alpha_Diff","SM_beta_Diff"]
# vars = ['Psycho_ges','ER_ges','Pic_Val','Pic_Ars',"PH_alpha_Diff","TT_alpha_Diff","ST_alpha_Diff","MT_alpha_Diff","BST_alpha_Diff","IP_alpha_Diff","SM_alpha_Diff","SM_beta_Diff"]
# vars.remove(d_var)

# Null model
print("ANALYSES FOR {}".format(d_var))
model = "{dv} ~ 1".format(dv=d_var)
res_0 = smf.ols('{}'.format(model), data=LOUD).fit()
print("Null model AIC =  ", res_0.aic)
null_aic = res_0.aic
last_aic = res_0.aic    # for deltas and comparisons

# Finding the Optimal Model with the other Variables
print("Finding the Optimal Model with provided Variables  ..")
aic_p = 0
ix = 0
aic_dict = {}
aicd_dict = {}
# First Best Variable
ix = 1
print("Iteration/Variable {}".format(ix))
model_bef = "{dv} ~ ".format(dv=d_var)
for var in vars:
    model_now = "{mb} {pv}".format(mb=model_bef, pv=var)
    print(model_now)
    res = smf.ols('{}'.format(model_now), data=LOUD).fit()
    aic_dict[var] = res.aic
    aicd_dict[var] = last_aic - res.aic
aic_dict_t = {v: k for k,v in aic_dict.items()}
print("AICs: ", aic_dict)
print("AIC Deltas: ", aicd_dict)
best_aic = np.array(list(aic_dict.values())).min()
best_var = aic_dict_t[best_aic]
print("Best Variable is: ", best_var)
aic_p = aic_pval(best_aic, last_aic)
if aic_p < 0.05 :
    model_bef = "{mb} {bv}".format(mb=model_bef, bv=best_var)
    vars.remove(best_var)
    last_aic = best_aic
    print("Current best model: ", model_bef, ", AIC = ", best_aic, ", AIC_delta = ", aicd_dict[best_var], ", AIC_p = ", aic_p)
    # now Iterate until Best Model
    while aic_p < 0.05 :
        ix = ix + 1
        aic_dict = {}
        aicd_dict = {}
        print("Iteration/Variable {}".format(ix))
        for var in vars:
            model_now = "{mb} + {pv}".format(mb=model_bef, pv=var)
            print(model_now)
            res = smf.ols('{}'.format(model_now), data=LOUD).fit()
            aic_dict[var] = res.aic
            aicd_dict[var] = last_aic - res.aic
        aic_dict_t = {v: k for k,v in aic_dict.items()}
        print("AICs: ", aic_dict)
        print("AIC Deltas: ", aicd_dict)
        best_aic = np.array(list(aic_dict.values())).min()
        best_var = aic_dict_t[best_aic]
        print("Best Variable is: ", best_var)
        aic_p = aic_pval(best_aic, last_aic)
        if aic_p < 0.05 :
            model_bef = "{mb} + {bv}".format(mb=model_bef, bv=best_var)
            vars.remove(best_var)
            last_aic = best_aic
            print("Current best model: ", model_bef, ", AIC = ", best_aic, ", AIC_delta = ", aicd_dict[best_var], ", AIC_p = ", aic_p)


print("Optimization for {dv} complete.".format(dv=d_var))
print("Optimal model is: ", model_bef)
res_opt = smf.ols('{}'.format(model_bef), data=LOUD).fit()
print(res_opt.summary())
