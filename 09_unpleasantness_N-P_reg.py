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
LO_cols = [col for col in df.columns if (("lateraloccipital" in col) and ("Ton" in col) and ("rh" in col) and (("alpha" in col) or ("beta_low" in col)))]
IT_cols = [col for col in df.columns if (("inferiortemporal" in col) and ("Ton" in col) and ("rh" in col) and (("alpha" in col) or ("beta_low" in col)))]
IP_cols = [col for col in df.columns if (("inferiorparietal" in col) and ("Ton" in col) and ("rh" in col) and (("alpha" in col) or ("beta_low" in col)))]
SM_cols = [col for col in df.columns if (("supramarginal" in col) and ("Ton" in col) and ("rh" in col) and (("alpha" in col) or ("beta_low" in col)))]
TT_cols = [col for col in df.columns if (("transversetemporal" in col) and ("Ton" in col) and ("rh" in col) and (("alpha" in col) or ("beta_low" in col)))]

ang_columns = base_cols + LO_cols + IT_cols + IP_cols + SM_cols + TT_cols
ang_dict = {col.replace("-","_"): df[col] for col in ang_columns}
ang_columns = [col.replace("-","_") for col in ang_columns]
df_ang = pd.DataFrame(ang_dict)
# remove rows with missing values
df_ang.dropna(inplace=True)

# now calc new columns/variables with power mean over parts per trial
df_ang["LO_alpha"] = (df_ang["Ton_Part1_lateraloccipital_rh_alpha"] + df_ang["Ton_Part2_lateraloccipital_rh_alpha"] +
                      df_ang["Ton_Part3_lateraloccipital_rh_alpha"] + df_ang["Ton_Part4_lateraloccipital_rh_alpha"]) / 4
df_ang["LO_beta"] = (df_ang["Ton_Part1_lateraloccipital_rh_beta_low"] + df_ang["Ton_Part2_lateraloccipital_rh_beta_low"] +
                     df_ang["Ton_Part3_lateraloccipital_rh_beta_low"] + df_ang["Ton_Part4_lateraloccipital_rh_beta_low"]) / 4
df_ang["IT_alpha"] = (df_ang["Ton_Part1_inferiortemporal_rh_alpha"] + df_ang["Ton_Part2_inferiortemporal_rh_alpha"] +
                      df_ang["Ton_Part3_inferiortemporal_rh_alpha"] + df_ang["Ton_Part4_inferiortemporal_rh_alpha"]) / 4
df_ang["IT_beta"] = (df_ang["Ton_Part1_inferiortemporal_rh_beta_low"] + df_ang["Ton_Part2_inferiortemporal_rh_beta_low"] +
                     df_ang["Ton_Part3_inferiortemporal_rh_beta_low"] + df_ang["Ton_Part4_inferiortemporal_rh_beta_low"]) / 4
df_ang["IP_alpha"] = (df_ang["Ton_Part1_inferiorparietal_rh_alpha"] + df_ang["Ton_Part2_inferiorparietal_rh_alpha"] +
                      df_ang["Ton_Part3_inferiorparietal_rh_alpha"] + df_ang["Ton_Part4_inferiorparietal_rh_alpha"]) / 4
df_ang["IP_beta"] = (df_ang["Ton_Part1_inferiorparietal_rh_beta_low"] + df_ang["Ton_Part2_inferiorparietal_rh_beta_low"] +
                     df_ang["Ton_Part3_inferiorparietal_rh_beta_low"] + df_ang["Ton_Part4_inferiorparietal_rh_beta_low"]) / 4
df_ang["SM_alpha"] = (df_ang["Ton_Part1_supramarginal_rh_alpha"] + df_ang["Ton_Part2_supramarginal_rh_alpha"] +
                      df_ang["Ton_Part3_supramarginal_rh_alpha"] + df_ang["Ton_Part4_supramarginal_rh_alpha"]) / 4
df_ang["SM_beta"] = (df_ang["Ton_Part1_supramarginal_rh_beta_low"] + df_ang["Ton_Part2_supramarginal_rh_beta_low"] +
                     df_ang["Ton_Part3_supramarginal_rh_beta_low"] + df_ang["Ton_Part4_supramarginal_rh_beta_low"]) / 4
df_ang["TT_alpha"] = (df_ang["Ton_Part1_transversetemporal_rh_alpha"] + df_ang["Ton_Part2_transversetemporal_rh_alpha"] +
                      df_ang["Ton_Part3_transversetemporal_rh_alpha"] + df_ang["Ton_Part4_transversetemporal_rh_alpha"]) / 4
df_ang["TT_beta"] = (df_ang["Ton_Part1_transversetemporal_rh_beta_low"] + df_ang["Ton_Part2_transversetemporal_rh_beta_low"] +
                     df_ang["Ton_Part3_transversetemporal_rh_beta_low"] + df_ang["Ton_Part4_transversetemporal_rh_beta_low"]) / 4

# make a new DF with the mean power vars only
pow_vars = ["LO_alpha","LO_beta","IT_alpha","IT_beta","IP_alpha","IP_beta","SM_alpha","SM_beta","TT_alpha","TT_beta"]

ANGD_dict = {c: df_ang[c] for c in df_ang.columns if c in base_cols or c in pow_vars}
ANGD = pd.DataFrame(ANGD_dict)

# then calculate the N-P diff Dataframe (per Subjects)
N_behav = pd.read_csv('{}NEMO_behav.csv'.format(pdir))
ANG = N_behav.copy()

for sub in ANG.Subjects:
    subdat = df_ang.loc[df_ang['Subject'] == sub]
    for p_var in pow_vars:
        ANG.loc[ANG.Subjects == sub, p_var+'_Diff'] = subdat[subdat['Cond']=='negative'][p_var].mean() - subdat[subdat['Cond']=='positive'][p_var].mean()


# Stat prep
d_var = "Ton_Ang"
d_var = "TT_beta_Diff"
obsvals = ANG[d_var]
# aicd_thresh = 5
def aic_pval(a,b):
    return np.exp((a - b)/2)   # calculates the evidence ratio for 2 aic values

vars = ['Psycho_ges','Angst_ges','ER_ges','Pic_Val','Pic_Ars',"LO_alpha_Diff","LO_beta_Diff","IT_alpha_Diff","IT_beta_Diff","IP_alpha_Diff","IP_beta_Diff","SM_alpha_Diff","SM_beta_Diff","TT_alpha_Diff","TT_beta_Diff"]
# vars = ['Psycho_ges','ER_ges','Pic_Val','Pic_Ars',"LO_alpha_Diff","LO_beta_Diff","IT_alpha_Diff","IT_beta_Diff","IP_alpha_Diff","IP_beta_Diff","SM_alpha_Diff","SM_beta_Diff","TT_alpha_Diff","TT_beta_Diff"]
vars.remove(d_var)

# Null model
print("ANALYSES FOR {}".format(d_var))
model = "{dv} ~ 1".format(dv=d_var)
res_0 = smf.ols('{}'.format(model), data=ANG).fit()
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
    res = smf.ols('{}'.format(model_now), data=ANG).fit()
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
            res = smf.ols('{}'.format(model_now), data=ANG).fit()
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
res_opt = smf.ols('{}'.format(model_bef), data=ANG).fit()
print(res_opt.summary())
