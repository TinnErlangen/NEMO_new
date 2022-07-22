import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from scipy import stats

save_dir = "/home/cora/hdd/MEG/NEMO_analyses_new/proc/dPTE/"

# Dataframe
NEMO = pd.read_csv("{}NEMO_cpb_final.csv".format(save_dir))  #power values are in fT^2 (*1e30) and log10 linearized; dPTE is *100 for scaling

# Variable Lists
beh_vars = ["TonLaut","TonAng"]
ton = "C(Ton, Treatment('r2'))"
emo = "C(Emo, Treatment('pos'))"
pow_vars = ["LO_a_pow","LO_b_pow","IP_a_pow","IP_b_pow","IT_a_pow","IT_b_pow","SM_a_pow","SM_b_pow","TT_a_pow","TT_b_pow"]

# Stat prep
d_var = "TonAng"
obsvals = NEMO[d_var]
# aicd_thresh = 5
def aic_pval(a,b):
    return np.exp((a - b)/2)   # calculates the evidence ratio for 2 aic values
def pseudo_r2(m, y):    # calculates the pseudo r2 from model summary(m) and observed vals (y)
    fitvals = m.fittedvalues
    r , v = stats.pearsonr(fitvals, y)
    return r**2

# Null model
print("Analyses for {}".format(d_var))
model = "{dv} ~ 1".format(dv=d_var)
res_0 = smf.mixedlm('{}'.format(model), data=NEMO, groups=NEMO['Subject']).fit(reml=False)
print(res_0.summary())
print("Null model AIC =  ", res_0.aic)
print("Null model PseudoR2 =  ", pseudo_r2(res_0,obsvals))
null_aic = res_0.aic
last_aic = res_0.aic    # for deltas and comparisons

# Experimental variables
print("Testing Experiment Variables..")
print("TON")
model = "{dv} ~ {t}".format(dv=d_var, t=ton)
res_ton = smf.mixedlm('{}'.format(model), data=NEMO, groups=NEMO['Subject']).fit(reml=False)
ton_aic = res_ton.aic
print(res_ton.summary())
print("Ton model results -- AIC = ", ton_aic, ", AIC_delta = ", null_aic - ton_aic, ", AIC_p = ", aic_pval(ton_aic,null_aic), ", PseudoR2 = ", pseudo_r2(res_ton,obsvals))
print("EMO")
model = "{dv} ~ {e}".format(dv=d_var, e=emo)
res_emo = smf.mixedlm('{}'.format(model), data=NEMO, groups=NEMO['Subject']).fit(reml=False)
emo_aic = res_emo.aic
print(res_emo.summary())
print("Emo model results -- AIC = ", emo_aic, ", AIC_delta = ", null_aic - emo_aic, ", AIC_p = ", aic_pval(emo_aic,null_aic), ", PseudoR2 = ", pseudo_r2(res_emo,obsvals))
# get current best model AIC
last_aic = np.array([ton_aic,emo_aic]).min()
# combine factors
print("EMO & TON")
model = "{dv} ~ {t} + {e}".format(dv=d_var, t=ton, e=emo)
res_comb = smf.mixedlm('{}'.format(model), data=NEMO, groups=NEMO['Subject']).fit(reml=False)
comb_aic = res_comb.aic
print(res_comb.summary())
print("Emo & Ton 2 factor model results -- AIC = ", comb_aic, ", AIC_delta = ", last_aic - comb_aic, ", AIC_p = ", aic_pval(comb_aic,last_aic), ", PseudoR2 = ", pseudo_r2(res_comb,obsvals))
# short-cut for TonAng, where we know this is best:
last_aic = comb_aic
# .. and just for reporting, we repeat the inferior interaction version
print("EMO * TON")
model = "{dv} ~ {t} * {e}".format(dv=d_var, t=ton, e=emo)
res_int = smf.mixedlm('{}'.format(model), data=NEMO, groups=NEMO['Subject']).fit(reml=False)
int_aic = res_int.aic
print(res_int.summary())
print("Emo * Ton interaction model results -- AIC = ", int_aic, ", AIC_delta = ", last_aic - int_aic, ", AIC_p = ", aic_pval(int_aic,last_aic), ", PseudoR2 = ", pseudo_r2(res_int,obsvals))

# Finding the Optimal Model with Power Variables
print("Finding the Optimal Model with Power Variables..")
aic_p = 0
ix = 0
aic_dict = {}
aicd_dict = {}
model_bef = "{dv} ~ ({t} + {e})".format(dv=d_var, t=ton, e=emo)   # combined emo+ton for TonAng
print("Finding 1st Best Variable..")
for pow_var in pow_vars:
    model_now = "{mb} * {pv}".format(mb=model_bef, pv=pow_var)
    print(model_now)
    res = smf.mixedlm('{}'.format(model_now), data=NEMO, groups=NEMO['Subject']).fit(reml=False)
    aic_dict[pow_var] = res.aic
    aicd_dict[pow_var] = last_aic - res.aic
aic_dict_t = {v: k for k,v in aic_dict.items()}
ix = 1
print("Iteration/Variable {}".format(ix))
print("AICs: ", aic_dict)
print("AIC Deltas: ", aicd_dict)
best_aic = np.array(list(aic_dict.values())).min()
best_var = aic_dict_t[best_aic]
print("Best Variable is: ", best_var)
aic_p = aic_pval(best_aic, last_aic)
if aic_p < 0.05 :
    model_1st = "{mb} * {bv}".format(mb=model_bef, bv=best_var)
    model_left = "{mb} * ({bv}".format(mb=model_bef, bv=best_var)
    pow_vars.remove(best_var)
    last_aic = best_aic
    print("Current best model: ", model_1st, ", AIC = ", best_aic, ", AIC_delta = ", aicd_dict[best_var], ", AIC_p = ", aic_p)
    while aic_p < 0.05 :
        for pow_var in pow_vars:
            model_now = "{ml} + {pv})".format(ml=model_left, pv=pow_var)
            print(model_now)
            res = smf.mixedlm('{}'.format(model_now), data=NEMO, groups=NEMO['Subject']).fit(reml=False)
            aic_dict[pow_var] = res.aic
            aicd_dict[pow_var] = last_aic - res.aic
        aic_dict_t = {v: k for k,v in aic_dict.items()}
        ix = ix + 1
        print("Iteration/Variable {}".format(ix))
        print("AICs: ", aic_dict)
        print("AIC Deltas: ", aicd_dict)
        best_aic = np.array(list(aic_dict.values())).min()
        best_var = aic_dict_t[best_aic]
        print("Best Variable is: ", best_var)
        aic_p = aic_pval(best_aic, last_aic)
        if aic_p < 0.05 :
            model_bef = "{ml} + {bv})".format(ml=model_left, bv=best_var)
            model_left = "{ml} + {bv}".format(ml=model_left, bv=best_var)
            pow_vars.remove(best_var)
            last_aic = best_aic
            print("Current best model: ", model_bef, ", AIC = ", best_aic, ", AIC_delta = ", aicd_dict[best_var], ", AIC_p = ", aic_p)
print("Optimization complete.")
print("Optimal model is: ", model_bef)
res_opt = smf.mixedlm('{}'.format(model_bef), data=NEMO, groups=NEMO['Subject']).fit(reml=False)
print(res_opt.summary())
print("Opt Model PseudoR2 = ", pseudo_r2(res_opt,obsvals))
