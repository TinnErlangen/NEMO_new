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
# aicd_thresh = 5
def aic_pval(a,b):
    return np.exp((a - b)/2)   # calculates the evidence ratio for 2 aic values
def pseudo_r2(m, y):    # calculates the pseudo r2 from model summary(m) and observed vals (y)
    fitvals = m.fittedvalues
    r , v = stats.pearsonr(fitvals, y)
    return r**2

# Building Otimal Models for each Power Node
for pow_var in pow_vars:
    d_var = pow_var
    obsvals = NEMO[d_var]
    p_vars = ["LO_a_pow","LO_b_pow","IP_a_pow","IP_b_pow","IT_a_pow","IT_b_pow","SM_a_pow","SM_b_pow","TT_a_pow","TT_b_pow"]
    p_vars.remove(d_var)

    # Null model
    print("ANALYSES FOR {}".format(d_var))
    model = "{dv} ~ 1".format(dv=d_var)
    res_0 = smf.mixedlm('{}'.format(model), data=NEMO, groups=NEMO['Subject']).fit(reml=False)
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
    print("Ton model results -- AIC = ", ton_aic, ", AIC_delta = ", null_aic - ton_aic, ", AIC_p = ", aic_pval(ton_aic,null_aic), ", PseudoR2 = ", pseudo_r2(res_ton,obsvals))
    print("EMO")
    model = "{dv} ~ {e}".format(dv=d_var, e=emo)
    res_emo = smf.mixedlm('{}'.format(model), data=NEMO, groups=NEMO['Subject']).fit(reml=False)
    emo_aic = res_emo.aic
    print("Emo model results -- AIC = ", emo_aic, ", AIC_delta = ", null_aic - emo_aic, ", AIC_p = ", aic_pval(emo_aic,null_aic), ", PseudoR2 = ", pseudo_r2(res_emo,obsvals))

    # Finding the Optimal Model with Power Variables
    print("Finding the Optimal Model with Power Variables..")
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
        res = smf.mixedlm('{}'.format(model_now), data=NEMO, groups=NEMO['Subject']).fit(reml=False)
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
        # TEST Emotion
        res_emo = smf.mixedlm('{mb} + Emo'.format(mb=model_bef), data=NEMO, groups=NEMO['Subject']).fit(reml=False)
        emo_aic = res_emo.aic
        emo_aicp = aic_pval(emo_aic,last_aic)
        print("Testing Emotion again -- AIC = ", emo_aic, ", AIC_delta = ", last_aic - emo_aic, ", AIC_p = ", emo_aicp)
        if emo_aicp < 0.05:
            print("EMO stays in the model.")
        else:
            print("No independent emotion effect remains.")

        # now Iterate until Best Model
        while aic_p < 0.05 :
            ix = ix + 1
            print("Iteration/Variable {}".format(ix))
            for p_var in p_vars:
                model_now = "{mb} + {pv}".format(mb=model_bef, pv=p_var)
                print(model_now)
                res = smf.mixedlm('{}'.format(model_now), data=NEMO, groups=NEMO['Subject']).fit(reml=False)
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
                # TEST Emotion
                res_emo = smf.mixedlm('{mb} + Emo'.format(mb=model_bef), data=NEMO, groups=NEMO['Subject']).fit(reml=False)
                emo_aic = res_emo.aic
                emo_aicp = aic_pval(emo_aic,last_aic)
                print("Testing Emotion again -- AIC = ", emo_aic, ", AIC_delta = ", last_aic - emo_aic, ", AIC_p = ", emo_aicp)
                if emo_aicp < 0.05:
                    print("EMO stays in the model.")
                else:
                    print("No independent emotion effect remains.")

    print("Optimization for {dv} complete.".format(dv=d_var))
    print("Optimal model is: ", model_bef)
    res_opt = smf.mixedlm('{}'.format(model_bef), data=NEMO, groups=NEMO['Subject']).fit(reml=False)
    print(res_opt.summary())
    print("Opt Model PseudoR2 = ", pseudo_r2(res_opt,obsvals))
