import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## Purpose: read R mboost summary output into python for exploration

# build containers for coefficients and selection frequencies
coefs = {}
sel_freqs = {}

# read in the text file with the mboost summary printout
with open("/home/cora/hdd/MEG/NEMO_analyses_new/colin/TonAng_orig_results", "r") as f:
    lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    x_lines = [l.split() for l in lines]
    # get indices of the coefficient and selection frequency start lines and then the related data
    coef_ix = x_lines.index(['Coefficients:'])
    sel_freqs_ix = x_lines.index(['Selection', 'frequencies:'], coef_ix)
    coef_lines = x_lines[(coef_ix + 1) : (sel_freqs_ix - 3)]
    sel_freqs_lines = x_lines[(sel_freqs_ix + 1) : ]
    # now collect the data together
    for i in range(int(len(coef_lines)/2)):
        ix_var = i*2
        ix_val = ix_var + 1
        for var_i, var in enumerate(coef_lines[ix_var]):
            coefs[var[1:]] = coef_lines[ix_val][var_i]
    for i in range(int(len(sel_freqs_lines)/2)):
        ix_var = i*2
        ix_val = ix_var + 1
        for var_i, var in enumerate(sel_freqs_lines[ix_var]):
            sel_freqs[var[1:]] = sel_freqs_lines[ix_val][var_i]
    # put the Intercept into its own variable
    intercept = coefs.pop('Intercept)')

# build a dataframe with Variable, Coef, Sel_Freq columns and fill in the data

res = pd.DataFrame(columns=['Variable','Coef','Sel_Freq'])
for i, (k, v) in enumerate(coefs.items()):
    res.loc[i,'Variable'] = k
    res.loc[i,'Coef'] = v
    res.loc[i,'Sel_Freq'] = sel_freqs[k]

# make sure numbers are converted
res = res.astype({'Coef':'float64','Sel_Freq':'float64'})

# NOW EXPLORE THE RESULTS

# e.g. plot a scatterplot of all the coefs with horizontal lines as .02/-.02 and explore with mouseover..
res.plot(x='Variable',y='Coef',kind='scatter')
plt.axhline(y=0.02, color='r', linestyle='-')
plt.axhline(y=-0.02, color='r', linestyle='-')
# plt.show()
