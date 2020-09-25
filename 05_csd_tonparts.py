## Compute the Cross-Spectral-Density ##

import mne
import matplotlib.pyplot as plt
import numpy as np
from mne.time_frequency import csd_morlet,csd_multitaper

# define file locations
preproc_dir = "D:/NEMO_analyses_new/preproc/"
proc_dir = "D:/NEMO_analyses_new/proc/"
# pass subject and run lists
subjs_all = ["NEM_10","NEM_11","NEM_12","NEM_14","NEM_15","NEM_16",
        "NEM_17","NEM_18","NEM_19","NEM_20","NEM_21","NEM_22",
        "NEM_23","NEM_24","NEM_26","NEM_27","NEM_28",
        "NEM_29","NEM_30","NEM_31","NEM_32","NEM_33","NEM_34",
        "NEM_35","NEM_36","NEM_37"]
excluded = ["NEM_19","NEM_21","NEM_30","NEM_32","NEM_33","NEM_37"]
subjs = ["NEM_10","NEM_11","NEM_12","NEM_14","NEM_15",
         "NEM_16","NEM_17","NEM_18","NEM_20","NEM_22",
         "NEM_23","NEM_24","NEM_26","NEM_27","NEM_28",
         "NEM_29","NEM_31","NEM_34","NEM_35","NEM_36"]
# subjs = ["NEM_10"]

# here, we only add the condition csds for analyses for the tone parts 1-4
conditions = {'ton_n_part1':['negative/r1/part1','negative/r2/part1','negative/s1/part1','negative/s2/part1'], 'ton_p_part1':['positive/r1/part1','positive/r2/part1','positive/s1/part1','positive/s2/part1'],
              'ton_n_part2':['negative/r1/part2','negative/r2/part2','negative/s1/part2','negative/s2/part2'], 'ton_p_part2':['positive/r1/part2','positive/r2/part2','positive/s1/part2','positive/s2/part2'],
              'ton_n_part3':['negative/r1/part3','negative/r2/part3','negative/s1/part3','negative/s2/part3'], 'ton_p_part3':['positive/r1/part3','positive/r2/part3','positive/s1/part3','positive/s2/part3'],
              'ton_n_part4':['negative/r1/part4','negative/r2/part4','negative/s1/part4','negative/s2/part4'], 'ton_p_part4':['positive/r1/part4','positive/r2/part4','positive/s1/part4','positive/s2/part4']}

# the frequency bands used in dictionary form
freqs = {"theta":list(np.arange(3,8)),"alpha":list(np.arange(8,14)),"beta_low":list(np.arange(14,22)),
         "beta_high":list(np.arange(22,31)),"gamma":list(np.arange(31,47))}
freqs_g = {"gamma_high":list(np.arange(65,96,2))}
cycles = {"theta":5,"alpha":7,"beta_low":9,"beta_high":11,"gamma":13}
cycles_g = {"gamma_high":15}

# the frequencies passed as lists (for CSD calculation)
freqs_n = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
cycs_n = [5, 5, 5, 5, 5, 7, 7,  7,  7,  7,  7,  9,  9,  9,  9,  9,  9,  9,  9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13]
freqs_g = [65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95]
cycs_g = [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
fmins = [3, 8, 14, 22, 31]
fmaxs = [7, 13, 21, 30, 46]

# # for each subject, calc csd_all& csd_g_all
# for sub in subjs:
#     epo = mne.read_epochs("{}{}-epo.fif".format(proc_dir,sub))
#     csd_n = csd_morlet(epo, frequencies=freqs_n, n_jobs=8, n_cycles=cycs_n, decim=1)
#     csd_n.save("{}{}-csd.h5".format(proc_dir,sub))
#     csd_g = csd_morlet(epo, frequencies=freqs_g, n_jobs=8, n_cycles=cycs_g, decim=1)
#     csd_g.save("{}{}-gamma-csd.h5".format(proc_dir,sub))

# for each subject, calc csd & csd_g for each ton part x condition
for sub in subjs:
    epo = mne.read_epochs("{}{}-epo.fif".format(proc_dir,sub))
    for cond,c in conditions.items():
        csd_n = csd_morlet(epo[c], frequencies=freqs_n, n_jobs=8, n_cycles=cycs_n, decim=1)
        csd_n.save("{}{}_{}-csd.h5".format(proc_dir,sub,cond))
        csd_g = csd_morlet(epo[c], frequencies=freqs_g, n_jobs=8, n_cycles=cycs_g, decim=1)
        csd_g.save("{}{}_{}-gamma-csd.h5".format(proc_dir,sub,cond))
