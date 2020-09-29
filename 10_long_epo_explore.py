# look at Grand Averages of Evoked, and of TFR
import mne
import numpy as np

# define file locations
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

# # the frequency bands used in dictionary form
# freqs = {"theta":list(np.arange(3,8)),"alpha":list(np.arange(8,14)),"beta_low":list(np.arange(14,22)),
#          "beta_high":list(np.arange(22,31)),"gamma":list(np.arange(31,47))}
# freqs_g = {"gamma_high":list(np.arange(65,96,2))}
# cycles = {"theta":5,"alpha":7,"beta_low":9,"beta_high":11,"gamma":13}
# cycles_g = {"gamma_high":15}
#
# # the frequencies passed as lists (for TFR calculation)
# freqs_n = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
# cycs_n = [5, 5, 5, 5, 5, 7, 7,  7,  7,  7,  7,  9,  9,  9,  9,  9,  9,  9,  9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13]
# freqs_g = [65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95]
# cycs_g = [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
# fmins = [3, 8, 14, 22, 31]
# fmaxs = [7, 13, 21, 30, 46]

# load the epos
epos = []
for sub in subjs:
    epo = mne.read_epochs("{}{}_long-epo.fif".format(proc_dir,sub))
    epo.info['bads'] = epo.info['bads'] + ['A51']
    epo.interpolate_bads()
    epos.append(epo)
# get layout for plotting later
layout = mne.find_layout(epo.info)
mag_names = [epo.ch_names[p] for p in mne.pick_types(epo.info, meg=True)]
layout.names = mag_names

# # do the GA Evoked (neg, pos, and contrast)
# # get lists and Evoked-containers started
# negevs = [epos[0]['negative'].average().apply_baseline()]
# posevs = [epos[0]['positive'].average().apply_baseline()]
# contrevs = [negevs[0].copy()]
# contrevs[0].data = negevs[0].data - posevs[0].data
# contrevs[0].comment = 'contrast neg-pos'
# # then loop through remaining epos for averaging
# for epo in epos[1:]:
#     negev = epo['negative'].average().apply_baseline()
#     posev = epo['positive'].average().apply_baseline()
#     contrev = negev.copy()
#     contrev.data = negev.data - posev.data
#     contrev.comment = 'contrast neg-pos'
#     negevs.append(negev)
#     posevs.append(posev)
#     contrevs.append(contrev)
# # calc GAs and plot
# GA_neg = mne.grand_average(negevs)
# GA_neg.plot_joint()
# GA_pos = mne.grand_average(posevs)
# GA_pos.plot_joint()
# GA_cont = mne.grand_average(contrevs)
# GA_cont.plot_joint()

# calculate TFRs per subject
freqs = np.arange(3,47,1)
n_cycles = 7
neg_TFRs = []
pos_TFRs = []
cont_TFRs = []
for epo in epos:
    neg_TFR = mne.time_frequency.tfr_morlet(epo['negative'], freqs, n_cycles, use_fft=False, return_itc=False, decim=1, n_jobs=6, picks=None, zero_mean=True, average=True, output='power')
    neg_TFR.apply_baseline((None,0),mode='percent')
    neg_TFRs.append(neg_TFR)
    pos_TFR = mne.time_frequency.tfr_morlet(epo['positive'], freqs, n_cycles, use_fft=False, return_itc=False, decim=1, n_jobs=6, picks=None, zero_mean=True, average=True, output='power')
    pos_TFR.apply_baseline((None,0),mode='percent')
    pos_TFRs.append(pos_TFR)
    cont_TFR = neg_TFR.copy()
    cont_TFR.data = neg_TFR.data - pos_TFR.data
    cont_TFRs.append(cont_TFR)
GA_neg_TFR = mne.grand_average(neg_TFRs)
GA_neg_TFR.save("{}GA_TFR_negative-tfr.h5".format(proc_dir))
GA_neg_TFR.plot_topo(baseline=None,mode=None,layout=layout,fig_facecolor='white',font_color='black')
GA_pos_TFR = mne.grand_average(pos_TFRs)
GA_pos_TFR.save("{}GA_TFR_positive-tfr.h5".format(proc_dir))
GA_pos_TFR.plot_topo(baseline=None,mode=None,layout=layout,fig_facecolor='white',font_color='black')
GA_cont_TFR = mne.grand_average(cont_TFRs)
GA_cont_TFR.save("{}GA_TFR_contrast_N-P-tfr.h5".format(proc_dir))
GA_cont_TFR.plot_topo(baseline=None,mode=None,layout=layout,fig_facecolor='white',font_color='black',vmin=-1,vmax=1)
# just during pictures
GA_cont_TFR.plot_topo(baseline=None,mode=None,layout=layout,fig_facecolor='white',font_color='black',tmin=None,tmax=4.0,fmin=5,fmax=35,vmin=-1,vmax=1)
# then tones
GA_cont_TFR.plot_topo(baseline=None,mode=None,layout=layout,fig_facecolor='white',font_color='black',tmin=3.4,tmax=11.8,fmin=5,fmax=35,vmin=-1,vmax=1)
# maybe produce a nice 'joint' plot for peaks for publication
#GA_cont_TFR.plot_joint(timefreqs={(1, 10): (0.1, 2)},baseline=None,mode=None,title="GA - TFR response Negative-Positive")  # set favorite topo peaks here, value tuple sets windows centered on time and freq

# do TFR analysis for HIGH GAMMA BAND
# calculate TFRs per subject
freqs_g = np.arange(65,96,2)
n_cycles_g = 15
neg_TFRs_g = []
pos_TFRs_g = []
cont_TFRs_g = []
for epo in epos:
    neg_TFR_g = mne.time_frequency.tfr_morlet(epo['negative'], freqs_g, n_cycles_g, use_fft=False, return_itc=False, decim=1, n_jobs=6, picks=None, zero_mean=True, average=True, output='power')
    neg_TFR_g.apply_baseline((None,0),mode='percent')
    neg_TFRs_g.append(neg_TFR_g)
    pos_TFR_g = mne.time_frequency.tfr_morlet(epo['positive'], freqs_g, n_cycles_g, use_fft=False, return_itc=False, decim=1, n_jobs=6, picks=None, zero_mean=True, average=True, output='power')
    pos_TFR_g.apply_baseline((None,0),mode='percent')
    pos_TFRs_g.append(pos_TFR_g)
    cont_TFR_g = neg_TFR_g.copy()
    cont_TFR_g.data = neg_TFR_g.data - pos_TFR_g.data
    cont_TFRs_g.append(cont_TFR_g)
GA_neg_TFR_g = mne.grand_average(neg_TFRs_g)
GA_neg_TFR_g.save("{}GA_TFR_negative_gamma-tfr.h5".format(proc_dir),overwrite=True)
GA_neg_TFR_g.plot_topo(baseline=None,mode=None,layout=layout,fig_facecolor='white',font_color='black',vmin=-0.5,vmax=0.5)
GA_pos_TFR_g = mne.grand_average(pos_TFRs_g)
GA_pos_TFR_g.save("{}GA_TFR_positive_gamma-tfr.h5".format(proc_dir),overwrite=True)
GA_pos_TFR_g.plot_topo(baseline=None,mode=None,layout=layout,fig_facecolor='white',font_color='black',vmin=-0.5,vmax=0.5)
GA_cont_TFR_g = mne.grand_average(cont_TFRs_g)
GA_cont_TFR_g.save("{}GA_TFR_contrast_N-P_gamma-tfr.h5".format(proc_dir),overwrite=True)
GA_cont_TFR_g.plot_topo(baseline=None,mode=None,layout=layout,fig_facecolor='white',font_color='black',vmin=-0.5,vmax=0.5)
# just during pictures
GA_cont_TFR_g.plot_topo(baseline=None,mode=None,layout=layout,fig_facecolor='white',font_color='black',tmin=None,tmax=4.0,vmin=-0.5,vmax=0.5)
# then tones
GA_cont_TFR_g.plot_topo(baseline=None,mode=None,layout=layout,fig_facecolor='white',font_color='black',tmin=3.4,tmax=11.8,vmin=-0.5,vmax=0.5)
# maybe produce a nice 'joint' plot for peaks for publication
#GA_cont_TFR_g.plot_joint(timefreqs={(1, 10): (0.1, 2)},baseline=None,mode=None,title="GA - TFR response Negative-Positive")  # set favorite topo peaks here, value tuple sets windows centered on time and freq
