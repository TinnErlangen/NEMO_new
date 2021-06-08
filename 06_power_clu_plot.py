## Power analyses

import mne
import numpy as np
# mne.viz.set_3d_backend('pyvista')
mne.viz.set_3d_backend('mayavi')
# import os
# os.environ['ETS_TOOLKIT'] = 'qt4'
# os.environ['QT_API'] = 'pyqt5'

## remember: BRA52, ((FAO18, WKI71 - excl.)) have fsaverage MRIs (originals were defective)


# proc_dir = "D:/NEMO_analyses_new/proc/"
proc_dir = "/home/cora/hdd/MEG/NEMO_analyses_new/proc/"
mri_dir = "/home/cora/hdd/MEG/freesurfer/subjects/"

sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13",
           "NEM_16":"KIO12","NEM_17":"DEN59","NEM_18":"SAG13","NEM_20":"PAG48","NEM_22":"EAM11",
           "NEM_23":"FOT12","NEM_24":"BII41","NEM_26":"ENR41","NEM_27":"HIU14","NEM_28":"WAL70",
           "NEM_29":"KIL72","NEM_31":"BLE94","NEM_34":"KER27","NEM_35":"MUN79","NEM_36":"BRA52_fa"}  ## order got corrected
excluded = {"NEM_30":"DIU11","NEM_32":"NAG83","NEM_33":"FAO18_fa","NEM_37":"EAM67","NEM_19":"ALC81","NEM_21":"WKI71_fa"}
# sub_dict = {"NEM_10":"GIZ04"}

# # the frequency bands used in dictionary form & freq_band bounds for averaging CSDs
# freqs = {"theta":list(np.arange(3,8)),"alpha":list(np.arange(8,14)),"beta_low":list(np.arange(14,22)),
#          "beta_high":list(np.arange(22,31)),"gamma":list(np.arange(31,47))}
# freqs_g = {"gamma_high":list(np.arange(65,96,2))}
# fmins = [3, 8, 14, 22, 31]
# fmaxs = [7, 13, 21, 30, 46]

# the new freq bands (from TFR) used for new DICS filters
freqs = {"theta_low":list(np.arange(4,5)),"theta_high":list(np.arange(5,7)),"alpha_low":list(np.arange(7,9)),"alpha_high":list(np.arange(9,14)),
         "beta_low":list(np.arange(14,21)),"beta_high":list(np.arange(21,32)),"gamma":list(np.arange(32,47))}
fmins = [4, 5, 7, 9, 14, 21, 32]
fmaxs = [5, 7, 9, 14, 21, 32, 46]

freq_tup = tuple(freqs.keys())

# the conditions
conditions = {'rest':'rest', 'tonbas':['tonbas','tonrat'], 'pic_n':'negative/pics', 'pic_p':'positive/pics',
              'ton_n':['negative/r1','negative/r2','negative/s1','negative/s2'], 'ton_p':['positive/r1','positive/r2','positive/s1','positive/s2']}
conds = list(conditions.keys())

## PARAMETERS of Power Group Analyses data
threshold = 2.861     ## choose initial T-threshold for clustering; based on p-value of .05 or .01 for df = (subj_n-1); with df=19 - 2.093, or 2.861
cond_a = 'ton_n'      ## specifiy the conditions to contrast
cond_b = 'ton_p'

# load fsaverage source space to morph to; prepare fsaverage adjacency matrices for cluster permutation analyses (1 surface, 1 volume)
fs_src = mne.read_source_spaces("{}fsaverage_oct6_mix-src.fif".format(proc_dir))
fs_lh = fs_src.pop(0)
fs_rh = fs_src.pop(0)
fs_surf = [fs_lh]+[fs_rh]
adjacency_s = mne.spatial_src_adjacency(fs_surf)
fs_surf_vertices = [s['vertno'] for s in fs_surf]
fs_mix = mne.read_source_spaces("{}fsaverage_oct6_mix-src.fif".format(proc_dir))  # have to reload after popping
# reduce to surface source spaces
fs_surf = fs_mix.copy()
for i in range(len(fs_surf)-2):
    a = fs_surf.pop()

# load GA difference data STCs
GA_stc_diff = mne.read_source_estimate("{}GA_fs_mix_ton_N-P_stc.h5".format(proc_dir),subject="fsaverage")
GA_stc_diff_surf = mne.read_source_estimate("{}GA_fs_surf_ton_N-P_stc.h5".format(proc_dir),subject="fsaverage")
# load data matrix for cluster permutation analysis on all frequencies
X_diff_s = np.load("{}GA_X_surf_ton_N-P_for_clu.npy".format(proc_dir))

brain = GA_stc_diff_surf.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,src=fs_surf,show_traces=False)
