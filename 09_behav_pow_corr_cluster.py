import numpy as np
import mne
import pandas as pd
import random
from scipy import stats
from mayavi import mlab
import matplotlib.pyplot as plt
plt.ion()

from sklearn.linear_model import LinearRegression

from mne.stats.cluster_level import _setup_adjacency, _find_clusters, \
    _reshape_clusters

mne.viz.set_3d_backend('pyvista')

# setup files and folders, subject lists

proc_dir = "/home/cora/hdd/MEG/NEMO_analyses_new/proc/"
mri_dir = "/home/cora/hdd/MEG/freesurfer/subjects/"
sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13",
           "NEM_16":"KIO12","NEM_17":"DEN59","NEM_18":"SAG13","NEM_20":"PAG48","NEM_22":"EAM11",
           "NEM_23":"FOT12","NEM_24":"BII41","NEM_26":"ENR41","NEM_27":"HIU14","NEM_28":"WAL70",
           "NEM_29":"KIL72","NEM_31":"BLE94","NEM_34":"KER27","NEM_35":"MUN79","NEM_36":"BRA52_fa"}  ## order got corrected
excluded_dict = {"NEM_30":"DIU11","NEM_32":"NAG83","NEM_33":"FAO18_fa","NEM_37":"EAM67","NEM_19":"ALC81","NEM_21":"WKI71_fa"}
subjs = ["NEM_10","NEM_11","NEM_12","NEM_14","NEM_15",
         "NEM_16","NEM_17","NEM_18","NEM_20","NEM_22",
         "NEM_23","NEM_24","NEM_26","NEM_27","NEM_28",
         "NEM_29","NEM_31","NEM_34","NEM_35","NEM_36"]
#subjs = ["NEM_10","NEM_11"]
save_dir = "/home/cora/hdd/MEG/NEMO_analyses_new/plots/behav_corr_new/"  # for saving plots

# values to choose from
freq_bands = {"theta":list(np.arange(3,8)),"alpha":list(np.arange(8,14)),"beta_low":list(np.arange(14,22)),
              "beta_high":list(np.arange(22,31)),"gamma":list(np.arange(31,47))}
freqs_g_band = {"gamma_high":list(np.arange(65,96,2))}
freqs_ix = {"theta": 0, "alpha": 1, "beta_low": 2, "beta_high": 3, "gamma": 4}
contrasts = ['pic_n-pic_p','ton_n-ton_p']
beh_vars = ['Pic_Val', 'Pic_Ars', 'Ton_Laut', 'Ton_Ang','Emo_Val', 'Emo_Ars', 'ER_ges', 'Angst_ges', 'Psycho_ges']

# an stc.data is array of n_dipoles x n_time ...
# that's why I collected the stc.data.T of every subject into a X_freq_diff array to do the cluster stats on ...
# however, I only saved that X_....npy for the gamma_high freq band and for all the label analyses..

## SET THE VARIABLES : Behav, Contrast & Frequency
contrast = 'ton_n-ton_p'
beh_cond = 'Ton_Ang'
freq = "gamma"
if freq != "gamma_high":
    ix = freqs_ix[freq]
print("Correlation Cluster Analysis")
print("Contrast:   {}".format(contrast))
print("Behav Variable:   {}".format(beh_cond))
print("Frequency:   {}".format(freq))

# get the behavioral data array ready & choose the variable
N_behav = pd.read_csv('{}NEMO_behav.csv'.format(proc_dir))
Behav = np.array(N_behav[beh_cond])


# prepare the data arrays / objects needed
all_diff_plot = []  # list for averaging and plotting group STC
X_diff = []  #  list for collecting data for cluster stat analyses
for sub in subjs:
    # load the STC data
    if freq == "gamma_high":
        stc_fsavg_diff = mne.read_source_estimate("{}{}_STC_fs_{}_gamma-stc".format(proc_dir,sub,contrast), subject='fsaverage')
        stc_fsavg_diff = stc_fsavg_diff.surface()
    else:
        stc_fsavg_diff = mne.read_source_estimate("{}{}_STC_fs_{}-stc".format(proc_dir,sub,contrast), subject='fsaverage')
        stc_fsavg_diff = stc_fsavg_diff.surface()
    # collect the individual stcs into lists
    all_diff_plot.append(stc_fsavg_diff)
    X_diff.append(stc_fsavg_diff.data.T)

# create group average stc for plotting later
stc_sum = all_diff_plot.pop()
for stc in all_diff_plot:
    stc_sum = stc_sum + stc
NEM_all_stc_diff = stc_sum / len(subjs)

# make data array for cluster permutation stats N-P stc vals
X_diff = np.array(X_diff).squeeze()
# get data for right frequency
if freq != "gamma_high":
    X_diff = X_diff[:,ix,:]
X_diff = np.array(X_diff).squeeze()

# calculate Pearson's r for each vertex to Behavioral variable of the subject
X_Rval = np.empty(X_diff.shape[1])
X_R_Tval = np.empty(X_diff.shape[1])
for vert_idx in range(X_diff.shape[1]):
    X_Rval[vert_idx], p = stats.pearsonr(X_diff[:,vert_idx],Behav)
# calculate an according t-value for each r
X_R_Tval = (X_Rval * np.sqrt((len(subjs)-2))) / np.sqrt(1 - X_Rval**2)

# setup for clustering -- t-thresholds for N=20: 2.086 (.05), 2.845 (.01), or 3.850 (.001)
threshold = 2.845
# load fsaverage source space to morph to; prepare fsaverage adjacency matrices for cluster permutation analyses (1 surface, 1 volume)
fs_src = mne.read_source_spaces("{}fsaverage_oct6_mix-src.fif".format(proc_dir))
fs_lh = fs_src.pop(0)
fs_rh = fs_src.pop(0)
fs_surf = [fs_lh]+[fs_rh]
adjacency_s = mne.spatial_src_adjacency(fs_surf)
fs_surf_vertices = [s['vertno'] for s in fs_surf]
fs_vol = mne.read_source_spaces("{}fsaverage_limb-src.fif".format(proc_dir))
adjacency_v = mne.spatial_src_adjacency(fs_vol)
fs_limb_vertices = [s['vertno'] for s in fs_vol]
fs_src = mne.read_source_spaces("{}fsaverage_oct6_mix-src.fif".format(proc_dir))  # have to reload after popping

# find clusters in the T-vals
print("Looking for clusters...")
clusters, cluster_stats = _find_clusters(X_R_Tval,threshold=threshold,
                                               adjacency=adjacency_s,
                                               tail=0)
print("Found {} initial clusters...".format(len(clusters)))
print("Max Cluster T-Sum is {}".format(cluster_stats.max()))
print("Min Cluster T-Sum is {}".format(cluster_stats.min()))

# do the random sign flip permutation
# setup
n_perms = 1000
cluster_H0 = np.zeros(n_perms)
# here comes the loop
print("Now starting {} permutations".format(n_perms))
for i in range(n_perms):
    if i in [10,20,50,100,200,300,400,500,600,700,800,900]:
        print("{} th iteration".format(i))
    # permute the behavioral values over subjects
    Beh_perm = random.sample(list(Behav),k=len(subjs))
    # calculate Pearson's r for each vertex to Behavioral variable of the subject
    XP_Rval = np.empty(X_diff.shape[1])
    XP_R_Tval = np.empty(X_diff.shape[1])
    for vert_idx in range(X_diff.shape[1]):
        XP_Rval[vert_idx], p = stats.pearsonr(X_diff[:,vert_idx],Beh_perm)
    # calculate an according t-value for each r
    XP_R_Tval = (XP_Rval * np.sqrt((len(subjs)-2))) / np.sqrt(1 - XP_Rval**2)
    # now find clusters in the T-vals
    perm_clusters, perm_cluster_stats = _find_clusters(XP_R_Tval,threshold=threshold,
                                                   adjacency=adjacency_s,
                                                   tail=0)
    if len(perm_clusters):
        cluster_H0[i] = np.abs(perm_cluster_stats).max()     # this should be changed to cluster_H0[i] = np.abs(perm_cluster_stats).max() for 2-tailed version
        # cluster_H0[i] = np.abs(perm_cluster_stats.min())
    else:
        cluster_H0[i] = np.nan

# get upper CI bound from cluster mass H0
clust_threshold = np.quantile(cluster_H0[~np.isnan(cluster_H0)], [.95])
print("T-Sum Threshold after permutation is {}".format(clust_threshold))
# good cluster inds
good_cluster_inds = np.where(np.abs(cluster_stats) > clust_threshold)[0]
print("{} good clusters remained.".format(len(good_cluster_inds)))

# # plot uncorrected correlation t-values on fsaverage
# X_R_Tval = np.expand_dims(X_R_Tval, axis=1)
# NEM_all_stc_diff.data = X_R_Tval
# NEM_all_stc_diff.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,show_traces=False,clim={'kind':'value','pos_lims':(2,4,6)})

# # then plot good clusters
# if len(good_cluster_inds):
#     for n,idx in enumerate(np.nditer(good_cluster_inds)):
#         temp_data = np.zeros((NEM_all_stc_diff.data.shape[0],1))
#         # for n,idx in enumerate(np.nditer(good_cluster_inds)):
#         #     temp_data[clusters[idx],n] = NEM_all_stc_diff.data[clusters[idx],0]
#         temp_data[clusters[idx],0] = NEM_all_stc_diff.data[clusters[idx],0]
#         temp_data[np.abs(temp_data)>0] = 1
#         stc_clu = NEM_all_stc_diff.copy()
#         stc_clu.data = temp_data
#         stc_clu.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,colormap='coolwarm',clim={'kind':'value','pos_lims':(0,0.5,1)})
#         # plot and save figs on inflated brain with annotation
#         fig = mlab.figure(size=(300, 300))
#         brain = stc_clu.plot(subjects_dir=mri_dir,subject='fsaverage',surface='inflated',hemi='both',colormap='coolwarm',clim={'kind':'value','pos_lims': (0,0.5,1)},figure=fig)
#         brain.add_annotation('HCPMMP1_combined', borders=1, alpha=0.9)
#         mlab.view(0, 90, 450, [0, 0, 0])
#         mlab.savefig('{d}{c}_corr_{f}_clu_{n}_rh.png'.format(d=save_dir,c=cond,f=freq,n=n), magnification=4)
#         mlab.view(180, 90, 450, [0, 0, 0])
#         mlab.savefig('{d}{c}_corr_{f}_clu_{n}_lh.png'.format(d=save_dir,c=cond,f=freq,n=n), magnification=4)
#         mlab.view(180, 0, 450, [0, 10, 0])
#         mlab.savefig('{d}{c}_corr_{f}_clu_{n}_top.png'.format(d=save_dir,c=cond,f=freq,n=n), magnification=4)
#         mlab.view(180, 180, 480, [0, 10, 0])
#         mlab.savefig('{d}{c}_corr_{f}_clu_{n}_bottom.png'.format(d=save_dir,c=cond,f=freq,n=n), magnification=4)
#         mlab.close(fig)
# else: print("No sign. clusters found")
