# look at Grand Averages of Evoked, and of TFR
import mne
import numpy as np
from mne.beamformer import tf_dics
from mne.viz import plot_source_spectrogram

# set directories
proc_dir = "D:/NEMO_analyses_new/proc/"
mri_dir = "D:/freesurfer/subjects/"
plot_dir = "D:/NEMO_analyses_new/plots/TF_dics/"
save_dir = "D:/NEMO_analyses_new/proc/TF_dics/"
sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13",
           "NEM_16":"KIO12","NEM_17":"DEN59","NEM_18":"SAG13","NEM_20":"PAG48","NEM_22":"EAM11",
           "NEM_23":"FOT12","NEM_24":"BII41","NEM_26":"ENR41","NEM_27":"HIU14","NEM_28":"WAL70",
           "NEM_29":"KIL72","NEM_31":"BLE94","NEM_34":"KER27","NEM_35":"MUN79","NEM_36":"BRA52_fa"}  ## order got corrected
excluded = {"NEM_30":"DIU11","NEM_32":"NAG83","NEM_33":"FAO18_fa","NEM_37":"EAM67","NEM_19":"ALC81","NEM_21":"WKI71_fa"}
# sub_dict = {"NEM_10":"GIZ04"}


# set parameters for TF-DICS beamformer
# frequency & time parameters
# frequencies = [[4.], [5.,6.], [7.,8.], [9.,10.,11.,12.,13.], [14.,15.,16.,17.,18.,19.,20.], [21.,22.,23.,24.,25.,26.,27.,28.,29.,30.,31.],[32.,33.,34.,35.,36.,37.,38.,39.,40.,41.,42.,43.,44.,45.,46.]]
frequencies = [[4.,5.], [5.,6.,7.], [7.,8.,9.], [9.,10.,11.,12.,13.,14.], [14.,15.,16.,17.,18.,19.,20.,21.], [21.,22.,23.,24.,25.,26.,27.,28.,29.,30.,31.,32.],[32.,33.,34.,35.,36.,37.,38.,39.,40.,41.,42.,43.,44.,45.,46.]]
freq_bins = [(i[0],i[-1]) for i in frequencies]  # needed for plotting !!
cwt_n_cycles = [4., 5., 6., 7., 8., 9., 9.]
#cwt_n_cycles = [[4], [5,5], [6,6], [7,7,7,7,7], [8,8,8,8,8,8,8], [9,9,9,9,9,9,9,9,9,9,9], [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9]]
win_lengths = [1, 1, 1, 1, 0.8, 0.6, 0.5]
tmin = -0.2
tmax = 11.8
tstep = 0.1
tmin_plot = 0
tmax_plot = 11.6
n_jobs = 4
# label of interest
loi = 'transversetemporal-rh'

coll_stcs_neg = []
coll_stcs_pos = []
coll_stcs_cont = []
GA_stcs_neg = []
GA_stcs_pos = []
GA_stcs_cont = []

for meg,mri in sub_dict.items():
    epo = mne.read_epochs("{}{}_long-epo.fif".format(proc_dir,meg))
    epo.info['bads'] = epo.info['bads'] + ['A51']
    fwd = mne.read_forward_solution("{}{}-fwd.fif".format(proc_dir,meg))
    labels = mne.read_labels_from_annot(mri, parc='aparc', hemi='both', surf_name='white', annot_fname=None, regexp=None, subjects_dir=mri_dir, sort=False, verbose=None)
    label = [l for l in labels if l.name == loi][0]
    stcs_neg = tf_dics(epo['negative'], fwd, noise_csds = None, tmin=tmin, tmax=tmax, tstep=tstep, win_lengths=win_lengths, subtract_evoked=False, mode='cwt_morlet', frequencies=frequencies, cwt_n_cycles=cwt_n_cycles,
                       reg = 0.05, label=label, pick_ori='max-power', inversion='single', depth=1.0, n_jobs=n_jobs)
    coll_stcs_neg.append(stcs_neg)
    for fb in range(len(freq_bins)):
        stcs_neg[fb].save("{}{}_TF_dics_neg_{}-{}_{}-stc.h5".format(save_dir,meg,freq_bins[fb][0],freq_bins[fb][-1],loi))
    fig_n = plot_source_spectrogram(stcs_neg, freq_bins, tmin=tmin_plot, tmax=tmax_plot, source_index=None, colorbar=True, cmap='hot_r', vmin=0, vmax=6e-26, show=False)    # freq_bins! or error
    fig_n.savefig("{}{}_neg_{}.png".format(plot_dir,meg,loi))
    stcs_pos = tf_dics(epo['positive'], fwd, noise_csds = None, tmin=tmin, tmax=tmax, tstep=tstep, win_lengths=win_lengths, subtract_evoked=False, mode='cwt_morlet', frequencies=frequencies, cwt_n_cycles=cwt_n_cycles,
                       reg = 0.05, label=label, pick_ori='max-power', inversion='single', depth=1.0, n_jobs=n_jobs)
    coll_stcs_pos.append(stcs_pos)
    for fb in range(len(freq_bins)):
        stcs_pos[fb].save("{}{}_TF_dics_pos_{}-{}_{}-stc.h5".format(save_dir,meg,freq_bins[fb][0],freq_bins[fb][-1],loi))
    fig_p = plot_source_spectrogram(stcs_pos, freq_bins, tmin=tmin_plot, tmax=tmax_plot, source_index=None, colorbar=True, cmap='hot_r', vmin=0, vmax=6e-26, show=False)
    fig_p.savefig("{}{}_pos_{}.png".format(plot_dir,meg,loi))
    # now calculate the difference between conditions
    stcs_diff = [(a[0]-a[1]) for a in zip(stcs_neg,stcs_pos)]
    coll_stcs_cont.append(stcs_diff)
    for fb in range(len(freq_bins)):
        stcs_diff[fb].save("{}{}_TF_dics_cont_N-P_{}-{}_{}-stc.h5".format(save_dir,meg,freq_bins[fb][0],freq_bins[fb][-1],loi))
    fig_c = plot_source_spectrogram(stcs_diff, freq_bins, tmin=tmin_plot, tmax=tmax_plot, source_index=None, colorbar=True, cmap='seismic', vmin=-1e-26, vmax=1e-26, show=False)
    fig_c.savefig("{}{}_cont_N-P_{}.png".format(plot_dir,meg,loi))

# GA for negatives
# have to average over vertices in single subjects first
for s in range(len(coll_stcs_neg)):
    for f in range(len(coll_stcs_neg[s])):
        avg = np.mean(coll_stcs_neg[s][f].data,axis=0,dtype='float64')
        avg = np.expand_dims(avg,axis=0)  # stc object needs first dimension for loc
        # now reduce vertices attribute to a single vertex to avoid error message on filling averaged data in -- NOTE: here [1], because RH, change to [0], when label is in LH (or to other, when in subcortical volume src)
        coll_stcs_neg[s][f].vertices[1] = np.array([coll_stcs_neg[0][f].vertices[1][0]], dtype='int64')  # use vertex no. of 1st subject for all to avoid error message later
        coll_stcs_neg[s][f].data = avg
GA_stcs_neg = coll_stcs_neg[0]
for s in range(len(coll_stcs_neg[1:])):
    GA_stcs_neg[f].data += coll_stcs_neg[s][f].data
GA_stcs_neg[f].data /= len(coll_stcs_neg)
for fb in range(len(freq_bins)):
    GA_stcs_neg[fb].save("{}GA_TF_dics_neg_{}-{}_{}-stc.h5".format(save_dir,freq_bins[fb][0],freq_bins[fb][-1],loi))
GA_fig_n = plot_source_spectrogram(GA_stcs_neg, freq_bins, tmin=tmin_plot, tmax=tmax_plot, source_index=None, colorbar=True, cmap='hot_r', vmin=0, vmax=6e-26, show=False)    # freq_bins! or error
GA_fig_n.savefig("{}GA_neg_{}.png".format(plot_dir,loi))

# GA for positives
# have to average over vertices in single subjects first
for s in range(len(coll_stcs_pos)):
    for f in range(len(coll_stcs_pos[s])):
        avg = np.mean(coll_stcs_pos[s][f].data,axis=0,dtype='float64')
        avg = np.expand_dims(avg,axis=0)  # stc object needs first dimension for loc
        # now reduce vertices attribute to a single vertex to avoid error message on filling averaged data in -- NOTE: here [1], because RH, change to [0], when label is in LH (or to other, when in subcortical volume src)
        coll_stcs_pos[s][f].vertices[1] = np.array([coll_stcs_pos[0][f].vertices[1][0]], dtype='int64')  # use vertex no. of 1st subject for all to avoid error message later
        coll_stcs_pos[s][f].data = avg
GA_stcs_pos = coll_stcs_pos[0]
for s in range(len(coll_stcs_pos[1:])):
    GA_stcs_pos[f].data += coll_stcs_pos[s][f].data
GA_stcs_pos[f].data /= len(coll_stcs_pos)
for fb in range(len(freq_bins)):
    GA_stcs_pos[fb].save("{}GA_TF_dics_pos_{}-{}_{}-stc.h5".format(save_dir,freq_bins[fb][0],freq_bins[fb][-1],loi))
GA_fig_p = plot_source_spectrogram(GA_stcs_pos, freq_bins, tmin=tmin_plot, tmax=tmax_plot, source_index=None, colorbar=True, cmap='hot_r', vmin=0, vmax=6e-26, show=False)    # freq_bins! or error
GA_fig_p.savefig("{}GA_pos_{}.png".format(plot_dir,loi))

# GA for contrast
# build a new list of contrasts from subtracting the averages N-P over vertices in the label for each subject
coll_stcs_cont_av = coll_stcs_neg.copy()
for s in range(len(coll_stcs_neg)):
    coll_stcs_cont_av[s] = [(a[0]-a[1]) for a in zip(coll_stcs_neg[s],coll_stcs_pos[s])]
GA_stcs_cont = coll_stcs_cont_av[0]
for s in range(len(coll_stcs_cont_av[1:])):
    GA_stcs_cont[f].data += coll_stcs_cont_av[s][f].data
GA_stcs_cont[f].data /= len(coll_stcs_cont_av)
for fb in range(len(freq_bins)):
    GA_stcs_cont[fb].save("{}GA_TF_dics_cont_N-P_{}-{}_{}-stc.h5".format(save_dir,freq_bins[fb][0],freq_bins[fb][-1],loi))
GA_fig_c = plot_source_spectrogram(GA_stcs_cont, freq_bins, tmin=tmin_plot, tmax=tmax_plot, source_index=None, colorbar=True, cmap='seismic', vmin=-1e-26, vmax=1e-26, show=False)    # freq_bins! or error
GA_fig_c.savefig("{}GA_cont_N-P_{}.png".format(plot_dir,loi))


## make a cluster-permutation T-test (N-P) for each freq_bin, looking for sign. time clusters
for f in range(len(freq_bins)):
    dat_list = []
    for s in range(len(coll_stcs_cont_av)):
        dat_list.append(coll_stcs_cont_av[s][f].data)
    dat_array = np.array(dat_list)
    dat_array = np.squeeze(dat_array)
    t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(dat_array,threshold=None, n_permutations=1024, tail=0, stat_fun=None, adjacency=None, n_jobs=6, step_down_p=0.05, t_power=1, out_type='indices')
