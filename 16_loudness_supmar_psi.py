# calculate PSI for Supramarginal Alpha and Beta #

import numpy as np

import mne
from mne.minimum_norm import make_inverse_operator, read_inverse_operator, write_inverse_operator, apply_inverse_epochs
from mne.connectivity import seed_target_indices, phase_slope_index

import pandas as pd
import matplotlib.pyplot as plt

proc_dir = "/home/cora/hdd/MEG/NEMO_analyses_new/proc/"
mri_dir = "/home/cora/hdd/MEG/freesurfer/subjects/"

sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13",
           "NEM_16":"KIO12","NEM_17":"DEN59","NEM_18":"SAG13","NEM_20":"PAG48","NEM_22":"EAM11",
           "NEM_23":"FOT12","NEM_24":"BII41","NEM_26":"ENR41","NEM_27":"HIU14","NEM_28":"WAL70",
           "NEM_29":"KIL72","NEM_31":"BLE94","NEM_34":"KER27","NEM_35":"MUN79","NEM_36":"BRA52_fa"}

# sub_dict = {"NEM_10":"GIZ04"}

# some preps
tmin = 3.6
tmax = 11.6

snr = 1.0  # use lower SNR for single epochs
lambda2 = 1.0 / snr ** 2
method = 'dSPM'

# collection container for GA
psi_stcs = []

for meg, mri in sub_dict.items():
    epo = mne.read_epochs("{}{}_long-epo.fif".format(proc_dir,meg))
    epo.info['bads'] = epo.info['bads'] + ['A51']
    epo.apply_baseline()
    # # make inverse operator
    # data_cov = mne.compute_covariance(epo, tmin=tmin, tmax=tmax, method='empirical',n_jobs=8)
    # noise_cov = mne.compute_covariance(epo, tmin=-0.2, tmax=0, method='empirical',n_jobs=8)
    # fwd = mne.read_forward_solution("{}{}-fwd.fif".format(proc_dir,meg))
    # inv_op = make_inverse_operator(epo.info, fwd, noise_cov, loose=dict(surface=0.2, volume=1.0), depth=0.8)
    # write_inverse_operator('{}{}-oct6-inv.fif'.format(proc_dir,meg), inv_op)
    # read inverse operator
    inv_op = read_inverse_operator('{}{}-oct6-inv.fif'.format(proc_dir,meg))
    # crop the epochs to the time interval of interest (this is done in_place)
    epo.crop(tmin=tmin, tmax=tmax)
    # then apply the filters to get source activity, separately for each condition
    stcs = apply_inverse_epochs(epo, inv_op, lambda2, method, pick_ori=None, return_generator=False)
    # read labels
    labels = mne.read_labels_from_annot(mri, parc='aparc', hemi='lh', surf_name='white', annot_fname=None, regexp=None, subjects_dir=mri_dir, sort=True)
    label = [l for l in labels if l.name == 'inferiorparietal-lh']
    del labels
    # calculate label timecourse for seed region
    seed_ts = mne.extract_label_time_course(stcs, label, inv_op['src'], mode='pca_flip')        # ! had 'mean-flip' before 
    sts = [np.delete(tc, np.arange(1,13), axis=0) for tc in seed_ts]    # get rid of the subcortical labels
    del seed_ts
    # Combine the seed time course with the source estimates.
    # index 0: time course extracted from label
    # index 1..xxxx: dSPM source space time courses
    comb_ts = list(zip(sts, stcs))
    # Construct indices to estimate connectivity between the label time course and all source space time courses
    src = inv_op['src']
    del inv_op
    vertices = [src[i]['vertno'] for i in range(len(src))]
    n_signals_tot = 1 + sum([len(vertices[i]) for i,src in enumerate(vertices)])
    indices = seed_target_indices([0], np.arange(1, n_signals_tot))
    # Compute the PSI for alpha and beta band
    fmin = (8., 22.)
    fmax = (13., 30.)
    sfreq = epo.info['sfreq']  # the sampling frequency
    psi, freqs, times, n_epochs, n_tapers = phase_slope_index(comb_ts, mode='multitaper', indices=indices,
                                                              sfreq=sfreq, fmin=fmin, fmax=fmax, n_jobs=8)
    del stcs, sts, comb_ts
    # Generate a SourceEstimate with the PSI. This is simple since we used a single
    # seed (inspect the indices variable to see how the PSI scores are arranged in
    # the output)
    psi_stc = mne.MixedSourceEstimate(psi, vertices=vertices, tmin=0, tstep=1, subject=mri)
    del psi
    morph = mne.read_source_morph("{}{}_fs_mix-morph.h5".format(proc_dir,meg))
    psi_fs_stc = morph.apply(psi_stc)
    del morph
    psi_stcs.append(psi_fs_stc)

# calc GA and save
GA_mix_stc = psi_stcs[0].copy()
GA_stc_data = np.mean([stc.data for stc in psi_stcs], axis=0)
GA_mix_stc.data = GA_stc_data
GA_mix_stc.save("{}GA_fs_mix_Loud_IP_PSI_stc.h5".format(proc_dir))
GA_surf_stc = GA_mix_stc.surface()
GA_surf_stc.save("{}GA_fs_surf_Loud_IP_PSI_stc.h5".format(proc_dir))

# plot
fs_src = mne.read_source_spaces("{}fsaverage_oct6_mix-src.fif".format(proc_dir))
fs_surf = fs_src[:2]
labels = mne.read_labels_from_annot('fsaverage', parc='aparc', hemi='lh', surf_name='white', annot_fname=None, regexp=None, subjects_dir=mri_dir, sort=True)
label = [l for l in labels if l.name == 'inferiorparietal-lh'][0]

# v_max = np.max(np.abs(psi))
# brain = GA_mix_stc.plot(surface='white', hemi='split', time_label='Phase Slope Index (PSI)',
#                      subjects_dir=mri_dir, src=fs_src, clim=dict(kind='percent', pos_lims=(90, 95, 100)),
#                      time_viewer=True)
# # brain.show_view('medial')
# brain.add_label(label, color='green', alpha=0.7)

brain = GA_surf_stc.plot(surface='inflated', hemi='lh', time_label='GA Phase Slope Index (PSI)',
                         subjects_dir=mri_dir, src=fs_surf, clim=dict(kind='percent', pos_lims=(95, 97.5, 100)),
                         time_viewer=True, show_traces=False)
brain.add_label(label, color='green', alpha=0.5)
brain.add_annotation('aparc', borders=1, alpha=0.9)


#  add a comparison of highly influenced vs. less influenced subjects
N_behav = pd.read_csv('{}NEMO_behav.csv'.format(proc_dir))
N_behav['LD_rank'] = N_behav['Ton_Laut'].rank(ascending=False)
LD_rank_dict = {s: r for s,r in zip(N_behav.Subjects,N_behav.LD_rank)}
subs = list(sub_dict.keys())
highdiff = []
lowdiff = []
for i,s in enumerate(subs):
    if LD_rank_dict[s] < 10.5:
        highdiff.append(psi_stcs[i])
    else:
        lowdiff.append(psi_stcs[i])
GA_highdiff = GA_mix_stc.copy()
GA_hd_data = np.mean([stc.data for stc in highdiff], axis=0)
GA_highdiff.data = GA_hd_data
GA_hd_surf = GA_highdiff.surface()
GA_hd_surf.save("{}GA_HD_surf_Loud_IP_PSI_stc.h5".format(proc_dir))
GA_lowdiff = GA_mix_stc.copy()
GA_ld_data = np.mean([stc.data for stc in lowdiff], axis=0)
GA_lowdiff.data = GA_ld_data
GA_ld_surf = GA_lowdiff.surface()
GA_ld_surf.save("{}GA_LD_surf_Loud_IP_PSI_stc.h5".format(proc_dir))
brain = GA_hd_surf.plot(surface='inflated', hemi='lh', time_label='HD Phase Slope Index (PSI)',
                        subjects_dir=mri_dir, src=fs_surf, clim=dict(kind='percent', pos_lims=(95, 97.5, 100)),
                        time_viewer=True, show_traces=False)
# brain = GA_ld_surf.plot(surface='inflated', hemi='lh', time_label='Phase Slope Index (PSI)',
#                         subjects_dir=mri_dir, src=fs_surf, clim=dict(kind='value', pos_lims=(0.005, 0.01, 0.02)),
#                         time_viewer=True, show_traces=False)
brain.add_label(label, color='green', alpha=0.5)

# # each subject
# save_dir = "/home/cora/hdd/MEG/NEMO_analyses_new/LOUD_net/"
# for i,sub in enumerate(subs):
#     stc = psi_stcs[i].surface()
#     brain = stc.plot(surface='inflated', hemi='lh', time_label='{} - Phase Slope Index (PSI)'.format(sub),
#                      subjects_dir=mri_dir, src=fs_surf, clim=dict(kind='value', pos_lims=(0.005, 0.01, 0.02)),
#                      time_viewer=True, show_traces=False)
#     brain.add_label(label, color='green', alpha=0.5)
#     brain.add_annotation('aparc', borders=1, alpha=0.9)
#     brain.save_image_sequence(time_idx = [0,1], fname_pattern = '{}{}_supmar_PSI_%.png'.format(save_dir,sub), use_abs_idx=True, montage=['lat', 'med'])
#     # brain.save_imageset(sub, ['med', 'lat', 'ros', 'caud'], 'jpg')
#
# for i,sub in enumerate(subs):
#     stc = psi_stcs[i]
#     stc.save('{}{}_lh_supmar_ab_PSI-stc.h5'.format(save_dir,sub))
