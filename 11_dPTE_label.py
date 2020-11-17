## Calculate dPTE Directed Connectivity in Labels and Freqs of interest ##
import mne
import numpy as np
from mne.time_frequency import tfr_array_morlet
from dPTE import epo_dPTE
from joblib import Parallel, delayed

# set directories
proc_dir = "D:/NEMO_analyses_new/proc/"
mri_dir = "D:/freesurfer/subjects/"
trans_dir = "D:/NEMO_analyses_new/trans_files/"
save_dir = "D:/NEMO_analyses_new/proc/dPTE/"
sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13",
            "NEM_16":"KIO12","NEM_17":"DEN59","NEM_18":"SAG13","NEM_20":"PAG48","NEM_22":"EAM11",
            "NEM_23":"FOT12","NEM_24":"BII41","NEM_26":"ENR41","NEM_27":"HIU14","NEM_28":"WAL70",
            "NEM_29":"KIL72","NEM_31":"BLE94","NEM_34":"KER27","NEM_35":"MUN79","NEM_36":"BRA52_fa"}  ## order got corrected
excluded = {"NEM_30":"DIU11","NEM_32":"NAG83","NEM_33":"FAO18_fa","NEM_37":"EAM67","NEM_19":"ALC81","NEM_21":"WKI71_fa"}
sub_dict = {"NEM_10":"GIZ04"}

# Set parameters
freqs_a = [9.,10.,11.,12.,13.,14.]
freqs_b = [14.,15.,16.,17.,18.,19.,20.,21.]
cycles_a = 7
cycles_b = 8
# time period of interest (during tones)
tmin = 3.6
tmax = 6.6
# labels of interest
lois = ['lateraloccipital-rh','inferiortemporal-rh','supramarginal-rh','transversetemporal-rh']

for meg, mri in sub_dict.items():
    epo = mne.read_epochs("{}{}_long-epo.fif".format(proc_dir,meg))
    epo.info['bads'] = epo.info['bads'] + ['A51']
    epo.apply_baseline()
    data_cov = mne.compute_covariance(epo, tmin=tmin, tmax=tmax, method='empirical',n_jobs=8)
    noise_cov = mne.compute_covariance(epo, tmin=-0.2, tmax=0, method='empirical',n_jobs=8)
    fwd = mne.read_forward_solution("{}{}-fwd.fif".format(proc_dir,meg))
    filters = mne.beamformer.make_lcmv(epo.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov, pick_ori='max-power', rank='info', weight_norm='unit-noise-gain-invariant', reduce_rank=True, depth=None, inversion='matrix') # consider reduce_rank
    filters.save('{}{}_dPTE_filters-lcmv.h5'.format(save_dir,meg))
    # crop the epochs to the time interval of interest (this is done in_place)
    epo.crop(tmin=tmin, tmax=tmax)
    # then apply the filters to get source activity, separately for each condition
    stc_neg = mne.beamformer.apply_lcmv_epochs(epo["negative"], filters, max_ori_out='signed', return_generator=False)
    stc_pos = mne.beamformer.apply_lcmv_epochs(epo["positive"], filters, max_ori_out='signed', return_generator=False)
    # read labels
    labels = mne.read_labels_from_annot(mri, parc='aparc', hemi='both', surf_name='white', annot_fname=None, regexp=None, subjects_dir=mri_dir, sort=False)
    labs = [l for l in labels if l.name in lois]
    # read trans file for extracting label timecourse
    trans = "{dir}{mri}_{meg}-trans.fif".format(dir=trans_dir,mri=mri,meg=meg)
    # calculate label timecourses neg & pos
    neg_ltc = mne.extract_label_time_course(stc_neg, labs, fwd['src'], mode='pca_flip', trans=trans, mri_resolution=False)
    # as this automatically includes the 12 subcortical labels of the mixed source space, reduce it to the 1st four labels
    
