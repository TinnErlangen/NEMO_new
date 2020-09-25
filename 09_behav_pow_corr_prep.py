## Preparation for fsaverage morphed difference STCs for the behav correlation cluster analysis ##
## prep mixed STCs, with .surface() they can later be reudced to surface ones

import mne
import numpy as np
mne.viz.set_3d_backend('pyvista')

## remember: BRA52, ((FAO18, WKI71 - excl.)) have fsaverage MRIs (originals were defective)


proc_dir = "D:/NEMO_analyses_new/proc/"
mri_dir = "D:/freesurfer/subjects/"
sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13",
           "NEM_16":"KIO12","NEM_17":"DEN59","NEM_18":"SAG13","NEM_20":"PAG48","NEM_22":"EAM11",
           "NEM_23":"FOT12","NEM_24":"BII41","NEM_26":"ENR41","NEM_27":"HIU14","NEM_28":"WAL70",
           "NEM_29":"KIL72","NEM_31":"BLE94","NEM_34":"KER27","NEM_35":"MUN79","NEM_36":"BRA52_fa"}  ## order got corrected
excluded = {"NEM_30":"DIU11","NEM_32":"NAG83","NEM_33":"FAO18_fa","NEM_37":"EAM67","NEM_19":"ALC81","NEM_21":"WKI71_fa"}
# sub_dict = {"NEM_10":"GIZ04"}

# the frequency bands used in dictionary form & freq_band bounds for averaging CSDs
freqs = {"theta":list(np.arange(3,8)),"alpha":list(np.arange(8,14)),"beta_low":list(np.arange(14,22)),
         "beta_high":list(np.arange(22,31)),"gamma":list(np.arange(31,47))}
freq_tup = tuple(freqs.keys())
freqs_g = {"gamma_high":list(np.arange(65,96,2))}
fmins = [3, 8, 14, 22, 31]
fmaxs = [7, 13, 21, 30, 46]
# the conditions
conditions = {'rest':'rest', 'tonbas':['tonbas','tonrat'], 'pic_n':'negative/pics', 'pic_p':'positive/pics',
              'ton_n':['negative/r1','negative/r2','negative/s1','negative/s2'], 'ton_p':['positive/r1','positive/r2','positive/s1','positive/s2']}
conds = list(conditions.keys())
## specifiy the conditions to contrast
cond_a = 'ton_n'
cond_b = 'ton_p'

# load fsaverage source Space
fs_src = mne.read_source_spaces("{}fsaverage_oct6_mix-src.fif".format(proc_dir))

## prep subject STCs, make Diff_STC and morph to 'fsaverage' -- save
for meg,mri in sub_dict.items():
    fwd = mne.read_forward_solution("{}{}-fwd.fif".format(proc_dir,meg))
    # load filters for DICS beamformer
    filters = mne.beamformer.read_beamformer('{}{}-dics.h5'.format(proc_dir,meg))
    # load CSDs for conditions to compare, apply filters
    csd_a = mne.time_frequency.read_csd("{}{}_{}-csd.h5".format(proc_dir,meg,cond_a))
    csd_b = mne.time_frequency.read_csd("{}{}_{}-csd.h5".format(proc_dir,meg,cond_b))
    stc_a, freqs_a = mne.beamformer.apply_dics_csd(csd_a.mean(fmins,fmaxs),filters)
    stc_b, freqs_b = mne.beamformer.apply_dics_csd(csd_b.mean(fmins,fmaxs),filters)
    # calculate the difference between conditions
    stc_diff = (stc_a - stc_b) / stc_b
    # morph diff to fsaverage
    morph = mne.read_source_morph("{}{}_fs_mix-morph.h5".format(proc_dir,meg))
    stc_fs_diff = morph.apply(stc_diff)
    stc_fs_diff.save("{}{}_STC_fs_{}-{}".format(proc_dir,meg,cond_a,cond_b))
    # then for gamma_high
    # load filters for DICS beamformer
    filters_g = mne.beamformer.read_beamformer('{}{}-gamma-dics.h5'.format(proc_dir,meg))
    # load CSDs for conditions to compare, apply filters
    csd_g_a = mne.time_frequency.read_csd("{}{}_{}-gamma-csd.h5".format(proc_dir,meg,cond_a))
    csd_g_b = mne.time_frequency.read_csd("{}{}_{}-gamma-csd.h5".format(proc_dir,meg,cond_b))
    stc_g_a, freqs_g_a = mne.beamformer.apply_dics_csd(csd_g_a.mean(65,95),filters_g)
    stc_g_b, freqs_g_b = mne.beamformer.apply_dics_csd(csd_g_b.mean(65,95),filters_g)
    # calculate the difference between conditions
    stc_diff_g = (stc_g_a - stc_g_b) / stc_g_b
    # morph diff to fsaverage
    stc_fs_diff_g = morph.apply(stc_diff_g)
    stc_fs_diff_g.save("{}{}_STC_fs_{}-{}_gamma".format(proc_dir,meg,cond_a,cond_b))
