## Calculate Time-Frequency Beamformer in Label ##
import mne
import numpy as np
from mne.beamformer import tf_dics
from mne.viz import plot_source_spectrogram
import time
import datetime

# set directories
proc_dir = "D:/NEMO_analyses_new/proc/"
mri_dir = "D:/freesurfer/subjects/"
plot_dir = "D:/NEMO_analyses_new/plots/TF_dics/"
save_dir = "D:/NEMO_analyses_new/proc/TF_dics/"
sub_dict = {"NEM_23":"FOT12","NEM_24":"BII41","NEM_26":"ENR41","NEM_27":"HIU14","NEM_28":"WAL70",
            "NEM_29":"KIL72","NEM_31":"BLE94","NEM_34":"KER27","NEM_35":"MUN79","NEM_36":"BRA52_fa"}  ## order got corrected
excluded = {"NEM_30":"DIU11","NEM_32":"NAG83","NEM_33":"FAO18_fa","NEM_37":"EAM67","NEM_19":"ALC81","NEM_21":"WKI71_fa"}
# sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13","NEM_16":"KIO12","NEM_17":"DEN59","NEM_18":"SAG13","NEM_20":"PAG48","NEM_22":"EAM11",}


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
loi = 'supramarginal-rh'


for meg,mri in sub_dict.items():
    start = time.perf_counter()
    epo = mne.read_epochs("{}{}_long-epo.fif".format(proc_dir,meg))
    epo.info['bads'] = epo.info['bads'] + ['A51']
    fwd = mne.read_forward_solution("{}{}-fwd.fif".format(proc_dir,meg))
    labels = mne.read_labels_from_annot(mri, parc='aparc', hemi='both', surf_name='white', annot_fname=None, regexp=None, subjects_dir=mri_dir, sort=False, verbose=None)
    label = [l for l in labels if l.name == loi][0]
    print("Calculating TF_DICS negative for {}".format(meg))
    stcs_neg = tf_dics(epo['negative'], fwd, noise_csds = None, tmin=tmin, tmax=tmax, tstep=tstep, win_lengths=win_lengths, subtract_evoked=False, mode='cwt_morlet', frequencies=frequencies, cwt_n_cycles=cwt_n_cycles,
                       reg = 0.05, label=label, pick_ori='max-power', inversion='single', depth=1.0, n_jobs=n_jobs)
    for fb in range(len(freq_bins)):
        stcs_neg[fb].save("{}{}_TF_dics_neg_{}-{}_{}-stc.h5".format(save_dir,meg,freq_bins[fb][0],freq_bins[fb][-1],loi))
    print("Calculating TF_DICS positive for {}".format(meg))
    stcs_pos = tf_dics(epo['positive'], fwd, noise_csds = None, tmin=tmin, tmax=tmax, tstep=tstep, win_lengths=win_lengths, subtract_evoked=False, mode='cwt_morlet', frequencies=frequencies, cwt_n_cycles=cwt_n_cycles,
                       reg = 0.05, label=label, pick_ori='max-power', inversion='single', depth=1.0, n_jobs=n_jobs)
    for fb in range(len(freq_bins)):
        stcs_pos[fb].save("{}{}_TF_dics_pos_{}-{}_{}-stc.h5".format(save_dir,meg,freq_bins[fb][0],freq_bins[fb][-1],loi))
    # now calculate the difference between conditions
    stcs_cont = [(a[0]-a[1]) for a in zip(stcs_neg,stcs_pos)]
    for fb in range(len(freq_bins)):
        stcs_cont[fb].save("{}{}_TF_dics_cont_N-P_{}-{}_{}-stc.h5".format(save_dir,meg,freq_bins[fb][0],freq_bins[fb][-1],loi))
    end = time.perf_counter()
    dur = np.round(end-start)
    print("Finished TF_DICS calc for {}".format(meg))
    print("This took {}".format(str(datetime.timedelta(seconds=dur))))
