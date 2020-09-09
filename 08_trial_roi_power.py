## this is to calculate the power in different regions of interest (ROIs) in selected frequencies for single trials, and to store them in a dataframe for each subject ##

import mne
import numpy as np
import pandas as pd

proc_dir = "D:/NEMO_analyses_new/proc/"
mri_dir = "D:/freesurfer/subjects/"
sub_dict = {"NEM_24":"BII41","NEM_26":"ENR41","NEM_27":"HIU14","NEM_28":"WAL70",
            "NEM_29":"KIL72","NEM_31":"BLE94","NEM_34":"KER27","NEM_35":"MUN79","NEM_36":"BRA52_fa"}  ## order got corrected
excluded = {"NEM_30":"DIU11","NEM_32":"NAG83","NEM_33":"FAO18_fa","NEM_37":"EAM67","NEM_19":"ALC81","NEM_21":"WKI71_fa"}
# # sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13",
#               "NEM_16":"KIO12","NEM_17":"DEN59","NEM_18":"SAG13","NEM_20":"PAG48","NEM_22":"EAM11",
#               "NEM_23":"FOT12",}

## PREP INFO
# the frequency bands used in dictionary form
freqs = {"theta":list(np.arange(3,8)),"alpha":list(np.arange(8,14)),"beta_low":list(np.arange(14,22)),
         "beta_high":list(np.arange(22,31)),"gamma":list(np.arange(31,47))}
freq_tup = tuple(freqs.keys())
freqs_g = {"gamma_high":list(np.arange(65,96,2))}
# the frequencies passed as lists (for CSD calculation) & freq_band bounds for averaging CSDs
freqs_n = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
cycs_n = [5, 5, 5, 5, 5, 7, 7,  7,  7,  7,  7,  9,  9,  9,  9,  9,  9,  9,  9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13]
freqs_g = [65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95]
cycs_g = [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
fmins = [3, 8, 14, 22, 31]
fmaxs = [7, 13, 21, 30, 46]   # remember to use (65,95) as (fmin,fmax) for the gamma_high calculation
# the conditions
conditions = {'rest':'rest', 'tonbas':['tonbas','tonrat'], 'pic_n':'negative/pics', 'pic_p':'positive/pics',
              'ton_n':['negative/r1','negative/r2','negative/s1','negative/s2'], 'ton_p':['positive/r1','positive/r2','positive/s1','positive/s2']}
conds = list(conditions.keys())

event_id = {'rest': 220,
            'tonbas/r1/part1': 191,'tonbas/r2/part1': 192, 'tonbas/s1/part1': 193, 'tonbas/s2/part1': 194,
            'tonbas/r1/part2': 291,'tonbas/r2/part2': 292, 'tonbas/s1/part2': 293, 'tonbas/s2/part2': 294,
            'tonbas/r1/part3': 391,'tonbas/r2/part3': 392, 'tonbas/s1/part3': 393, 'tonbas/s2/part3': 394,
            'tonbas/r1/part4': 491,'tonbas/r2/part4': 492, 'tonbas/s1/part4': 493, 'tonbas/s2/part4': 494,
            'tonbas/r1/part5': 591,'tonbas/r2/part5': 592, 'tonbas/s1/part5': 593, 'tonbas/s2/part5': 594,
            'tonrat/r1/part1': 691,'tonrat/r2/part1': 692, 'tonrat/s1/part1': 693, 'tonrat/s2/part1': 694,
            'tonrat/r1/part2': 791,'tonrat/r2/part2': 792, 'tonrat/s1/part2': 793, 'tonrat/s2/part2': 794,
            'tonrat/r1/part3': 891,'tonrat/r2/part3': 892, 'tonrat/s1/part3': 893, 'tonrat/s2/part3': 894,
            'tonrat/r1/part4': 991,'tonrat/r2/part4': 992, 'tonrat/s1/part4': 993, 'tonrat/s2/part4': 994,
            'negative/pics': 70, 'positive/pics': 80,
            'negative/r1/part1': 110, 'negative/r1/part2': 111, 'negative/r1/part3': 112, 'negative/r1/part4': 113,
            'positive/r1/part1': 120, 'positive/r1/part2': 121, 'positive/r1/part3': 122, 'positive/r1/part4': 123,
            'negative/r2/part1': 130, 'negative/r2/part2': 131, 'negative/r2/part3': 132, 'negative/r2/part4': 133,
            'positive/r2/part1': 140, 'positive/r2/part2': 141, 'positive/r2/part3': 142, 'positive/r2/part4': 143,
            'negative/s1/part1': 150, 'negative/s1/part2': 151, 'negative/s1/part3': 152, 'negative/s1/part4': 153,
            'positive/s1/part1': 160, 'positive/s1/part2': 161, 'positive/s1/part3': 162, 'positive/s1/part4': 163,
            'negative/s2/part1': 170, 'negative/s2/part2': 171, 'negative/s2/part3': 172, 'negative/s2/part4': 173,
            'positive/s2/part1': 180, 'positive/s2/part2': 181, 'positive/s2/part3': 182, 'positive/s2/part4': 183}
trig_id = {v: k for k,v in event_id.items()}   # this reverses the dictionary and will be useful later

## POWER ROI CALCULATIONS
for meg,mri in sub_dict.items():
    roi_power = []
    # load the epochs, and restrict them to the experimental trials
    epo_all = mne.read_epochs("{}{}-epo.fif".format(proc_dir,meg))
    epos = epo_all['negative','positive']
    # load the forward model and labels
    fwd = mne.read_forward_solution("{}{}-fwd.fif".format(proc_dir,meg))
    labels = mne.read_labels_from_annot(mri, parc='aparc', hemi='both', surf_name='white', annot_fname=None, regexp=None, subjects_dir=mri_dir, sort=False, verbose=None)
    labels_sub = mne.get_volume_labels_from_src(fwd['src'], mri, mri_dir)
    labels_all = labels + labels_sub
    # load filters for DICS beamformer
    filters = mne.beamformer.read_beamformer('{}{}-dics.h5'.format(proc_dir,meg))
    filters_g = mne.beamformer.read_beamformer('{}{}-gamma-dics.h5'.format(proc_dir,meg))
    # create a file to save the values
    filename = "{}{}_trial_roi_power.txt".format(proc_dir,meg)
    with open(filename, "w") as file:
        file.write("Tri_Ord\tEvent_ID\tTrig\tFreq")
        for l in labels_all:
            file.write("\t{}".format(l.name))
        file.write("\n")
        # now loop through the single epochs and calculate the power values
        for i in range(len(epos)):
            epo = epos[i]
            # calculate csd and csd_gamma
            csd_n = mne.time_frequency.csd_morlet(epo, frequencies=freqs_n, n_jobs=8, n_cycles=cycs_n, decim=1)
            csd_g = mne.time_frequency.csd_morlet(epo, frequencies=freqs_g, n_jobs=8, n_cycles=cycs_g, decim=1)
            # apply filters
            stc, frqs = mne.beamformer.apply_dics_csd(csd_n.mean(fmins,fmaxs),filters)
            stc_g, frqs_g = mne.beamformer.apply_dics_csd(csd_g.mean(65,95),filters_g)
            # extract label timecourses (i.e. power values) for each lower frequency...
            for f_ix,f in enumerate(freq_tup):
                stc_f = stc.copy()
                stc_f.crop(tmin=f_ix,tmax=f_ix)
                ltc_f = stc_f.extract_label_time_course(labels,fwd['src'])
                file.write("{i}\t{e}\t{t}\t{f}".format(i=i,e=[trig_id[v] for v in epo.event_id.values()][0],t=epo.events[0][-1],f=f))
                for lp in ltc_f:
                    file.write("\t{}".format(lp[0]))
                file.write("\n")
            ltc_g = stc_g.extract_label_time_course(labels,fwd['src'])
            file.write("{i}\t{e}\t{t}\t{f}".format(i=i,e=[trig_id[v] for v in epo.event_id.values()][0],t=epo.events[0][-1],f='gamma_high'))
            for lp in ltc_g:
                file.write("\t{}".format(lp[0]))
            file.write("\n")
