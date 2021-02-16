## Calculate single trial alpha/beta power in selected labels to make stats with dPTE##
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()
import statsmodels.api as sm
import statsmodels.formula.api as smf

# set directories
proc_dir = "/home/cora/hdd/MEG/NEMO_analyses_new/proc/"  # on workstation D:/
mri_dir = "/home/cora/hdd/MEG/freesurfer/subjects/"
trans_dir = "/home/cora/hdd/MEG/NEMO_analyses_new/trans_files/"
save_dir = "/home/cora/hdd/MEG/NEMO_analyses_new/proc/dPTE/"
sub_dict = {"NEM_10":"GIZ04","NEM_11":"WOO07","NEM_12":"TGH11","NEM_14":"FIN23","NEM_15":"KIL13",
            "NEM_16":"KIO12","NEM_17":"DEN59","NEM_18":"SAG13","NEM_20":"PAG48","NEM_22":"EAM11",
            "NEM_23":"FOT12","NEM_24":"BII41","NEM_26":"ENR41","NEM_27":"HIU14","NEM_28":"WAL70",
            "NEM_29":"KIL72","NEM_31":"BLE94","NEM_34":"KER27","NEM_35":"MUN79","NEM_36":"BRA52_fa"}  ## order got corrected
excluded = {"NEM_30":"DIU11","NEM_32":"NAG83","NEM_33":"FAO18_fa","NEM_37":"EAM67","NEM_19":"ALC81","NEM_21":"WKI71_fa"}
# sub_dict = {"NEM_10":"GIZ04"}

# parameters from dPTE calc
freqs_a = [9.,10.,11.,12.,13.,14.]
freqs_b = [14.,15.,16.,17.,18.,19.,20.,21.]
cycles_a = 7
cycles_b = 8
# time period of interest (during tones)
tmin = 3.6
tmax = 6.6
# labels of interest
lois = ['lateraloccipital-rh','inferiorparietal-rh','inferiortemporal-rh','supramarginal-rh','transversetemporal-rh']
labels_limb = ['Left-Thalamus-Proper','Left-Caudate','Left-Putamen','Left-Pallidum','Left-Hippocampus','Left-Amygdala',
              'Right-Thalamus-Proper','Right-Caudate','Right-Putamen','Right-Pallidum','Right-Hippocampus','Right-Amygdala']

# prep the dPTE_power pandas dataframe to sort the values into later
# load the old dPTE alpha dataframe
df_NEM_dPTE_alpha = pd.read_csv("{}NEMO_dPTE_alpha.csv".format(save_dir))
# build new dataframe
columns_old = ["Subject","Emo","Connection","dPTE"]
columns = ["Subject","Emo"]
emos = ["neg","pos"]
conxs = ["LO_IP","LO_IT","LO_SM","LO_TT","IP_IT",
         "IP_SM","IP_TT","IT_SM","IT_TT","SM_TT"]
for c in conxs:
    columns.append(c + "_a_dPTE")
index = range(int(len(df_NEM_dPTE_alpha)/10))   # get index by trial count
df_NEM_a_dPTE_power = pd.DataFrame(columns=columns,index=index)
# now re-sort the dPTE values into the new df
for t in range(int(len(df_NEM_dPTE_alpha)/10)):
    trial = df_NEM_dPTE_alpha[(t*10+0):(t*10+10)]
    df_NEM_a_dPTE_power.iloc[t].at['Subject'] = trial.iloc[0,0]
    df_NEM_a_dPTE_power.iloc[t].at['Emo'] = trial.iloc[0,1]
    df_NEM_a_dPTE_power.iloc[t].at['LO_IP_a_dPTE'] = trial.iloc[0,3]
    df_NEM_a_dPTE_power.iloc[t].at['LO_IT_a_dPTE'] = trial.iloc[1,3]
    df_NEM_a_dPTE_power.iloc[t].at['LO_SM_a_dPTE'] = trial.iloc[2,3]
    df_NEM_a_dPTE_power.iloc[t].at['LO_TT_a_dPTE'] = trial.iloc[3,3]
    df_NEM_a_dPTE_power.iloc[t].at['IP_IT_a_dPTE'] = trial.iloc[4,3]
    df_NEM_a_dPTE_power.iloc[t].at['IP_SM_a_dPTE'] = trial.iloc[5,3]
    df_NEM_a_dPTE_power.iloc[t].at['IP_TT_a_dPTE'] = trial.iloc[6,3]
    df_NEM_a_dPTE_power.iloc[t].at['IT_SM_a_dPTE'] = trial.iloc[7,3]
    df_NEM_a_dPTE_power.iloc[t].at['IT_TT_a_dPTE'] = trial.iloc[8,3]
    df_NEM_a_dPTE_power.iloc[t].at['SM_TT_a_dPTE'] = trial.iloc[9,3]
# make sure float64 numbers are trated as such
df_NEM_a_dPTE_power = df_NEM_a_dPTE_power.infer_objects()

# now prep containers for collecting the power values
power = {"Subject":[],"Emo":[],"LO_a_pow":[],"LO_b_pow":[],"IP_a_pow":[],"IP_b_pow":[],
         "IT_a_pow":[],"IT_b_pow":[],"SM_a_pow":[],"SM_b_pow":[],"TT_a_pow":[],"TT_b_pow":[]}

# now calculate the alpha and beta power in our labels of interest for each trial
for meg, mri in sub_dict.items():
    # prep
    epo = mne.read_epochs("{}{}_long-epo.fif".format(proc_dir,meg))
    epo.info['bads'] = epo.info['bads'] + ['A51']
    # load forward solution
    fwd = mne.read_forward_solution("{}{}-fwd.fif".format(proc_dir,meg))
    # load all cortical (and subcortical) labels from our mixed SRC
    labels = mne.read_labels_from_annot(mri, parc='aparc', hemi='both', surf_name='white', annot_fname=None, regexp=None, subjects_dir=mri_dir, sort=False, verbose=None)
    # labels_sub = mne.get_volume_labels_from_src(fwd['src'], mri, mri_dir)     # this is currently producing an error, asking for nibabel (import nibabel as nib), which is not installed it appears..
    # labels_all = labels + labels_sub
    # then reduce them to our labels of interest
    labs = [l for l in labels if l.name in lois]
    labs_sorted = [labs[i] for i in [2,0,1,3,4]]    # sort them to order of lois
    # calculate CSD and DICS beamformer filters for all epos
    csd_a = mne.time_frequency.csd_morlet(epo, frequencies=freqs_a, n_jobs=8, n_cycles=cycles_a, decim=1)
    csd_b = mne.time_frequency.csd_morlet(epo, frequencies=freqs_b, n_jobs=8, n_cycles=cycles_b, decim=1)
    filters_a = mne.beamformer.make_dics(epo.info,fwd,csd_a.mean(),pick_ori='max-power',reduce_rank=False,depth=1.0,inversion='single')
    filters_a.save('{}{}-long-alpha-dics.h5'.format(save_dir,meg))
    filters_b = mne.beamformer.make_dics(epo.info,fwd,csd_b.mean(),pick_ori='max-power',reduce_rank=False,depth=1.0,inversion='single')
    filters_a.save('{}{}-long-beta-dics.h5'.format(save_dir,meg))
    del csd_a, csd_b
    # crop the epochs to the time interval of interest (this is done in_place)
    epo.crop(tmin=tmin, tmax=tmax)

    # now loop through negative and positive epochs seperately, calc CSD and apply filters per trial
    for i in range(len(epo["negative"])):
        epoch = epo["negative"][i]
        csd_a = mne.time_frequency.csd_morlet(epoch, frequencies=freqs_a, n_jobs=8, n_cycles=cycles_a, decim=1)
        csd_b = mne.time_frequency.csd_morlet(epoch, frequencies=freqs_b, n_jobs=8, n_cycles=cycles_b, decim=1)
        stc_a, frqs_a = mne.beamformer.apply_dics_csd(csd_a.mean(),filters_a)
        stc_b, frqs_b = mne.beamformer.apply_dics_csd(csd_b.mean(),filters_b)
        # extract label timecourses alpha & beta
        ltc_a = stc_a.extract_label_time_course(labs_sorted,fwd['src'])
        ltc_b = stc_b.extract_label_time_course(labs_sorted,fwd['src'])
        # fill values into the result dictionary (for dataframe)
        power['Subject'].append(meg)
        power['Emo'].append("neg")
        power['LO_a_pow'].append(ltc_a[0][0])
        power['IP_a_pow'].append(ltc_a[1][0])
        power['IT_a_pow'].append(ltc_a[2][0])
        power['SM_a_pow'].append(ltc_a[3][0])
        power['TT_a_pow'].append(ltc_a[4][0])
        power['LO_b_pow'].append(ltc_b[0][0])
        power['IP_b_pow'].append(ltc_b[1][0])
        power['IT_b_pow'].append(ltc_b[2][0])
        power['SM_b_pow'].append(ltc_b[3][0])
        power['TT_b_pow'].append(ltc_b[4][0])
    for i in range(len(epo["positive"])):
        epoch = epo["positive"][i]
        csd_a = mne.time_frequency.csd_morlet(epoch, frequencies=freqs_a, n_jobs=8, n_cycles=cycles_a, decim=1)
        csd_b = mne.time_frequency.csd_morlet(epoch, frequencies=freqs_b, n_jobs=8, n_cycles=cycles_b, decim=1)
        stc_a, frqs_a = mne.beamformer.apply_dics_csd(csd_a.mean(),filters_a)
        stc_b, frqs_b = mne.beamformer.apply_dics_csd(csd_b.mean(),filters_b)
        # extract label timecourses alpha & beta
        ltc_a = stc_a.extract_label_time_course(labs_sorted,fwd['src'])
        ltc_b = stc_b.extract_label_time_course(labs_sorted,fwd['src'])
        # fill values into the result dictionary (for dataframe)
        power['Subject'].append(meg)
        power['Emo'].append("pos")
        power['LO_a_pow'].append(ltc_a[0][0])
        power['IP_a_pow'].append(ltc_a[1][0])
        power['IT_a_pow'].append(ltc_a[2][0])
        power['SM_a_pow'].append(ltc_a[3][0])
        power['TT_a_pow'].append(ltc_a[4][0])
        power['LO_b_pow'].append(ltc_b[0][0])
        power['IP_b_pow'].append(ltc_b[1][0])
        power['IT_b_pow'].append(ltc_b[2][0])
        power['SM_b_pow'].append(ltc_b[3][0])
        power['TT_b_pow'].append(ltc_b[4][0])

# now fill the power values into a dataframe
df_NEM_pow = pd.DataFrame.from_dict(power)
# combine it with the dPTE dataframe and save
df_NEM_a_dPTE_power = df_NEM_a_dPTE_power.join(df_NEM_pow, on=None, how='left', lsuffix='', rsuffix='_p', sort=False)
df_NEM_a_dPTE_power = df_NEM_a_dPTE_power.drop(columns=['Subject_p','Emo_p'])
df_NEM_a_dPTE_power.to_csv("{}NEMO_a_dPTE_power.csv".format(save_dir), index=False)

# NOW DO STATS
# start with last cluster: TT-beta
res_TT_b_emo = smf.mixedlm('TT_b_pow ~ Emo', data=df_NEM_a_dPTE_power, groups=df_NEM_a_dPTE_power['Subject']).fit(reml=False)
print(res_TT_b_emo.summary())
print(res_TT_b_emo.params)
print(res_TT_b_emo.aic)
