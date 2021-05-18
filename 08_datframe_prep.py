## load the trial power values of all subjects, and create a big Dataframe for the LMM analysis ##

import mne
import numpy as np
import pandas as pd
from io import StringIO
# import feather

# set directories to load files
behav_dir = "/home/cora/hdd/MEG/NEMO_analyses_new/behav/"   # on workstation D:/
proc_dir = "/home/cora/hdd/MEG/NEMO_analyses_new/proc/"

# subjects list
subjs = ["NEM_10","NEM_11","NEM_12","NEM_14","NEM_15",
         "NEM_16","NEM_17","NEM_18","NEM_20","NEM_22",
         "NEM_23","NEM_24","NEM_26","NEM_27","NEM_28",
         "NEM_29","NEM_31","NEM_34","NEM_35","NEM_36"]
# subjs = ["NEM_10","NEM_11"]
# trigger/event conditions dictionary
event_id = {'negative/pics': 70, 'positive/pics': 80,
            'negative/r1/part1': 110, 'negative/r1/part2': 111, 'negative/r1/part3': 112, 'negative/r1/part4': 113,
            'positive/r1/part1': 120, 'positive/r1/part2': 121, 'positive/r1/part3': 122, 'positive/r1/part4': 123,
            'negative/r2/part1': 130, 'negative/r2/part2': 131, 'negative/r2/part3': 132, 'negative/r2/part4': 133,
            'positive/r2/part1': 140, 'positive/r2/part2': 141, 'positive/r2/part3': 142, 'positive/r2/part4': 143,
            'negative/s1/part1': 150, 'negative/s1/part2': 151, 'negative/s1/part3': 152, 'negative/s1/part4': 153,
            'positive/s1/part1': 160, 'positive/s1/part2': 161, 'positive/s1/part3': 162, 'positive/s1/part4': 163,
            'negative/s2/part1': 170, 'negative/s2/part2': 171, 'negative/s2/part3': 172, 'negative/s2/part4': 173,
            'positive/s2/part1': 180, 'positive/s2/part2': 181, 'positive/s2/part3': 182, 'positive/s2/part4': 183}
trig_id = {v: k for k,v in event_id.items()}   # this reverses the dictionary and will be useful later

# overview of labels/regions of interest to choose from
aparc_limb_labels = ["bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "cuneus", "entorhinal", "fusiform", "inferiorparietal", "inferiortemporal", "isthmuscingulate",
                     "lateraloccipital", "lateralorbitofrontal", "lingual", "medialorbitofrontal", "middletemporal", "parahippocampal", "paracentral", "parsopercularis", "parsorbitalis",
                     "parstriangularis", "pericalcarine", "postcentral", "posteriorcingulate", "precentral", "precuneus", "rostralanteriorcingulate", "rostralmiddlefrontal",
                     "superiorfrontal", "superiorparietal", "superiortemporal", "supramarginal", "frontalpole", "temporalpole", "transversetemporal", "insula",
                     "Thalamus-Proper", "Caudate", "Putamen", "Pallidum", "Hippocampus", "Amygdala"]
rois_emo = ["Amygdala","insula","parsorbitalis","medialorbitofrontal","lateralorbitofrontal"]
rois_aud = ["Thalamus-Proper","transversetemporal","superiortemporal","bankssts","supramarginal"]
rois_vis = ["lateraloccipital","cuneus","pericalcarine","lingual","fusiform","inferiortemporal"]
rois_att = ["superiorparietal","inferiorparietal","rostralanteriorcingulate","caudalanteriorcingulate","posteriorcingulate"]
rois_mem = ["Hippocampus","parahippocampal","entorhinal"]
rois_mot = ["precentral"]

# set columns construction variables: Cond, ROI, Freq
conds = ["Pic","Ton_Part1","Ton_Part2","Ton_Part3","Ton_Part4"]
hems = ["-lh","-rh"]
rois_pic = aparc_limb_labels
rois_ton = aparc_limb_labels
freqs = ["theta","alpha","beta_low","beta_high","gamma","gamma_high"]
start_cols = ["Subject","Psych_ges","Angst_ges","ER_ges","Trial_O","Ton","Trial_N","Cond","PicVal","PicArs"]
#mid_cols = ["Ton"]
end_cols = ["TonLaut","TonAng","MoodVal","MoodArs"]

# get dicts here with psych variables for fixed vars
Psych_ges = [70,58,19,19,25,23,3,43,52,29,37,25,56,17,16,49,26,28,46,45]
Angst_ges = [15,9,9,7,7,7,1,5,8,6,8,2,14,5,0,6,7,4,14,8]
ER_ges = [60,56,65,77,87,77,93,55,65,86,67,81,83,75,70,71,62,78,60,69]
ton_dict = {110:"r1",120:"r1",130:"r2",140:"r2",150:"s1",160:"s1",170:"s2",180:"s2"}
con_dict = {110:"negative/r1",120:"positive/r1",130:"negative/r2",140:"positive/r2",150:"negative/s1",160:"positive/s1",170:"negative/s2",180:"positive/s2"}

# start with building superDataframe
columns = start_cols
cond = conds[0]
for roi in rois_pic:
    for hem in hems:
        for freq in freqs:
            columns.append("{c}_{r}{h}_{f}".format(c=cond,r=roi,h=hem,f=freq))
#columns += mid_cols
for cond in conds[1:]:
    for roi in rois_ton:
        for hem in hems:
            for freq in freqs:
                columns.append("{c}_{r}{h}_{f}".format(c=cond,r=roi,h=hem,f=freq))
columns += end_cols

df_NEM = pd.DataFrame(columns=columns)

# now loop through the subjects and fill in the data
for sub_ix,sub in enumerate(subjs):
    # keep track
    print("Processing subject:   {}".format(sub))
    # load the 3 required .txt files into dataframes: rat, res, trial_power
    rat = pd.read_table("{}rat_{}.txt".format(behav_dir,sub))
    res = pd.read_table("{}res_{}.txt".format(behav_dir,sub))
    res = res[4:]   #ignore baseline ratings on tones and look only at experimental ratings
    trip = pd.read_table("{}{}_trial_roi_power.txt".format(proc_dir,sub))
    # keep track
    print("Data loaded.")
    # create an empty subject dataframe
    df_sub = pd.DataFrame(columns=columns,index=range(res.shape[0]))
    # fill in behavioral data row by row
    for i in range(res.shape[0]):
        df_sub.iloc[i].at['Subject'] = sub
        df_sub.iloc[i].at['Psych_ges'] = Psych_ges[sub_ix]
        df_sub.iloc[i].at['Angst_ges'] = Angst_ges[sub_ix]
        df_sub.iloc[i].at['ER_ges'] = ER_ges[sub_ix]
        df_sub.iloc[i].at['Trial_O'] = i
        df_sub.iloc[i].at['Trial_N'] = res.iloc[i].at['Trial']
        if res.iloc[i].at['Cat'] == 'N':
            df_sub.iloc[i].at['Cond'] = 'negative'
        else:
            df_sub.iloc[i].at['Cond'] = 'positive'
        df_sub.iloc[i].at['PicVal'] = rat.at[rat[rat['Trial'] == res.iloc[i].at['Trial']].index[0], 'PicVal']
        df_sub.iloc[i].at['PicArs'] = rat.at[rat[rat['Trial'] == res.iloc[i].at['Trial']].index[0], 'PicArs']
        df_sub.iloc[i].at['Ton'] = ton_dict[res.iloc[i].at['Ton']]
        df_sub.iloc[i].at['TonLaut'] = res.iloc[i].at['Laut']
        df_sub.iloc[i].at['TonAng'] = res.iloc[i].at['Angenehm']
        df_sub.iloc[i].at['MoodVal'] = res.iloc[i].at['Valence']
        df_sub.iloc[i].at['MoodArs'] = res.iloc[i].at['Arousal']
    # keep track
    print("Done with behavioral data. Now starting trial power...")
    # fill trial power values row by row
    for ix in range(df_sub.shape[0]):
        # keep track
        print("Filling in values for trial:   {}".format(ix))
        big_trial = con_dict[res.iloc[ix].at['Ton']]
        # keep track
        print("Condition:   {}".format(big_trial))
        w = 5
        while w > 0:
            trial = trip.head(6)
            if trial.empty:
                break
            if (df_sub.iloc[ix].at['Cond'] in big_trial) and (df_sub.iloc[ix].at['Cond'] in trial.iloc[0].at['Event_ID']):
                if w > 4:
                    if 'pics' in trial.iloc[0].at['Event_ID']:
                        cond = 'Pic'
                        # keep track
                        print("Filling in -   {}".format(cond))
                        for roi in rois_pic:
                            for hem in hems:
                                for freq in freqs:
                                    df_sub.iloc[ix].at["{c}_{r}{h}_{f}".format(c=cond,r=roi,h=hem,f=freq)] = trial.at[trial[trial['Freq']==freq].index[0], '{r}{h}'.format(r=roi,h=hem)]
                        trip = trip[6:]  # now take the used trial out
                        w = 4
                        continue
                if 'part' in trial.iloc[0].at['Event_ID']:
                    if (df_sub.iloc[ix].at['Ton'] in big_trial) and (df_sub.iloc[ix].at['Ton'] in trial.iloc[0].at['Event_ID']):
                        if w > 3:
                            if 'part1' in trial.iloc[0].at['Event_ID']:
                                cond = "Ton_Part1"
                                # keep track
                                print("Filling in -   {}".format(cond))
                                for roi in rois_ton:
                                    for hem in hems:
                                        for freq in freqs:
                                            df_sub.iloc[ix].at["{c}_{r}{h}_{f}".format(c=cond,r=roi,h=hem,f=freq)] = trial.at[trial[trial['Freq']==freq].index[0], '{r}{h}'.format(r=roi,h=hem)]
                                trip = trip[6:]  # now take the used trial out
                                w = 3
                                continue
                        if w > 2:
                            if 'part2' in trial.iloc[0].at['Event_ID']:
                                cond = "Ton_Part2"
                                # keep track
                                print("Filling in -   {}".format(cond))
                                for roi in rois_ton:
                                    for hem in hems:
                                        for freq in freqs:
                                            df_sub.iloc[ix].at["{c}_{r}{h}_{f}".format(c=cond,r=roi,h=hem,f=freq)] = trial.at[trial[trial['Freq']==freq].index[0], '{r}{h}'.format(r=roi,h=hem)]
                                trip = trip[6:]  # now take the used trial out
                                w = 2
                                continue
                        if w > 1:
                            if 'part3' in trial.iloc[0].at['Event_ID']:
                                cond = "Ton_Part3"
                                # keep track
                                print("Filling in -   {}".format(cond))
                                for roi in rois_ton:
                                    for hem in hems:
                                        for freq in freqs:
                                            df_sub.iloc[ix].at["{c}_{r}{h}_{f}".format(c=cond,r=roi,h=hem,f=freq)] = trial.at[trial[trial['Freq']==freq].index[0], '{r}{h}'.format(r=roi,h=hem)]
                                trip = trip[6:]  # now take the used trial out
                                w = 1
                                continue
                        if 'part4' in trial.iloc[0].at['Event_ID']:
                            cond = "Ton_Part4"
                            # keep track
                            print("Filling in -   {}".format(cond))
                            for roi in rois_ton:
                                for hem in hems:
                                    for freq in freqs:
                                        df_sub.iloc[ix].at["{c}_{r}{h}_{f}".format(c=cond,r=roi,h=hem,f=freq)] = trial.at[trial[trial['Freq']==freq].index[0], '{r}{h}'.format(r=roi,h=hem)]
                            trip = trip[6:]  # now take the used trial out
            w = 0    # outdent this one further ??
            print("Proceeding to next trial...")
    df_NEM = pd.concat([df_NEM,df_sub])

# when all subject data are collected, save the dataframe (feather format is fast and readable in R)
df_NEM.index = list(range(df_NEM.shape[0]))  # fix the index for saving
# make sure the numbers are in the correct dtype
df_NEM = df_NEM.infer_objects()
# convert the power values for stats: multiply by e+30, to make fT^2 out of T^2 & then take the log10 to make linear; power columns are [10:-4]
df_NEM.iloc[:, 10:-4] = df_NEM.iloc[:, 10:-4].mul(1e+30)
df_NEM.iloc[:, 10:-4] = np.log10(df_NEM.iloc[:, 10:-4])
# save
# df_NEM.to_feather("{}NEMO_complete.feather".format(proc_dir))
df_NEM.to_csv("{}NEMO_complete.csv".format(proc_dir), index=False)
