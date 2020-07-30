# This script creates a Cycler object to loop over your raw data to mark bad segments
# Run this file from command line with '-i' for interactive mode
# Then use the cyc.go() command each time to pop the next file in the list for inspection and annotation - then use cyc.save() when done
# ...then cyc.go() again for the next file... until the list is empty

import mne
import matplotlib.pyplot as plt
import numpy as np

plt.ion() #this keeps plots interactive

# define file locations
proc_dir = "D:/NEMO_analyses_new/preproc/"
# pass subject and run lists
subjs_all = ["NEM_10","NEM_11","NEM_12","NEM_14","NEM_15","NEM_16",
        "NEM_17","NEM_18","NEM_19","NEM_20","NEM_21","NEM_22",
        "NEM_23","NEM_24","NEM_26","NEM_27","NEM_28",
        "NEM_29","NEM_30","NEM_31","NEM_32","NEM_33","NEM_34",
        "NEM_35","NEM_36","NEM_37"]
excluded = ["NEM_19","NEM_21","NEM_30","NEM_32","NEM_33","NEM_37"]
subjs = ["NEM_10","NEM_11","NEM_12","NEM_14","NEM_15",
         "NEM_16","NEM_17","NEM_18","NEM_20","NEM_22",
         "NEM_23","NEM_24","NEM_26","NEM_27","NEM_28",
         "NEM_29","NEM_31","NEM_34","NEM_35","NEM_36"]
#subjs = ["NEM_10","NEM_11"]
runs = ["1","2","3","4"]
# create new lists, if you want to try things on single subjects or runs

#dictionary with conditions/triggers for plotting
event_id = {'rest': 220, 'ton_r1': 191,'ton_r2': 192, 'ton_s1': 193, 'ton_s2': 194,
            'negative/pic1': 10, 'positive/pic1': 20, 'negative/pic2': 30,
            'positive/pic2': 40, 'negative/pic3': 50, 'positive/pic3': 60,
            'negative/r1': 110,'positive/r1': 120, 'negative/r2': 130, 'positive/r2': 140,
            'negative/s1': 150, 'positive/s1': 160,'negative/s2': 170, 'positive/s2': 180}

# collecting the files for annotation into a list
filelist = []
for sub in subjs:
    for run in runs:
        filelist.append('{dir}nc_{sub}_{run}-raw.fif'.format(dir=proc_dir,sub=sub,run=run))

#definition of cycler object to go through the file list for annotation
class Cycler():

    def __init__(self,filelist):
        self.filelist = filelist    # when initializing the object, the filelist is collected

    def go(self):
        self.fn = self.filelist.pop(0)    # this pops the first raw file from the list
        self.raw = mne.io.Raw(self.fn)
        self.events = np.load(self.fn[:-8]+"_events.npy")   # and loads the events
        self.raw.plot(duration=15.0,n_channels=90,scalings=dict(mag=0.5e-12),events=self.events,event_id=event_id)    #  these parameters work well for inspection, but change to your liking (works also interactively during plotting)
        self.raw.plot_psd(fmax=95)    # we also plot the PSD, which is helpful to spot bad channels

    def plot(self,n_channels=90):
        self.raw.plot(duration=15.0,n_channels=90,scalings=dict(mag=0.5e-12),events=self.events,event_id=event_id)

    def show_file(self):
        print("Current Raw File: " + self.fn)    # use this to find out which subject/run we're looking at currently

    def save(self):
        self.raw.save(self.fn[:-8]+'_m-raw.fif', overwrite=True)   # important: save the annotated file in the end !

cyc = Cycler(filelist)


# Tipps: click on bad channels to mark them (they're easily spotted from the PSD plot); press 'a' to switch in annotation mode and drag the mouse over 'BAD' segments to mark with that label
# important: close the plot to save the markings! - then do cyc.save() to save the file
