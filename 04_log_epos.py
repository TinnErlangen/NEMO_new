#step 3 - visual inspection and marking of bad epochs

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
# subjs = ["NEM_10"]
runs = ["1","2","3","4"]
# runs = ["3","4"]
# create new lists, if you want to try things on single subjects or runs

#collecting the files for annotation
filelist = []
for sub in subjs:
    for run in runs:
        filelist.append('{dir}{sub}_{run}-epo.fif'.format(dir=proc_dir,sub=sub,run=run))

#definition of cycler object to go through the file list for annotation
class Cycler():

    def __init__(self,filelist):
        self.filelist = filelist

    def go(self):
        self.fn = self.filelist.pop(0)
        self.epo = mne.read_epochs(self.fn)
        self.epo.plot(n_epochs=10,n_channels=90,scalings=dict(mag=3e-12)) #these parameters work well for inspection of my 2sec epochs
        self.fig = self.epo.plot_psd(fmax=95,bandwidth=1,average=False)

    def plot(self,n_epochs=10,n_channels=90):
        self.epo.plot(n_epochs=n_epochs,n_channels=n_channels,scalings=dict(mag=3e-12))

    def show_file(self):
        print("Current Epoch File: " + self.fn)

    def save(self):
        self.epo.save(self.fn[:-8]+'-epo.fif',overwrite=True)
        self.fig.savefig(self.fn[:-8]+"-psd")
        self.fig.clear()
        with open(self.fn[:-8]+'_epolog.txt', "w") as file:
            file.write(str(self.epo.info["bads"]))
            file.write("\n\n")
            file.write(str(self.epo.drop_log))
            file.write("\n\n")
            drops_b = self.epo.drop_log.count(('BAD_',))
            inds_b = []
            start = 0
            for d in range(drops_b):
                i = self.epo.drop_log.index(('BAD_',), start)
                inds_b.append(i)
                start = i + 1
            file.write("Bad epochs: "+str(drops_b)+" Original index numbers: "+str(inds_b))
            drops_u = self.epo.drop_log.count(('USER',))
            inds_u = []
            start = 0
            for d in range(drops_u):
                i = self.epo.drop_log.index(('USER',), start)
                inds_u.append(i)
                start = i + 1
            file.write("\nUser dropped epochs: "+str(drops_u)+" Original index numbers: "+str(inds_u))
            if self.fn[-9] in ['3','4']:
                inds = inds_b + inds_u
                tris = []
                for ind in inds:
                    tri = ind / 5
                    if ind % 5 != 0:
                        tri = tri + 1
                    tris.append(int(tri))
                if self.fn[-9] == '4':
                    tris = [t+32 for t in tris]
                file.write("\nAffected trial numbers: "+str(tris))

cyc = Cycler(filelist)

#run this file from command line with '-i' for interactive mode
#then use the cyc.go() command each time to pop the next file in the list for annotation and cyc.save() when done
#then cyc.go() again... until list is empty
