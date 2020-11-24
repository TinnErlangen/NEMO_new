## Load the GA results of dPTE Directed Connectivity in Labels and make a plot ##
import mne
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

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
lois = ['lateraloccipital-rh','inferiortemporal-rh','inferiorparietal-rh','supramarginal-rh','transversetemporal-rh']
labels_limb = ['Left-Thalamus-Proper','Left-Caudate','Left-Putamen','Left-Pallidum','Left-Hippocampus','Left-Amygdala',
              'Right-Thalamus-Proper','Right-Caudate','Right-Putamen','Right-Pallidum','Right-Hippocampus','Right-Amygdala']

# load the GA results
GA_neg_a = np.load("{}GA_dPTE_alpha_neg.npy".format(save_dir))
GA_neg_b = np.load("{}GA_dPTE_beta_neg.npy".format(save_dir))
GA_pos_a = np.load("{}GA_dPTE_alpha_pos.npy".format(save_dir))
GA_pos_b = np.load("{}GA_dPTE_beta_pos.npy".format(save_dir))

# reorder the data in my preferred label order
i_labs = {0:'inferiorparietal-rh',1:'inferiortemporal-rh',2:'lateraloccipital-rh',3:'supramarginal-rh',
          4:'transversetemporal-rh',5:'amygdala-lh',6:'amygdala-rh'}
labels = ['amygdala-lh','amygdala-rh','lateraloccipital-rh','inferiorparietal-rh','inferiortemporal-rh','supramarginal-rh','transversetemporal-rh']
ind = np.array([5,6,2,0,1,3,4])
GA_neg_a = GA_neg_a[ind,:][:,ind]
GA_neg_b = GA_neg_b[ind,:][:,ind]
GA_pos_a = GA_pos_a[ind,:][:,ind]
GA_pos_b = GA_pos_b[ind,:][:,ind]
# subtract 0.5 from values to center on 0 and use a divergent colormap - leaving zeroes untouched
GA_z_neg_a = GA_neg_a.copy()
GA_z_neg_a[GA_neg_a!=0] -= 0.5
GA_z_neg_b = GA_neg_b.copy()
GA_z_neg_b[GA_neg_b!=0] -= 0.5
GA_z_pos_a = GA_pos_a.copy()
GA_z_pos_a[GA_pos_a!=0] -= 0.5
GA_z_pos_b = GA_pos_b.copy()
GA_z_pos_b[GA_pos_b!=0] -= 0.5

# PLOT ALL - CATS x FREQS
# now build the figure
fig = plt.figure(figsize=(10, 10))

sub1 = fig.add_subplot(221)
sub1.set_title('dPTE Negative Alpha')
sub1.imshow(GA_z_neg_a,vmin=-0.003,vmax=0.003,cmap='seismic')
sub1.set_xticks(np.arange(len(labels)))
sub1.set_yticks(np.arange(len(labels)))
sub1.set_xticklabels(labels)
sub1.set_yticklabels(labels)
# Loop over data dimensions and create text annotations.
for i in range(len(labels)):
    for j in range(len(labels)):
        text = sub1.text(j, i, str(GA_neg_a[i, j])[:5], ha="center", va="center", color="w")

sub2 = fig.add_subplot(222)
sub2.set_title('dPTE Positive Alpha')
sub2.imshow(GA_z_pos_a,vmin=-0.003,vmax=0.003,cmap='seismic')
sub2.set_xticks(np.arange(len(labels)))
sub2.set_yticks(np.arange(len(labels)))
sub2.set_xticklabels(labels)
sub2.set_yticklabels(labels)
# Loop over data dimensions and create text annotations.
for i in range(len(labels)):
    for j in range(len(labels)):
        text = sub2.text(j, i, str(GA_pos_a[i, j])[:5], ha="center", va="center", color="w")

sub3 = fig.add_subplot(223)
sub3.set_title('dPTE Negative Beta')
sub3.imshow(GA_z_neg_b,vmin=-0.003,vmax=0.003,cmap='seismic')
sub3.set_xticks(np.arange(len(labels)))
sub3.set_yticks(np.arange(len(labels)))
sub3.set_xticklabels(labels)
sub3.set_yticklabels(labels)
# Loop over data dimensions and create text annotations.
for i in range(len(labels)):
    for j in range(len(labels)):
        text = sub3.text(j, i, str(GA_neg_b[i, j])[:5], ha="center", va="center", color="w")

sub4 = fig.add_subplot(224)
sub4.set_title('dPTE Positive Beta')
sub4.imshow(GA_z_pos_b,vmin=-0.003,vmax=0.003,cmap='seismic')
sub4.set_xticks(np.arange(len(labels)))
sub4.set_yticks(np.arange(len(labels)))
sub4.set_xticklabels(labels)
sub4.set_yticklabels(labels)
# Loop over data dimensions and create text annotations.
for i in range(len(labels)):
    for j in range(len(labels)):
        text = sub4.text(j, i, str(GA_pos_b[i, j])[:5], ha="center", va="center", color="w")

# for each subplot, rotate the tick labels and set their alignment
plt.sca(sub1)
plt.setp(sub1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.sca(sub2)
plt.setp(sub2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.sca(sub3)
plt.setp(sub3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.sca(sub4)
plt.setp(sub4.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

plt.tight_layout()

# DIFFERENCE PLOT NEG-POS
GA_diff_a = GA_z_neg_a - GA_z_pos_a
GA_diff_b = GA_z_neg_b - GA_z_pos_b

# now build the figure
fig = plt.figure(figsize=(10, 10))

sub1 = fig.add_subplot(211)
sub1.set_title('dPTE Alpha Neg-Pos Difference')
sub1.imshow(GA_diff_a,vmin=-0.003,vmax=0.003,cmap='seismic')
sub1.set_xticks(np.arange(len(labels)))
sub1.set_yticks(np.arange(len(labels)))
sub1.set_xticklabels(labels)
sub1.set_yticklabels(labels)
# Loop over data dimensions and create text annotations.
for i in range(len(labels)):
    for j in range(len(labels)):
        text = sub1.text(j, i, str(GA_diff_a[i, j])[:5], ha="center", va="center", color="w")

sub2 = fig.add_subplot(212)
sub2.set_title('dPTE Beta Neg-Pos Difference')
sub2.imshow(GA_diff_b,vmin=-0.003,vmax=0.003,cmap='seismic')
sub2.set_xticks(np.arange(len(labels)))
sub2.set_yticks(np.arange(len(labels)))
sub2.set_xticklabels(labels)
sub2.set_yticklabels(labels)
# Loop over data dimensions and create text annotations.
for i in range(len(labels)):
    for j in range(len(labels)):
        text = sub2.text(j, i, str(GA_diff_b[i, j])[:5], ha="center", va="center", color="w")
# for each subplot, rotate the tick labels and set their alignment
plt.sca(sub1)
plt.setp(sub1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.sca(sub2)
plt.setp(sub2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

plt.tight_layout()
