
# the frequency bands used in dictionary form
freqs = {"theta":list(np.arange(3,8)),"alpha":list(np.arange(8,14)),"beta_low":list(np.arange(14,22)),
         "beta_high":list(np.arange(22,31)),"gamma":list(np.arange(31,47))}
freqs_g = {"gamma_high":list(np.arange(65,96,2))}
fmins = [3, 8, 14, 22, 31]
fmaxs = [7, 13, 21, 30, 46]

conditions = {'rest':'rest', 'tonbas':['tonbas','tonrat'], 'pic_n':'negative/pics', 'pic_p':'positive/pics',
              'ton_n':['negative/r1','negative/r2','negative/s1','negative/s2'], 'ton_p':['positive/r1','positive/r2','positive/s1','positive/s2']}
conds = list(conditions.keys())
conds = ['rest','tonbas']

for meg,mri in sub_dict.items():
    # epo = mne.read_epochs("{}{}-epo.fif".format(proc_dir,meg))
    # fwd = mne.read_forward_solution("{}{}-fwd.fif".format(proc_dir,meg))
    # load filter versions for lower freq bands
    filters_m = mne.beamformer.read_beamformer('{}{}-m-dics.h5'.format(proc_dir,meg))
    filters_s = mne.beamformer.read_beamformer('{}{}-s-dics.h5'.format(proc_dir,meg))
    # load CSDs for conditions to compare, and apply filters
    csd_rest = mne.time_frequency.read_csd("{}{}_rest-csd.h5".format(proc_dir,meg))
    csd_tonbas = mne.time_frequency.read_csd("{}{}_tonbas-csd.h5".format(proc_dir,meg))
    stc_m_rest, freqs_m_rest = mne.beamformer.apply_dics_csd(csd_rest.mean(fmins,fmaxs),filters_m)
    stc_m_tonbas, freqs_m_tonbas = mne.beamformer.apply_dics_csd(csd_tonbas.mean(fmins,fmaxs),filters_m)
    stc_m_diff = stc_m_tonbas - stc_m_rest / stc_m_rest
    stc_s_rest, freqs_s_rest = mne.beamformer.apply_dics_csd(csd_rest.mean(fmins,fmaxs),filters_s)
    stc_s_tonbas, freqs_s_tonbas = mne.beamformer.apply_dics_csd(csd_tonbas.mean(fmins,fmaxs),filters_s)
    stc_s_diff = stc_s_tonbas - stc_s_rest / stc_s_rest
    # copmare both filter results
    mne.viz.set_3d_backend('pyvista')
    src = mne.read_source_spaces("{}{}_oct6_mix-src.fif".format(proc_dir,meg))
    brain = stc_m_diff.plot(subjects_dir=mri_dir,subject=mri,surface='white',hemi='both',time_viewer=True,src=src)
    brain.add_annotation('aparc', borders=1, alpha=0.9)
    brain = stc_s_diff.plot(subjects_dir=mri_dir,subject=mri,surface='white',hemi='both',time_viewer=True,src=src)
    brain.add_annotation('aparc', borders=1, alpha=0.9)
