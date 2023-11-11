## snippets for exploring subcortical vertices, sensitivity, and label timecourses 

# test commit from vscode


src[2].keys()
dict_keys(['id', 'type', 'shape', 'src_mri_t', 'mri_ras_t', 'vox_mri_t', 'interpolator', 'mri_file', 'mri_width', 'mri_height', 'mri_depth', 'mri_volume_name', 'neighbor_vert',
           'seg_name', 'np', 'ntri', 'coord_frame', 'rr', 'nn', 'tris', 'nuse', 'inuse', 'vertno', 'nuse_tri', 'use_tris', 'nearest', 'nearest_dist', 'pinfo', 'patch_inds', 'dist', 'dist_limit', 'subject_his_id'])



vert_ids = [(s['seg_name'],s['nuse']) for s in src[2:]]
>>> vert_ids
[('Left-Thalamus-Proper', 61), ('Left-Caudate', 30), ('Left-Putamen', 40), ('Left-Pallidum', 12), ('Left-Hippocampus', 35), ('Left-Amygdala', 12), ('Right-Thalamus-Proper', 80), ('Right-Caudate', 27), ('Right-Putamen', 35), ('Right-Pallidum', 15), ('Right-Hippocampus', 40), ('Right-Amygdala', 15)]
>>> surf = sens_stc.surface()



labels = mne.read_labels_from_annot('BRA52_fa', parc='aparc', hemi='both', surf_name='white', annot_fname=None, regexp=None, subjects_dir=mri_dir, sort=False, verbose=None)
labels
sens_ltc = sens_stc.extract_label_time_course(labels,src)
sens_ltc.shape
sens_ltc[-12:]
labels_limb
