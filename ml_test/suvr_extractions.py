# -*- coding: utf-8 -*-

import os
from nilearn import image, datasets
import nibabel as nib
import pandas as pd
import glob
from nilearn.input_data import NiftiLabelsMasker
import ants
from antspynet import brain_extraction

in_dir = '/home/jason/INPUTS/test_data'
out_dir = '/home/jason/OUTPUTS/ml_test'
mni_ni = datasets.load_mni152_template()

nib.save(mni_ni, f'{out_dir}/atlasni.nii.gz')
mni=f'{out_dir}/atlasni.nii.gz'
fixed = ants.image_read(mni)

#make output directory
if not os.path.exists(out_dir):
	os.mkdir(out_dir)
else:
	print('Directory exists, continue')

atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-1mm')

#save atlas as nii
nib.save(atlas.maps, f'{out_dir}/havox_atlas.nii.gz')
havox_atlas=f'{out_dir}/havox_atlas.nii.gz'
move_atlas = ants.image_read(havox_atlas)

subject_suvrs = pd.DataFrame()
subject_suvrs['roi'] = atlas.labels[1:50]

for subject in sorted(os.listdir(in_dir)):
	if subject.startswith('.'):
		# ignore hidden files and other junk
		continue
	
	subject_path = f'{in_dir}/{subject}'
	subject_out = f'{out_dir}/{subject}'
	
	if not os.path.exists(subject_out):
		os.mkdir(subject_out)
	else:
		print('Subject Out Directory exists, continue')
	
	subject_amyloid = glob.glob(f'{subject_path}/assessors/*AMYLOIDQA_v4*/gtmpvc.cblmgmwm.output/rbv.nii.gz')[0]
	
	
	print(f'Amyloid: {subject_amyloid}')
	
	# load amyloid PET data 
	pet_img = image.load_img(subject_amyloid)
	pet_ants = ants.image_read(subject_amyloid)
	
	#import MRI as ANTs image
	orig_file = glob.glob(f'{subject_path}/*MR1/assessors/*FS7_v1*/{subject}/*FS7_v1*/out/SUBJ/mri/orig.mgz')[0]
	
	#transform MRI to MNI space
	# Skull Strip Original T1

	raw = ants.image_read(orig_file)
	extracted_mask = brain_extraction(raw, modality='t1')
	
	#save mask
	mask_file = f'{subject_out}/brain_mask.nii.gz'
	ants.image_write(extracted_mask, mask_file)
	
	#Apply mask with skull stripped
	masked_image = ants.mask_image(raw, extracted_mask)
	
	# Load orig T1 image as moving image for registration
	moving = masked_image

	# Do Registration of Moving to Fixed
	reg = ants.registration(fixed, moving, type_of_transform='SyN')
	
	#inverse transform atlas.maps to PET space for subject
	warped_atlas = ants.apply_transforms(
		pet_ants, move_atlas, reg['invtransforms'], interpolator='nearestNeighbor'
		)
	warped_atlas_file = f'{subject_out}/warped_havox_atlas.nii.gz'
	ants.image_write(warped_atlas, warped_atlas_file)
	
	#load mask
	warped_atlas_ni = image.load_img(warped_atlas_file)
	masker = NiftiLabelsMasker(labels_img=warped_atlas_ni, standardize = False)
	
	roi_suvrs = masker.fit_transform(pet_img)
	
	reshaped_suvrs = roi_suvrs.reshape(roi_suvrs.shape[1])
	
	subject_suvrs[subject] = reshaped_suvrs
	
subject_suvrs.to_csv(f'{out_dir}/suvrs.csv', index=False)

