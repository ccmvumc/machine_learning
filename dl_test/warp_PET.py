# -*- coding: utf-8 -*-

import glob
import os

import ants
from antspynet import brain_extraction
from nilearn import datasets
import nibabel as nib


in_dir = '/Users/jasonrussell/Documents/INPUTS/test_data'
atlas_ni = datasets.load_mni152_template()
out_dir = '/Users/jasonrussell/Documents/OUTPUTS/dl_test'


# Convert atlas_file to ants image fixed target for registration

nib.save(atlas_ni, f'{out_dir}/atlasni.nii.gz')
atlas=f'{out_dir}/atlasni.nii.gz'
fixed = ants.image_read(atlas)

for subject in sorted(os.listdir(in_dir)):
	if subject.startswith('.'):
		# ignore hidden files and other junk
		continue
	if subject.startswith('covariates'):
		# ignore covariates csv
		continue
	if subject.startswith('dummy_'):
		continue

	subject_amyloid = f'{in_dir}/{subject}/scans/amyloid.nii.gz'

	print('Amyloid:', subject_amyloid)

	subject_out = f'{out_dir}/{subject}'

	os.makedirs(subject_out)

	# Get full file path to input images
	orig_file = f'{in_dir}/{subject}/scans/orig.mgz'
	amyloid_file =  subject_amyloid
	
	# Skull Strip Original T1
	raw = ants.image_read(orig_file)
	extracted_mask = brain_extraction(raw, modality='t1')

	#Apply mask with skull stripped
	masked_image = ants.mask_image(raw, extracted_mask)

	# Load orig T1 image as moving image for registration
	moving = masked_image

	# Do Registration of Moving to Fixed
	reg = ants.registration(fixed, moving, type_of_transform='SyN')

	# Save warped orig
	warped_orig_file = f'{subject_out}/warped_orig.nii.gz'
	ants.image_write(reg['warpedmovout'], warped_orig_file)

	# Apply transform to FEOBV image which is already in same space
	warped_amyloid_file = f'{subject_out}/warped_FEOBV.nii.gz'
	amyloid = ants.image_read(amyloid_file)
	warped_amyloid = ants.apply_transforms(fixed, amyloid, reg['fwdtransforms'])
	ants.image_write(warped_amyloid, warped_amyloid_file)

	# Smoothing FEOBV
	smoothed_warped_amyloid_file = f'{subject_out}/smoothed_warped_amyloid.nii.gz'
	smoothed_amyloid = ants.smooth_image(warped_amyloid, 3)
	ants.image_write(smoothed_amyloid, smoothed_warped_amyloid_file)
	
