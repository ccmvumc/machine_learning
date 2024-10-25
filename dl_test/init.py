import os
import glob
import shutil

in_dir = '/Users/jasonrussell/Documents/INPUTS/test_data'

for subject in sorted(os.listdir(in_dir)):
	if subject.startswith('.'):
		continue
	if subject.startswith('dummy'):
		continue
	if not os.path.exists(f'{in_dir}/{subject}/scans'):
		os.mkdir(f'{in_dir}/{subject}/scans')
	else:
		print('Directory exists, continue')
	anat = glob.glob(f'{in_dir}/{subject}/{subject}_MR1/assessors/*FS7_v1*/{subject}/*FS7_v1*/out/SUBJ/mri/orig.mgz')[0]
	shutil.copy(anat, f'{in_dir}/{subject}/scans/orig.mgz')
	func = f'{in_dir}/{subject}/{subject}_MR1/scans/801/fMRI_REST1.nii.gz'
	shutil.copy(func, f'{in_dir}/{subject}/scans/fMRI_REST1.nii.gz')
	pet = glob.glob(f'{in_dir}/{subject}/assessors/*AMYLOIDQA_v4*/gtmpvc.cblmgmwm.output/rbv.nii.gz')[0]
	shutil.copy(pet, f'{in_dir}/{subject}/scans/amyloid.nii.gz')
