from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.spm import Realign, SliceTiming, Coregister, NewSegment, Normalize12, Smooth
from nipype import Node, Workflow
from nipype.algorithms.misc import Gunzip
import os
from os.path import join as opj
import glob
from nipype.interfaces.freesurfer import MRIConvert
from nipype import IdentityInterface
import shutil


subject_list = []

infosource = Node(IdentityInterface(fields=['subject_id']), name="infosource")
infosource.iterables = [('subject_id', subject_list)]

in_dir = '/Users/jasonrussell/Documents/INPUTS/test_data'


for subject in sorted(os.listdir(in_dir)):
	if subject.startswith('.'):
		continue
	if subject.startswith('dummy'):
		continue
	subject_list.append(subject)


	
anat_file = opj('{subject_id}','scans','orig.mgz')
func_file = opj('{subject_id}','scans','fMRI_REST1.nii')

templates = {
	'anat': anat_file,
	'func': func_file
	}

selectfiles = Node(SelectFiles(templates, 
							   base_directory='/Users/jasonrussell/Documents/INPUTS/test_data'),
				   name="selectfiles")


from nipype.interfaces.matlab import MatlabCommand
MatlabCommand.set_default_paths('source')

#gunzip
gunzip_anat = Node(Gunzip(), name='gunzip_anat')

#preprocess fMRI images using SPM
# Realign motion correction
realigner = Node(interface=Realign(), name = 'realign')
realigner.inputs.register_to_mean = True
realigner.inputs.jobtype = 'estwrite'

# slice timing correction
slice_time = Node(SliceTiming(), name="slice_time")
slice_time.inputs.num_slices = 60
slice_time.inputs.time_repetition = 1.6
slice_time.inputs.time_acquisition = 1.6 - (1.6/60)
slice_time.inputs.slice_order = list(range(1, 61, 2)) + list(range(2, 61, 2))
slice_time.inputs.ref_slice = 30

# coregistration 

coreg = Node(Coregister(), name="coreg")

# segmentation
tpm_img = '/Users/jasonrussell/Documents/MATLAB/spm12/tpm/TPM.nii'
tissue1 = ((tpm_img, 1), 1, (True, False), (False, False))
tissue2 = ((tpm_img, 2), 1, (True, False), (False, False))
tissue3 = ((tpm_img, 3), 2, (True, False), (False, False))
tissue4 = ((tpm_img, 4), 3, (False, False), (False, False))
tissue5 = ((tpm_img, 5), 4, (False, False), (False, False))
tissue6 = ((tpm_img, 6), 2, (False, False), (False, False))
tissues = [tissue1, tissue2, tissue3, tissue4, tissue5, tissue6]

segmentation = Node(NewSegment(tissues=tissues), name="segmentation")

# create node to convert mgz to nii
mri_convert = Node(MRIConvert(), name='mri_convert')

# normalization
normalization = Node(Normalize12(), name="normalization")

# smooth
smooth = Node(interface=Smooth(), name="smooth")
smooth.inputs.fwhm = 6

#datasink
datasink = Node(DataSink(base_directory='/Users/jasonrussell/Documents/OUTPUTS/dl_test'),
				name = 'datasink')

preproc = Workflow(name='preproc', base_dir = '/Users/jasonrussell/Documents/OUTPUTS/dl_test')


preproc.connect([
	(infosource, selectfiles, [('subject_id', 'subject_id')]),
	(selectfiles, realigner, [('func', 'in_files')]),
	(realigner, slice_time, [('realigned_files', 'in_files')]),
	(selectfiles, mri_convert, [('anat', 'in_file')]),
	(mri_convert, gunzip_anat, [('out_file', 'in_file')]),
	(gunzip_anat, coreg, [('out_file', 'source')]),
	(realigner, coreg, [('mean_image', 'target')]),
	(gunzip_anat, coreg, [('out_file', 'apply_to_files')]),
	(coreg, segmentation, [('coregistered_files', 'channel_files')]),
	(segmentation, normalization, [('forward_deformation_field', 'deformation_file')]),
	(coreg, normalization, [('coregistered_files', 'image_to_align')]),
	(slice_time, normalization, [('timecorrected_files', 'apply_to_files')]),
	(normalization, smooth, [('normalized_files', 'in_files')]),
	(smooth, datasink, [('smoothed_files', 'smoothed')]),
	(coreg, datasink, [('coregistered_files', 'coregistered')]),
	(normalization, datasink, [('normalized_files', 'normalized')]),
	(segmentation, datasink, [('bias_corrected_images', 'bias_corrected')]),
	(realigner, datasink, [('realignment_parameters', 'realignment_parameters')])
])

if __name__ == '__main__':
	preproc.run('MultiProc', plugin_args={'n_procs': 12})

preproc.write_graph(graph2use='exec', format='png', simple_form=False)
