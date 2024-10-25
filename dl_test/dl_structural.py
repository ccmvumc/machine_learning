import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from nilearn import image
import glob
import pandas as pd
import nibabel as nib
import os
import numpy as np
import torch.nn.functional as F


out_dir = '/Users/jasonrussell/Documents/OUTPUTS/dl_test'
in_dir = '/Users/jasonrussell/Documents/INPUTS/test_data'

#to run on mbp
device = torch.device("cpu")

#Define CNN for each modality
class MRI_CNN(nn.Module):
	def __init__(self):
		super(MRI_CNN, self).__init__()
		self.conv1 = nn.Conv3d(1, 16, kernel_size=5)
		self.conv2 = nn.Conv3d(16, 32, kernel_size=5)
		self.pool = nn.MaxPool3d(2)
		self.fc1 = nn.Linear(32 * 46 * 55 * 44, 128)
		
	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 32 * 46 * 55 * 44)
		x = F.relu(self.fc1(x))
		return x
		
		
class PET_CNN(nn.Module):
	def __init__(self):
		super(PET_CNN, self).__init__()
		self.conv1 = nn.Conv3d(1, 16, kernel_size=5)
		self.conv2 = nn.Conv3d(16, 32, kernel_size=5)
		self.pool = nn.MaxPool3d(2)
		self.fc1 = nn.Linear(32 * 46 * 55 * 44, 128)
		
	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 32 * 46 * 55 * 44)
		x = F.relu(self.fc1(x))
		return x
		
# Define the fusion model
class MultimodalFusionModel(nn.Module):
	def __init__(self):
		super(MultimodalFusionModel, self).__init__()
		self.mri_cnn = MRI_CNN()
		self.pet_cnn = PET_CNN()
		# Fully connected layer for classification after fusion
		self.fc = nn.Linear(128 * 2, 2)  #2 modalities, 2 classes (binary classification)

	def forward(self, mri, pet):
		mri_features = self.mri_cnn(mri)
		pet_features = self.pet_cnn(pet)
		
		# Concatenate features from all modalities
		fused_features = torch.cat((mri_features, pet_features), dim=1)
		
		# Classification
		out = self.fc(fused_features)
		return out
		

class MultimodalDataset(Dataset):
	def __init__(self, mri_paths, pet_paths, labels):
		# MRI, PET file paths and corresponding labels
		self.mri_paths = mri_paths
		self.pet_paths = pet_paths
		self.labels = labels
	
	def __len__(self):
		# Returns the total number of samples
		return len(self.labels)
	
	def __getitem__(self, idx):
		# Load MRI and PET images
		mri_img = nib.load(self.mri_paths[idx]).get_fdata()
		pet_img = nib.load(self.pet_paths[idx]).get_fdata()
		
		# Normalize images
		mri_img = (mri_img - mri_img.mean()) / mri_img.std()
		pet_img = (pet_img - pet_img.mean()) / pet_img.std()
		
		# Convert to PyTorch tensors and add channel dimension
		mri_img = torch.tensor(mri_img, dtype=torch.float32).unsqueeze(0)
		pet_img = torch.tensor(pet_img, dtype=torch.float32).unsqueeze(0)
		
		# Load label
		label = torch.tensor(self.labels[idx], dtype=torch.long)
		
		return mri_img, pet_img, label

# Initialize the model
model = MultimodalFusionModel().to(device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

#load images images
amyloid_img_paths = glob.glob(f'{out_dir}/*/smoothed_warped_amyloid.nii.gz')
mri_img_paths = glob.glob(f'{out_dir}/*/warped_orig.nii.gz')
amyloid_img_paths = sorted(amyloid_img_paths)
mri_img_paths = sorted(mri_img_paths)

#generate subject list in order of import
subject_list_amyloid=[]
subject_list_mri=[]

for subject in mri_img_paths:
	path = os.path.normpath(subject)
	sub = path.split(os.sep)
	sub_id = sub[-2]
	subject_list_mri.append(sub_id)

for subject in amyloid_img_paths:
	path = os.path.normpath(subject)
	sub = path.split(os.sep)
	sub_id = sub[-2]
	subject_list_amyloid.append(sub_id)

print(f'MRI subject order: {subject_list_mri}')
print(f'Amyloid subject order: {subject_list_amyloid}')

if subject_list_mri == subject_list_amyloid:
	print("subject inputs match")
else:
	print("ERROR - SUBJECTS DON'T MATCH")

subs_array = np.array(subject_list_mri)

# Import categorical data
cat_df = pd.read_csv(f'{in_dir}/dummy_data.csv')
cat_df_sorted = cat_df.sort_values('id')

print(cat_df_sorted)

categorizer = cat_df_sorted['CN_to_MCI'].tolist()
categorizer, categorizer_key = pd.factorize(np.array(categorizer))

print(categorizer_key)

# Create Dataset and DataLoader
dataset = MultimodalDataset(mri_img_paths, amyloid_img_paths, categorizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True) #update batch size when more data

# Training loop
for epoch in range(10):  # Number of epochs
	model.train()
	running_loss = 0.0
	for i, (mri, pet, labels) in enumerate(dataloader):
		mri, pet, labels = mri.to(device), pet.to(device), labels.to(device)
		
		# Zero the parameter gradients
		optimizer.zero_grad()
		
		# Forward pass
		outputs = model(mri, pet)
		loss = criterion(outputs, labels)
		
		# Backward pass and optimization
		loss.backward()
		optimizer.step()
		
		# Print loss
		running_loss += loss.item()
		if i % 10 == 9:  # Print every 10 batches
			print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10:.3f}')
			running_loss = 0.0
