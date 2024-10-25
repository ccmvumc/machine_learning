import os
from nilearn import datasets
import glob
import pandas as pd
from nilearn.input_data import NiftiLabelsMasker
from sklearn.model_selection import GridSearchCV, train_test_split
from nilearn.connectome import ConnectivityMeasure
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import joblib


in_dir = '/home/jason/INPUTS/test_data'

out_dir = '/home/jason/OUTPUTS/ml_test'

atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-1mm')

#import csv containing amyloid suvrs
amyloid = pd.read_csv(f'{out_dir}/suvrs.csv')

amyloid_suvrs = amyloid.T.iloc[1:,:].to_numpy()

y = amyloid_suvrs

fmri_ls = []

#import images to list
for subject in sorted(os.listdir(in_dir)):
	if subject.startswith('.'):
		# ignore hidden files and other junk
		continue
	
	subject_path = f'{in_dir}/{subject}'
	
	fmri_path = glob.glob(f'{out_dir}/preproc/*{subject}/smooth/swarfMRI_REST1.nii')
	
	fmri_ls.append(fmri_path[0])
	
print("Data paths set, preparing ML model")

masker = NiftiLabelsMasker(labels_img=atlas.maps, standardize=True)
time_series = [masker.fit_transform(fmri_img) for fmri_img in fmri_ls]

# Calculate connectivity matrices (functional connectivity) for each participant
connectivity_measure = ConnectivityMeasure(kind='correlation')
connectivity_matrices = connectivity_measure.fit_transform(time_series)

# Flatten the connectivity matrices into feature vectors
fmri_features = [conn_matrix.flatten() for conn_matrix in connectivity_matrices]
fmri_features = np.array(fmri_features)

# Align the feature matrix (X) with SUVR data (Y)
X = fmri_features 
y = amyloid_suvrs 

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#setup hyperparameter tuning
param_grid = {
    'estimator__alpha': [0.1, 1.0, 10.0, 100.0]
}


print("Data prepared, fitting ML model")
# Multi-Output Ridge Regression
model = MultiOutputRegressor(Ridge())
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', verbose=1, n_jobs=-1)
print('Running test on ML model')
grid_search.fit(X_train, y_train)

# Get the best model and its hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")

# Get the best hyperparameters and the best cross-validated score
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Best Cross-validated R-squared: {grid_search.best_score_}")

#Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
r_squared = r2_score(y_test, y_pred, multioutput='uniform_average')
print(f"Test R-squared: {r_squared}")

# Save the trained model to a file
joblib.dump(best_model, f'{out_dir}/trained_model.pkl')