#!/bin/bash

#initialize fs

export FREESURFER_HOME=/Applications/freesurfer/7.4.1
source $FREESURFER_HOME/SetUpFreeSurfer.sh
pip install nipype

# Define the base directory
BASE_DIR="/Users/jasonrussell/Documents/INPUTS/test_data"

# Iterate through each subject directory
for SUBJECT_DIR in "$BASE_DIR"/*; do
    # Check if the path is a directory
    if [ -d "$SUBJECT_DIR" ]; then
        # Define the path to the fMRI_REST1.nii.gz file
        FILE_PATH="$SUBJECT_DIR/scans/fMRI_REST1.nii.gz"
        
        # Check if the file exists
        if [ -f "$FILE_PATH" ]; then
            # Unzip the file
            echo "Unzipping: $FILE_PATH"
            gunzip "$FILE_PATH"
        else
            echo "File not found: $FILE_PATH"
        fi
    fi
done

echo 'fs_initialized, fMRI unzipped'
