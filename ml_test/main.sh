#!/bin/bash

source ./initialize.sh
python ./fmri_preprocessing.py
python ./suvr_extractions.py
python ./fmri_amy_predict.py

