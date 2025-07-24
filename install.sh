#!/bin/bash

YOUR_PATH_TO_CONDA=anaconda3



# Sklearn==0.21.3 might be wrong in some scenarios, thus we fix bugs with the scripts below.
# Please set your path to conda here
bash fix_contents.sh ${YOUR_PATH_TO_CONDA}/envs/off-moo/lib/python3.8/site-packages/sklearn/cross_decomposition/pls_.py "pinv2" "pinv"




