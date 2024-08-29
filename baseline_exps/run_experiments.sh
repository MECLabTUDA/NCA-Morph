#!/bin/bash
EXP_FOLDER=/local/scratch/aranem/BMVC_2024_code_base/baseline_exps/
TRAIN_ABSTRACT_PATH=/local/scratch/aranem/BMVC_2024_code_base/scripts/train_abstract.py

for exp in 0_0.py 0_1.py 0_2.py 1_0.py 1_1.py 1_2.py 2_0.py 2_1.py 2_2.py 3_0.py 3_1.py 3_2.py
do
cp $EXP_FOLDER$exp $TRAIN_ABSTRACT_PATH
python $TRAIN_ABSTRACT_PATH
done