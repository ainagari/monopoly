#!/bin/bash

# Run this script to extract representations and similarities for all languages, datasets and models


# ENGLISH

for DATASET in mono_poly polysemy_bands
do
python monopoly_extract_reps.py --language en --dataset $DATASET --control_type poly-bal
python monopoly_extract_reps.py --language en --dataset $DATASET --control_type poly-bal --cased
python monopoly_extract_reps.py --language en --dataset $DATASET --control_type poly-bal --multilingual --cased
done

for CONTROL in	poly-rand poly-same
do
python monopoly_extract_reps.py --language en --dataset polysemy_bands --control_type $CONTROL
python monopoly_extract_reps.py --language en --dataset polysemy_bands --control_type $CONTROL --cased
python monopoly_extract_reps.py --language en --dataset polysemy_bands --control_type $CONTROL --multilingual --cased
done



# OTHER LANGUAGES

for LANG in es el fr
do
for DATASET in mono_poly polysemy_bands
do

python monopoly_extract_reps.py --language $LANG --dataset $DATASET --control_type poly-bal
python monopoly_extract_reps.py --language $LANG --dataset $DATASET --control_type poly-bal --multilingual --cased

done
done

for LANG in  es el fr
do
for CONTROL in poly-rand poly-same
do
python monopoly_extract_reps.py --language $LANG --dataset polysemy_bands --control_type $CONTROL
python monopoly_extract_reps.py --language $LANG --dataset polysemy_bands --control_type $CONTROL --multilingual --cased

done
done









# 22/02/2021 re-running mono poly, where I changed the poly words.

#for LANG in es el fr
#do

#python monopoly_extract_reps.py --language $LANG --dataset mono_poly --control_type control
#python monopoly_extract_reps.py --language $LANG --dataset mono_poly --control_type control --multilingual 
#python monopoly_extract_reps.py --language $LANG --dataset mono_poly --control_type control --multilingual --cased

#done

