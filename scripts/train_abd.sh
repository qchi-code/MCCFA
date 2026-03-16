#!/bin/bash
GPUID1=3
export CUDA_VISIBLE_DEVICES=$GPUID1
# Specs.
SEED=2025
DATASET=SABS
FOLD=(0 1 2 3 4)  # indicate testing fold (will be trained on the rest!)
DATA=/home/ac/data/cq/MCCFA/data/SABS
# Run.
for EVAL_FOLD in "${FOLD[@]}"
do
  SAVE_FOLDER=/home/cq/medical_code/MCCFA/scripts/results_sabs_2d/train/fold${EVAL_FOLD}
  if [ ! -d $SAVE_FOLDER ]
  then
    mkdir -p $SAVE_FOLDER
  fi
  python3 main_train.py \
  --data_root ${DATA} \
  --save_root ${SAVE_FOLDER} \
  --dataset ${DATASET} \
  --n_sv 5000 \
  --fold ${EVAL_FOLD} \
  --seed ${SEED}
done