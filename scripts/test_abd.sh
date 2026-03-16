#!/bin/bash
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1
SEED=2025
DATASET=SABS # CHAOST2  SABS
FOLD=(0 1 2 3 4)  # indicate testing fold (will be trained on the rest!)
DATA=/home/ac/data/cq/MCCFA/data/SABS
ALL_SUPP=(0 1 2 3 4 5 6 7) # CHAOST2: 0-4, CMR: 0-7 SABS: 0-6
# Setting=setting2
# Run.
for EVAL_FOLD in "${FOLD[@]}"
do
  MODELPATH=/home/cq/medical_code/MCCFA/scripts/results_sabs_2d/train/fold${EVAL_FOLD}/model.pth
  for SUPP_IDX in "${ALL_SUPP[@]}"
  do
    # SAVE_FOLDER=/home/cq/medical_code/ADNet-main/scripts/results_abd_2d/test/fold${EVAL_FOLD}/${Setting}/${SUPP_IDX}
    SAVE_FOLDER=/home/cq/medical_code/MCCFA/scripts/results_sabs_set2/test/fold${EVAL_FOLD}/${SUPP_IDX}
    if [ ! -d $SAVE_FOLDER ]
    then
      mkdir -p $SAVE_FOLDER
    fi
    python3 main_inference.py \
    --data_root ${DATA} \
    --save_root ${SAVE_FOLDER} \
    --pretrained_root "${MODELPATH}" \
    --dataset ${DATASET} \
    --fold ${EVAL_FOLD} \
    --seed ${SEED} \
    --supp_idx ${SUPP_IDX}
  done
done

# Note: EP2 is default, for EP1 set --EP1 True, --n_shot 3.