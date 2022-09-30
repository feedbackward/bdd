#!/bin/bash

## Settings specific to traditional ERM.
RISK_NAME="erm"
STEP_SIZE=("0.1" "0.5" "1.0" "1.5" "2.0")

## Dataset name is passed to this script.
DATA="$1"

## Run the driver script for the prescribed settings.
for idx_s in "${!STEP_SIZE[@]}"
do
    TASK="s${idx_s}r0"
    python "learn_driver.py" --algo-ancillary="$ALGO_ANCILLARY" --algo-main="$ALGO_MAIN" --batch-size="$BATCH_SIZE" --data="$DATA" --dispersion="$DISPERSION" --entropy="$ENTROPY" --loss-base="$LOSS_BASE" --model="$MODEL" --noise-rate="$NOISE_RATE" --num-epochs="$NUM_EPOCHS" --num-trials="$NUM_TRIALS" --risk-name="$RISK_NAME" --save-dist="$SAVE_DIST" --step-size="${STEP_SIZE[idx_s]}" --task-name="$TASK"
done
