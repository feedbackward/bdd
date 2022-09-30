#!/bin/bash

## Settings specific to "CVaR".
PROB=("0.15" "0.3" "0.45" "0.6" "0.75")
RISK_NAME="cvar"
STEP_SIZE=("0.1" "0.5" "1.0" "1.5" "2.0")

## Dataset name is passed to this script.
DATA="$1"

## Run the driver script for the prescribed settings.
for idx_s in "${!STEP_SIZE[@]}"
do
    for idx_r in "${!PROB[@]}"
    do
	TASK="s${idx_s}r${idx_r}"
	python "learn_driver.py" --algo-ancillary="$ALGO_ANCILLARY" --algo-main="$ALGO_MAIN" --batch-size="$BATCH_SIZE" --data="$DATA" --entropy="$ENTROPY" --loss-base="$LOSS_BASE" --model="$MODEL" --noise-rate="$NOISE_RATE" --num-epochs="$NUM_EPOCHS" --num-trials="$NUM_TRIALS" --prob="${PROB[idx_r]}" --risk-name="$RISK_NAME" --save-dist="$SAVE_DIST" --step-size="${STEP_SIZE[idx_s]}" --task-name="$TASK"
    done
done
