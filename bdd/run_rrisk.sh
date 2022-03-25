#!/bin/bash

## Settings specific to "R-Risk".
ALPHA=("0.0" "0.5" "1.0" "1.5" "2.0")
DISPERSION="barron"
ETA="1.0"
RISK_NAME="rrisk"
SIGMA="1.0"
STEP_SIZE=("0.1" "0.5" "1.0" "1.25")

## Dataset name is passed to this script.
DATA="$1"

## Run the driver script for the prescribed settings.
for idx_s in "${!STEP_SIZE[@]}"
do
    for idx_r in "${!ALPHA[@]}"
    do
	TASK="s${idx_s}r${idx_r}"
	python "learn_driver.py" --algo-ancillary="$ALGO_ANCILLARY" --algo-main="$ALGO_MAIN" --alpha="${ALPHA[idx_r]}" --batch-size="$BATCH_SIZE" --data="$DATA" --dispersion="$DISPERSION" --entropy="$ENTROPY" --eta="$ETA" --loss-base="$LOSS_BASE" --model="$MODEL" --num-epochs="$NUM_EPOCHS" --num-trials="$NUM_TRIALS" --risk-name="$RISK_NAME" --sigma="$SIGMA" --save-dist="$SAVE_DIST" --step-size="${STEP_SIZE[idx_s]}" --task-name="$TASK"
    done
done
