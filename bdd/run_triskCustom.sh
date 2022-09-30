#!/bin/bash

## Settings specific to "T-Risk".
ALPHA=("-2.0" "-1.5" "-1.0" "-0.5" "0.0" "0.5" "1.0" "1.5" "2.0")
DISPERSION="barron"
ETATILDE="1.0"
RISK_NAME="triskCustom"
SET_THRESHOLD="median"
SIGMA="0.99"
STEP_SIZE=("0.1" "0.5" "1.0" "1.5" "2.0")

## Dataset name is passed to this script.
DATA="$1"

## Run the driver script for the prescribed settings.
for idx_s in "${!STEP_SIZE[@]}"
do
    for idx_r in "${!ALPHA[@]}"
    do
	TASK="s${idx_s}r${idx_r}"
	python "learn_driver.py" --algo-ancillary="$ALGO_ANCILLARY" --algo-main="$ALGO_MAIN" --alpha="${ALPHA[idx_r]}" --batch-size="$BATCH_SIZE" --data="$DATA" --dispersion="$DISPERSION" --entropy="$ENTROPY" --etatilde="$ETATILDE" --loss-base="$LOSS_BASE" --model="$MODEL" --noise-rate="$NOISE_RATE" --num-epochs="$NUM_EPOCHS" --num-trials="$NUM_TRIALS" --risk-name="$RISK_NAME" --save-dist="$SAVE_DIST" --set-threshold="$SET_THRESHOLD" --sigma="$SIGMA" --step-size="${STEP_SIZE[idx_s]}" --task-name="$TASK"
    done
done
