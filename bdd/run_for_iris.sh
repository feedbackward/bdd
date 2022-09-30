#!/bin/bash

export ALGO_ANCILLARY="SGD"
export ALGO_MAIN="Ave"
export BATCH_SIZE="0"
export ENTROPY="55127178903847889548173206514705792351"
export LOSS_BASE="logistic"
export MODEL="linreg_multi"
export NOISE_RATE="0.0"
export NUM_EPOCHS="30"
export NUM_TRIALS="25"
export SAVE_DIST="yes"

bash run_triskSigL.sh "iris"
bash run_erm.sh "iris"
bash run_meanvar.sh "iris"
