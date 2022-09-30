#!/bin/bash

## Import settings that are common to all datasets and methods.
source run_common.sh

## A simple loop over all the datasets specified.
for arg
do
    bash run_triskSigL.sh "${arg}"
    bash run_erm.sh "${arg}"
    bash run_meanvar.sh "${arg}"
done
