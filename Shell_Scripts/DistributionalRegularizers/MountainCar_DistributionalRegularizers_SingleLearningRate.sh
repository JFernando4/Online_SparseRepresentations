#!/usr/bin/env bash

number_of_runs=$1
LR=$2
RF=$3
B=$4
MA=$5

echo "Number of Runs: $number_of_runs"
echo "Learnng Rate: $LR"
echo "Regularization Factor: $RF"
echo "Beta: $B"
echo "Moving Average Alpha: $MA"
for ((i=1; i <= $number_of_runs; i++))
do
    echo "Run $i..."
    python3 ./DistritbutionalReg_Experiment.py -lr $LR -reg_factor $RF -beta $B -ma_alpha $MA -verbose
done

# Parameter Sweep:
# learning rate = {0.01, 0.004, 0.001, 0.00025, 0.0000625}
# reg_factor = {0.1, 0.01, 0.001}
# beta = {0.1, 0.2, 0.5}
# ma_alpha = {0.1, 0.01}