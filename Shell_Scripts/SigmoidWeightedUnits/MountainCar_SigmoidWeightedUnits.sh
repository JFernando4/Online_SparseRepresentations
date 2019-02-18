#!/usr/bin/env bash

number_of_runs=$1
architecture=$2

export PYTHONPATH=.
for LR in 0.01 0.004 0.001 0.0005 0.00025 0.000125 0.0000625 0.00003125
do
    echo "Learnng Rate: $LR"
    for ((i=1; i <= $number_of_runs; i++))
    do
        echo "Run $i..."
        python3 ./SigmoidWeightedUnits_Experiment.py -lr $LR -architecture $architecture -verbose
    done
done

# Parameter Sweep:
# learning rate = {0.01, 0.004, 0.001, 0.0005, 0.00025, 0.000125, 0.0000625, 0.00003125}
# architectures = {'silu-silu', 'silu-dsilu', 'dsilu-silu', 'dsilu-dsilu'}