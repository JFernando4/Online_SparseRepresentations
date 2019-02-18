#!/usr/bin/env bash

number_of_runs=$1
LR=$2

export PYTHONPATH=.
for FREQ in 10 50 100 200 400
do
    echo "Target Network Update Frequency: $FREQ"
    for ((i=1; i <= $number_of_runs; i++))
    do
        echo "Run $i..."
        python3 ./ExperienceReplay_Experiment.py -buffer_size 10000 -tnet_update_freq $FREQ -verbose -lr $LR
    done
done

# Parameter Sweep:
# learning rate = {0.01, 0.004, 0.001, 0.00025, 0.0000625}
# buffer size = {10k, 20k, 40k}
# target network update frequency = {10, 50, 100, 200, 400}