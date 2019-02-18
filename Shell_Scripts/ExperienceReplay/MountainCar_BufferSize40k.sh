#!/usr/bin/env bash

number_of_runs=$1

export PYTHONPATH=.
for FREQ in 10 50 100 200 400
do
    echo "Target Network Update Frequency: $FREQ"
    for ((i=1; i <= $number_of_runs; i++))
    do
        echo "Run $i..."
        python3 ./ExperienceReplay_Experiment.py -buffer_size 40000 -tnet_update_freq $FREQ -verbose
    done
done