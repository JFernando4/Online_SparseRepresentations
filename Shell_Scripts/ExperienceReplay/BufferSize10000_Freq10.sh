#!/usr/bin/env bash

number_of_runs=$1

export PYTHONPATH=.
for (( i=1; i <= $number_of_runs; i++ ))
do
    echo "Run $i..."
    python3 ./ExperienceReplay_Experiment.py -buffer_size 10000 -tnet_update_freq 10 -verbose
done
