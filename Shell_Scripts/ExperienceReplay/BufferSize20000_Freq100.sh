#!/usr/bin/env bash

echo Enter number of runs:
read number_of_runs

export PYTHONPATH=.
for (( i=1; i <= $number_of_runs; i++ ))
do
    echo "Run $i..."
    python3 ./ExperienceReplay_Experiment.py -buffer_size 20000 -tnet_update_freq 100
done
