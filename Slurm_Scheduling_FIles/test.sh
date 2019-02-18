#!/usr/bin/env bash
#SBATCH --mail-user=jfhernan@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=1-2%1
#SBATCH --time=1:00:00
#SBATCH --account=def-sutton
#SBATCH --mem=1000M
#SBATCH --job-name=test
#SBATCH --output=./outputs/test-%A_%a.out

source ./bin/activate
export PYTHONPATH=.
python3 ./ExperienceReplay_Experiment.py -buffer_size 10000 -tnet_update_freq 10 -lr 0.001 -env mountain_car
deactivate
