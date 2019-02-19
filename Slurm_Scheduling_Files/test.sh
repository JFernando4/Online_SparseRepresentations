#!/usr/bin/env bash
#SBATCH --mail-user=jfhernan@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=1-2%1
#SBATCH --time=1:00:00
#SBATCH --account=def-sutton
#SBATCH --mem=1000M
#SBATCH --job-name=test
#SBATCH --output=./outputs/test-%A_%a.out

echo "the buffer size is: $BF"
echo "the target network update frequency is: $LR"

#source ./bin/activate
#export PYTHONPATH=.
#python3 ./ExperienceReplay_Experiment.py -buffer_size $BS -tnet_update_freq $FREQ -lr $LR -env $ENV
#deactivate
