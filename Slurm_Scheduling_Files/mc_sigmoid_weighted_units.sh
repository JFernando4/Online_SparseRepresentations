#!/usr/bin/env bash
#SBATCH --mail-user=jfhernan@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=1-30%1
#SBATCH --time=1:00:00
#SBATCH --account=def-sutton
#SBATCH --mem=1000M
#SBATCH --job-name=mc_swu
#SBATCH --output=./outputs/mc_swu-%A_%a.out

source ./bin/activate
export PYTHONPATH=.
python3 ./SigmoidWeightedUnits_Experiment.py -architecture $ARCH -lr $LR -env mountain_car
deactivate
