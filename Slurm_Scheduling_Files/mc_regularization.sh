#!/usr/bin/env bash
#SBATCH --mail-user=jfhernan@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=1-30%1
#SBATCH --time=2:00:00
#SBATCH --account=def-sutton
#SBATCH --mem=1000M
#SBATCH --job-name=mc_reg
#SBATCH --output=./outputs/mc_reg-%A_%a.out

source ./bin/activate
export PYTHONPATH=.
python3 ./Regularization_Experiment.py -env mountain_car -reg $REG -lr $LR -layer1_factor $L1F -layer2_factor $L2F
deactivate
