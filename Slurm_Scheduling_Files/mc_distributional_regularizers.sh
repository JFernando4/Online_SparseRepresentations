#!/usr/bin/env bash
#SBATCH --mail-user=jfhernan@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=1-30%1
#SBATCH --time=2:00:00
#SBATCH --account=def-sutton
#SBATCH --mem=1000M
#SBATCH --job-name=mc_distreg
#SBATCH --output=./outputs/mc_distreg-%A_%a.out

source ./bin/activate
export PYTHONPATH=.
python3 ./DistritbutionalReg_Experiment.py -env mountain_car -lr $LR -reg_factor $RF -beta $BETA -ma_alpha $MAA
deactivate

# Parameter Sweep:
# learning rate = {0.01, 0.004, 0.001, 0.00025, 0.0000625}
# reg_factor = {0.1, 0.01, 0.001}
# beta = {0.1, 0.2, 0.5}
# ma_alpha = {0.1, 0.01}
