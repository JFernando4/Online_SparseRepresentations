#!/usr/bin/env bash
#SBATCH --mail-user=jfhernan@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=1-30%1
#SBATCH --time=3:00:00
#SBATCH --account=def-sutton
#SBATCH --mem=1000M
#SBATCH --job-name=mc_reg
#SBATCH --output=./outputs/mc_reg-%A_%a.out

source ./bin/activate
export PYTHONPATH=.
python3 ./Regularization_Experiment.py -env mountain_car -reg $REG -lr $LR -layer1_factor $L1F -layer2_factor $L2F -olayer_factor $LoF -verbose
deactivate

# Parameter Sweep:
# learning rate = {0.004, 0.001, 0.00025}
# reg_factor_layer1 = {0, 0.1, 0.01, 0.001}
# reg_factor_layer2 = {0, 0.1, 0.01, 0.001}
# reg_factor_output_layer = {0, 0.1, 0.01, 0.001}
    # No regularization is applied to the output layer
