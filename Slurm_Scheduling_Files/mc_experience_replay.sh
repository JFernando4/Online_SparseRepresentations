#!/usr/bin/env bash
#SBATCH --mail-user=jfhernan@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=1-30
#SBATCH --time=2:30:00
#SBATCH --account=def-sutton
#SBATCH --mem=1000M
#SBATCH --job-name=mc_er
#SBATCH --output=./outputs/mc_er-%A_%a.out

source ./bin/activate
export PYTHONPATH=.
python3 ./ExperienceReplay_Experiment.py -buffer_size $BUFFER -tnet_update_freq $FREQ -lr $LR -env mountain_car -run_number $SLURM_ARRAY_TASK_ID
deactivate

# Parameter Sweep:
# learning rate = {0.01, 0.004, 0.001, 0.00025, 0.0000625}
# buffer size = {10k, 20k, 40k}
# target network update frequency = {10, 50, 100, 200, 400}
    # We tested a frequency of 1 but in most runs learning was very brittle. In the few runs where the network
    # actually managed to learn, it often forgot what it had learned after a few episodes.