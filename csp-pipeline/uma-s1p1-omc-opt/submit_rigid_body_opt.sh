#!/bin/bash
#SBATCH --output=slurmoutputs/R-%x.%j.out
#SBATCH --error=slurmoutputs/R-%x.%j.err
#SBATCH --job-name=uma-rigidbody-opt
#SBATCH --array=0-47
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=genai-mingjieliu
#SBATCH --qos=genai-mingjieliu
#SBATCH --distribution=cyclic:cyclic
#SBATCH --time=6:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

module load conda
conda activate fairchem
pwd; hostname; date

START=$((0 + SLURM_ARRAY_TASK_ID * 50))
END=$(( START + 50))

# START=2130
# END=2132
python optimize_batch.py --start $START --end $END
