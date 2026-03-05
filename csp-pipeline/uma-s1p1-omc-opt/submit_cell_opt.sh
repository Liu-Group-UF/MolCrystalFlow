#!/bin/bash
#SBATCH --output=slurmoutputs/R-%x.%j.out
#SBATCH --error=slurmoutputs/R-%x.%j.err
#SBATCH --job-name=uma-molcrystalflow-cell-opt
#SBATCH --array=0-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=genai-mingjieliu
#SBATCH --qos=genai-mingjieliu
#SBATCH --distribution=cyclic:cyclic
#SBATCH --time=24:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

module load conda
conda activate fairchem
pwd; hostname; date

START=$((1700 + SLURM_ARRAY_TASK_ID * 10))
END=$(( START + 10))

# START=2130
# END=2132
python3 cell_opt_optimize_batch.py --start $START --end $END
