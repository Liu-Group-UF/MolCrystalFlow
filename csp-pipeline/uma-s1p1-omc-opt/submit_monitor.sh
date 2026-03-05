#!/bin/bash
#SBATCH --output=slurmoutputs/R-%x.%j.out
#SBATCH --error=slurmoutputs/R-%x.%j.err
#SBATCH --job-name=pipeline-monitor
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --account=genai-mingjieliu
#SBATCH --qos=genai-mingjieliu
#SBATCH --time=72:00:00
#SBATCH --partition=hpg-default

# =============================================================================
# ONE-SHOT Pipeline Monitor Job
# =============================================================================
# Submit this SINGLE job to run the ENTIRE UMA-OMC relaxation pipeline:
#
#   1. Submits rigid-body optimization jobs (array job)
#   2. Monitors until all rigid-body jobs complete
#   3. Combines all rigid-body results
#   4. Filters structures by inter-BB distance (> 1.0 Å)
#   5. Submits cell optimization jobs (array job)
#   6. Monitors until all cell-opt jobs complete
#   7. Combines all cell-opt results
#
# Prerequisites:
#   - Input XYZ file exists (e.g., pred_combined.xyz from MolCrystalFlow generation)
#
# Usage:
#   # Default: uses pred_combined.xyz as input
#   sbatch submit_monitor.sh
#
#   # Custom input file:
#   INPUT_XYZ=my_crystals.xyz sbatch submit_monitor.sh
#
#   # Skip rigid-body (already done, start from cell-opt):
#   SKIP_RIGID=true sbatch submit_monitor.sh
#
#   # Custom chunk sizes:
#   RIGID_CHUNK_SIZE=100 CELL_CHUNK_SIZE=20 sbatch submit_monitor.sh
# =============================================================================

module load conda
conda activate fairchem

pwd; hostname; date

# Create output directories
mkdir -p slurmoutputs
mkdir -p opt-results
mkdir -p cell-opt-results
mkdir -p cell-opt-logs
mkdir -p log-files

# Configuration
INPUT_XYZ="${INPUT_XYZ:-pred_combined.xyz}"
RIGID_CHUNK_SIZE="${RIGID_CHUNK_SIZE:-50}"
CELL_CHUNK_SIZE="${CELL_CHUNK_SIZE:-10}"
MIN_INTER_BB_DIST="${MIN_INTER_BB_DIST:-1.0}"
SKIP_RIGID="${SKIP_RIGID:-false}"

echo "=============================================="
echo "Pipeline Monitor Configuration"
echo "=============================================="
echo "Input XYZ: $INPUT_XYZ"
echo "Rigid-body chunk size: $RIGID_CHUNK_SIZE"
echo "Cell-opt chunk size: $CELL_CHUNK_SIZE"
echo "Min inter-BB distance: $MIN_INTER_BB_DIST Å"
echo "Skip rigid-body monitoring: $SKIP_RIGID"
echo "=============================================="

# Build command
CMD="python monitor_pipeline.py --mode full --input $INPUT_XYZ"
CMD="$CMD --rigid_chunk_size $RIGID_CHUNK_SIZE"
CMD="$CMD --cell_chunk_size $CELL_CHUNK_SIZE"
CMD="$CMD --min_dist $MIN_INTER_BB_DIST"

if [ "$SKIP_RIGID" = "true" ]; then
    CMD="$CMD --skip_rigid_body"
fi

echo "Running: $CMD"
echo ""

# Run the pipeline monitor
$CMD

EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Pipeline completed successfully!"
else
    echo "Pipeline failed with exit code $EXIT_CODE"
fi
echo "=============================================="

date
exit $EXIT_CODE
