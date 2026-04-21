# Crystal Structure Prediction Pipeline

This directory contains a complete pipeline for molecular crystal structure prediction using MolCrystalFlow, UMA-OMC relaxation, and DFT validation.

## Full Pipeline Overview

```
molecule.xyz (input conformer)
        ↓
    [1. MolCrystalFlow Generation]
        ↓
<formula>/generated/pred_combined.xyz
        ↓
    [2. UMA-OMC Relaxation]
        ↓
<formula>/results/top10_for_dft.xyz
        ↓
    [3. DFT Validation (PBE-D3 & PBE-MBD)]
        ↓
<formula>/dft_combined_ranking.csv
<formula>/stability_ranking_<formula>.pdf
<formula>/relaxed_structures/

Note:
- UMA-OMC relaxation outputs live under `<formula>/results/`. In some current runs you may see an extra nested path like
  `<formula>/results/<formula>/results/` (this is produced by the underlying relaxation scripts and is where the relaxed outputs are found).
- `<formula>/relaxed_structures/` is only created after DFT result collection when `--extract_structures` is used.
```

## Directory Structure

```
csp-pipeline-demo/
├── csp_pipeline.py              # High-level pipeline orchestrator
│
├── molcrystalflow_gen/          # MolCrystalFlow generation module
│   ├── __init__.py
│   ├── packing_gen.py           # Core packing generation logic
│   └── utility_scripts/
│
├── uma-s1p1-omc-opt/            # UMA-OMC relaxation pipeline
│   ├── run_relaxation_pipeline.py
│   ├── monitor_pipeline.py
│   ├── collect_relaxation_results.py
│   └── README.md
│
├── dft-run/                     # DFT validation scripts
│   ├── setup_dft_jobs.py
│   ├── collect_dft_results.py
│   └── README.md
│
├── <FORMULA>/                   # Molecule-specific outputs
│   ├── molecule.xyz             # Input conformer
│   ├── generated/               # MolCrystalFlow outputs
│   ├── results/                 # UMA-OMC relaxation results
│   ├── dft_jobs/                # DFT job directories
│   ├── relaxed_structures/      # (Created after DFT collection) extracted DFT-relaxed structures
│   ├── dft_combined_ranking.csv # Final ranking
│   └── stability_ranking_<formula>.pdf

```

## Quick Start

### Option A: Full Pipeline (High-Level Orchestrator)

The `csp_pipeline.py` script orchestrates the entire workflow:

```bash
# Full pipeline (interactive)
python csp_pipeline.py \
    --xyz molecule.xyz \
    --ckpt_path /path/to/molcrystalflow/checkpoint \
    --full --wait

# Or run steps independently:
# Step 1: Generate structures
python csp_pipeline.py --xyz molecule.xyz --ckpt_path /path/to/ckpt --generate

# Step 2: Submit relaxation
python csp_pipeline.py --formula C6H6 --relax --wait

# Step 3: Setup DFT jobs
python csp_pipeline.py --formula C6H6 --setup_dft

# Step 4: Collect DFT results
python csp_pipeline.py --formula C6H6 --collect_dft --plot
```

### Option B: Manual Step-by-Step

#### Step 1: Generate Crystal Structures

```bash
# Using the high-level orchestrator
python csp_pipeline.py \
    --xyz molecule.xyz \
    --z_values 2 4 \
    --num_samples 100 \
    --axis_flip \
    --filter_overlap \
    --ckpt_path /path/to/molcrystalflow/checkpoint \
    --generate

# Or directly using the packing generation module
python -m molcrystalflow_gen.packing_gen \
    --xyz molecule.xyz \
    --z_values 2 4 \
    --num_samples 100 \
    --ckpt_path /path/to/checkpoint
```

Output: `<formula>/generated/pred_combined.xyz`

#### Step 2: UMA-OMC Relaxation

```bash
# High-level entrypoint
python csp_pipeline.py --formula <FORMULA> --relax --wait

# Without --wait, only rigid-body jobs are submitted.
python csp_pipeline.py --formula <FORMULA> --relax
```

Output:
- `<formula>/results/top10_for_dft.xyz`
- `<formula>/results/top10_for_dft.csv`

What `csp_pipeline.py --relax` does now:
- It runs `uma-s1p1-omc-opt/run_relaxation_pipeline.py submit_rigid_body` from `<formula>/results/`.
- That step auto-generates `submit_rigid_body_opt_generated.sh` in `<formula>/results/` and submits it with `sbatch`.
- If you also pass `--wait`, the pipeline waits for rigid-body chunk outputs, runs `combine_and_filter`, then runs `submit_cell_opt`.
- The cell-opt step auto-generates `submit_cell_opt_generated.sh` in `<formula>/results/` and submits that with `sbatch`.
- After cell-opt completes, `csp_pipeline.py` collects the results into `<formula>/results/cell_opt_all_results.csv` and `<formula>/results/top10_for_dft.csv`.

Notes:
- `--structures_per_job` maps to the rigid-body `chunk_size`.
- The current UMA-OMC stepwise CLI derives the number of jobs from input size and chunk size, so `--num_relax_jobs` is currently informational rather than enforced.
- The current underlying UMA-OMC scripts do not yet consume the high-level `--relax_conda_env`, `--relax_partition`, or `--relax_time` flags.

If you want to run the UMA step manually instead of through `csp_pipeline.py`, use:

```bash
cd <FORMULA>/results

# 1. Submit rigid-body jobs
python ../../uma-s1p1-omc-opt/run_relaxation_pipeline.py submit_rigid_body \
    --input ../generated/pred_combined.xyz \
    --chunk_size 50

# 2. After rigid-body finishes
python ../../uma-s1p1-omc-opt/run_relaxation_pipeline.py combine_and_filter --min_dist 1.0

# 3. Submit cell optimization jobs
python ../../uma-s1p1-omc-opt/run_relaxation_pipeline.py submit_cell_opt \
    --input filtered_by_inter_bb.xyz \
    --chunk_size 10

# 4. After cell-opt finishes, collect results
python ../../uma-s1p1-omc-opt/collect_relaxation_results.py \
    --results_dir cell-opt-results \
    --log_dir cell-opt-logs \
    --filter_indices filtered_by_inter_bb_indices.npy \
    --output_dir . \
    --formula <FORMULA> \
    --top_n 10
```

#### Step 3: DFT Validation

```bash
# High-level entrypoint
python csp_pipeline.py --formula <FORMULA> --setup_dft

# Submit all generated DFT jobs
python csp_pipeline.py --formula <FORMULA> --submit_dft

# Or submit and wait for completion
python csp_pipeline.py --formula <FORMULA> --submit_dft --wait

# After completion, collect results and generate plots
python csp_pipeline.py --formula <FORMULA> --collect_dft --plot

# Structures are extracted by default during --collect_dft.
# To skip generating <formula>/relaxed_structures/, add: --no_structures
```

Output: 
- `<formula>/dft_combined_ranking.csv`
- `<formula>/stability_ranking_<formula>.pdf`
- `<formula>/relaxed_structures/` (created by default during DFT result collection; disable with --no_structures)

What `csp_pipeline.py --setup_dft` does now:
- It uses `<formula>/results/top10_for_dft.xyz` as the structure input for DFT setup.
- It passes the single-molecule formula explicitly, so job names stay as `<FORMULA>` rather than auto-detecting the full unit-cell composition.
- It calls `dft-run/setup_dft_jobs.py` once with `--theories d3 mbd`.
- That script creates `<formula>/dft_jobs/pbe-d3/`, `<formula>/dft_jobs/pbe-mbd/`, and `<formula>/dft_jobs/submit_all_dft.sh`.

If you want to run the DFT setup manually instead of through `csp_pipeline.py`, use:

```bash
cd dft-run

python setup_dft_jobs.py \
    --input ../<FORMULA>/results/top10_for_dft.xyz \
    --output_dir ../<FORMULA>/dft_jobs \
    --formula <FORMULA> \
    --theories d3 mbd

cd ../<FORMULA>/dft_jobs
bash submit_all_dft.sh
```

## Example: C9H9N3O5

```bash
# Option 1: Full pipeline with high-level orchestrator
python csp_pipeline.py \
    --xyz molecule.xyz \
    --ckpt_path /path/to/ckpt \
    --full \
    --gt_energies 3rd-csp-competition-ground-truth/gt_energies.json

# Option 2: Step-by-step
# 1. Generate structures
python csp_pipeline.py --xyz molecule.xyz --z_values 2 4 --num_samples 100 \
    --axis_flip --ckpt_path /path/to/ckpt --generate

# 2. Relax with UMA-OMC
python csp_pipeline.py --formula C9H9N3O5 --relax --wait

# 3. Setup and submit DFT
python csp_pipeline.py --formula C9H9N3O5 --setup_dft
python csp_pipeline.py --formula C9H9N3O5 --submit_dft --wait

# 4. Collect results and plot
python csp_pipeline.py --formula C9H9N3O5 --collect_dft --plot \
     --gt_energies 3rd-csp-competition-ground-truth/gt_energies.json
```

## Pipeline Steps Detail

### 1. MolCrystalFlow Generation
- Input: Molecule conformer XYZ file
- Parameters: Z values, number of samples, axis flip, overlap filtering
- Output: Generated crystal structures organized by formula

### 2. UMA-OMC Relaxation
- **Rigid-body optimization**: Optimize molecular positions/rotations
- **Inter-BB filtering**: Remove structures with overlapping molecules
- **Cell optimization**: Full cell + atomic relaxation with UMA-OMC
- **Ranking**: Sort by predicted energy, extract top-N for DFT
- **Submission scripts**: `submit_rigid_body_opt_generated.sh` and `submit_cell_opt_generated.sh` are generated inside `<formula>/results/` when the corresponding steps are submitted

### 3. DFT Validation
- **PBE-D3(BJ)**: Fast dispersion-corrected DFT
- **PBE-MBD**: Accurate many-body dispersion for molecular crystals
- **Ranking**: Final energy-based ranking for CSP competition
- **Setup input**: DFT setup consumes `top10_for_dft.xyz` and uses the molecular formula supplied by `csp_pipeline.py` for naming

## Output Files

| File | Description |
|------|-------------|
| `<formula>/molecule.xyz` | Input molecule conformer |
| `<formula>/generated/crystals_z*.xyz` | Generated structures by Z value |
| `<formula>/generated/pred_combined.xyz` | All generated structures |
| `<formula>/results/submit_rigid_body_opt_generated.sh` | Auto-generated SLURM array script for rigid-body UMA-OMC jobs |
| `<formula>/results/submit_cell_opt_generated.sh` | Auto-generated SLURM array script for cell-opt UMA-OMC jobs |
| `<formula>/results/cell_opt_all_results.csv` | All relaxed structures with metadata |
| `<formula>/results/top10_for_dft.xyz` | Top 10 relaxed structures for DFT setup |
| `<formula>/results/top10_for_dft.csv` | Top 10 for DFT validation |
| `<formula>/dft_jobs/pbe-d3/` | PBE-D3 job directories |
| `<formula>/dft_jobs/pbe-mbd/` | PBE-MBD job directories |
| `<formula>/dft_jobs/submit_all_dft.sh` | Master script to submit all generated DFT jobs |
| `<formula>/dft_combined_ranking.csv` | Final DFT rankings |
| `<formula>/stability_ranking_<formula>.pdf` | Stability ranking plot |
| `<formula>/relaxed_structures/` | (Created after DFT collection) Extracted DFT-relaxed structures |

## Code Organization

### csp_pipeline.py (High-Level Orchestrator)
The main entry point that coordinates all pipeline steps:
- `run_generation()` - Step 1: MolCrystalFlow structure generation
- `submit_relaxation()` - Step 2: UMA-OMC stepwise submission and optional waiting/collection
- `setup_dft_jobs()` - Step 3: DFT job preparation
- `collect_dft_results()` - Step 4: Result collection and ranking
- `run_full_pipeline()` - Run all steps in sequence

### molcrystalflow_gen/packing_gen.py (Generation Module)
Core packing generation logic:
- `generate_crystal_structures()` - High-level generation function
- `load_model()` - Load MolCrystalFlow checkpoint
- `extract_molecule_features()` - Feature extraction
- `run_inference()` - Run model inference
- `filter_structures_by_overlap()` - Overlap filtering

## Requirements

- Python 3.10+
- PyTorch, ASE, pymatgen, RDKit, Open Babel
- MolCrystalFlow model checkpoint
- UMA-OMC model (via fairchem)
- VASP 6.x with D3/MBD support
