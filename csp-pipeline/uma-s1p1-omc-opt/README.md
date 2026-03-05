# UMA-OMC Crystal Structure Relaxation Pipeline

This directory contains scripts for post-generation relaxation of MolCrystalFlow-predicted crystal structures using the UMA-OMC (Universal Machine Learning Atomic - Organic Molecular Crystal) model.

## Pipeline Overview

```
data/xyz/pred_combined.xyz (original MolCrystalFlow generated structures)
        ↓
    [Rigid-body optimization with UMA-OMC]
        ↓
data/xyz/filtered_by_inter_bb.xyz (filtered by inter-BB distance > 1.0 Å)
        ↓
    [Cell + atomic optimization with UMA-OMC]
        ↓
<FORMULA>/results/cell_opt_all_results.csv (all relaxed structures with metadata)
<FORMULA>/results/top10_for_dft.xyz (top 10 lowest energy for DFT validation)
```

## Index Chain (Traceability)

Each structure can be traced back through the pipeline:

| Index | Description |
|-------|-------------|
| `orig_idx` | Index in `data/xyz/pred_combined.xyz` (original generated) |
| `filtered_idx` | Index in `data/xyz/filtered_by_inter_bb.xyz` (after inter-BB filtering) |
| `cell_opt_idx` | Index in combined cell-opt results |

The mapping `filtered_idx → orig_idx` is stored in `data/npy/filtered_by_inter_bb_indices.npy`.

## Quick Start

### Option 1: One-shot automated pipeline
```bash
# Submit a single job that handles everything
sbatch submit_monitor.sh
```

### Option 2: Step-by-step
```bash
# 1. Combine generated XYZ files
python run_relaxation_pipeline.py combine_input --input_dir ./generated --output data/xyz/pred_combined.xyz

# 2. Submit rigid-body optimization
python run_relaxation_pipeline.py submit_rigid_body --input data/xyz/pred_combined.xyz

# 3. After rigid-body completes, combine and filter
python run_relaxation_pipeline.py combine_and_filter --min_dist 1.0

# 4. Submit cell optimization
python run_relaxation_pipeline.py submit_cell_opt --input data/xyz/filtered_by_inter_bb.xyz

# 5. After cell-opt completes, collect results (auto-organizes by formula)
python collect_relaxation_results.py --top_n 10
```

## Main Scripts

| Script | Purpose |
|--------|---------|
| `collect_relaxation_results.py` | Collect cell-opt results, compute properties, extract top-N for DFT |
| `run_relaxation_pipeline.py` | Interactive/stepwise pipeline orchestration |
| `monitor_pipeline.py` | Automated pipeline monitor for one-shot submission |
| `optimize_batch.py` | Rigid-body optimization batch script |
| `cell_opt_optimize_batch.py` | Cell + atomic optimization batch script |

## Directory Structure

```
uma-s1p1-omc-opt/
├── data/
│   ├── xyz/                     # Input XYZ files
│   │   ├── pred_combined.xyz    # Combined generated structures
│   │   └── filtered_by_inter_bb.xyz  # Filtered structures
│   └── npy/                     # Index mapping files
│       └── filtered_by_inter_bb_indices.npy
│
├── opt-results/                 # Rigid-body optimization outputs
├── log-files/                   # Rigid-body optimization logs
├── cell-opt-results/            # Cell optimization outputs
├── cell-opt-logs/               # Cell optimization logs
│
├── <FORMULA>/                   # Results organized by molecule formula
│   └── results/
│       ├── cell_opt_all_results.csv  # All structures with full metadata
│       ├── top10_for_dft.xyz         # Top 10 for DFT evaluation
│       └── top10_for_dft.csv         # Metadata for top 10
│
├── collect_relaxation_results.py    # Main collection script
├── run_relaxation_pipeline.py       # Pipeline orchestration
├── monitor_pipeline.py              # Automated monitor
├── optimize_batch.py                # Rigid-body batch optimization
├── cell_opt_optimize_batch.py       # Cell-opt batch optimization
│
├── submit_monitor.sh            # SLURM: one-shot pipeline
├── submit_rigid_body_opt.sh     # SLURM: rigid-body array job
├── submit_cell_opt.sh           # SLURM: cell-opt array job
│
├── utils/                       # Helper modules
│   └── rigid_constraint.py      # Rigid-body constraint implementation
│
└── archive/                     # Deprecated/old files
```

## Output CSV Columns

`results/cell_opt_all_results.csv`:

| Column | Description |
|--------|-------------|
| `cell_opt_idx` | Index in combined cell-opt output |
| `filtered_idx` | Index in filtered_by_inter_bb.xyz |
| `orig_idx` | Index in pred_combined.xyz (original) |
| `n_atoms` | Number of atoms |
| `n_bb` | Number of building blocks (molecules) |
| `formulas` | Chemical formula of each building block |
| `dominant_formula` | Single formula if all BBs identical |
| `opt_steps` | BFGS optimization steps |
| `total_energy_eV` | Total potential energy (eV) |
| `E_per_mol_eV` | Energy per molecule (eV/mol) |
| `max_force_eV_A` | Maximum force at convergence (eV/Å) |
| `density_g_cm3` | Crystal density (g/cm³) |
| `a, b, c` | Lattice parameters (Å) |
| `alpha, beta, gamma` | Lattice angles (degrees) |
| `volume` | Unit cell volume (Å³) |

## Usage Examples

### Collect results with custom settings
```bash
python collect_relaxation_results.py \
    --results_dir cell-opt-results \
    --log_dir cell-opt-logs \
    --filter_indices filtered_by_inter_bb_indices.npy \
    --output_dir results \
    --top_n 10 \
    --save_combined_xyz  # Optional: also save combined XYZ
```

### Submit pipeline with custom parameters
```bash
# Custom chunk sizes and filtering threshold
RIGID_CHUNK_SIZE=100 CELL_CHUNK_SIZE=20 MIN_INTER_BB_DIST=1.5 sbatch submit_monitor.sh
```

## Notes

- Missing job chunks due to hardware failure are automatically detected and reported
- Energy values are in eV; density in g/cm³
- Top-N structures include full metadata in XYZ header for DFT input
