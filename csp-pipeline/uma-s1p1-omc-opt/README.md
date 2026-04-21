# UMA-OMC Crystal Structure Relaxation Pipeline

This directory contains the UMA-OMC relaxation scripts used after MolCrystalFlow generation. In the current repo flow, these scripts are usually driven by `csp_pipeline.py`, but they can also be run manually.

## Overview

Current per-formula flow:

```text
<FORMULA>/generated/pred_combined.xyz
        ↓
    [Rigid-body optimization with UMA-OMC]
        ↓
<FORMULA>/results/filtered_by_inter_bb.xyz
        ↓
    [Cell + atomic optimization with UMA-OMC]
        ↓
<FORMULA>/results/cell_opt_all_results.csv
<FORMULA>/results/top10_for_dft.xyz
<FORMULA>/results/top10_for_dft.csv
```

## How It Is Used

There are two supported modes:

1. Through the high-level wrapper in `csp_pipeline.py`
2. Directly by calling `run_relaxation_pipeline.py` subcommands

When called through `csp_pipeline.py`, the working directory is:

```text
<FORMULA>/results/
```

and all generated submission scripts, logs, chunk outputs, and collected files are written there.

## Quick Start

### Via `csp_pipeline.py`

From `csp-pipeline/`:

```bash
# Submit rigid-body UMA jobs only
python csp_pipeline.py --formula <FORMULA> --relax

# Submit rigid-body jobs, wait for completion, then continue through
# filtering, cell-opt submission, and result collection
python csp_pipeline.py --formula <FORMULA> --relax --wait
```

What this currently does:
- `--relax` submits rigid-body jobs only
- `--relax --wait` waits for rigid-body chunks, runs filtering, submits cell-opt jobs, waits for cell-opt chunks, and collects final CSV/XYZ outputs
- submission scripts are generated as:
  - `<FORMULA>/results/submit_rigid_body_opt_generated.sh`
  - `<FORMULA>/results/submit_cell_opt_generated.sh`

Current limitation:
- the wrapper still accepts `--num_relax_jobs`, `--relax_conda_env`, `--relax_partition`, and `--relax_time`, but the underlying stepwise UMA CLI does not fully consume those overrides yet

### Manual Step-by-Step

From `csp-pipeline/`:

```bash
cd <FORMULA>/results

# 1. Submit rigid-body jobs
python ../../uma-s1p1-omc-opt/run_relaxation_pipeline.py submit_rigid_body \
    --input ../generated/pred_combined.xyz \
    --chunk_size 50

# 2. After rigid-body completes, combine and filter
python ../../uma-s1p1-omc-opt/run_relaxation_pipeline.py combine_and_filter \
    --min_dist 1.0

# 3. Submit cell optimization jobs
python ../../uma-s1p1-omc-opt/run_relaxation_pipeline.py submit_cell_opt \
    --input filtered_by_inter_bb.xyz \
    --chunk_size 10

# 4. After cell-opt completes, collect results
python ../../uma-s1p1-omc-opt/collect_relaxation_results.py \
    --results_dir cell-opt-results \
    --log_dir cell-opt-logs \
    --filter_indices filtered_by_inter_bb_indices.npy \
    --output_dir . \
    --formula <FORMULA> \
    --top_n 10
```

### One-Shot Monitor Job

```bash
sbatch submit_monitor.sh
```

This is still available, but the high-level repo workflow now primarily uses `csp_pipeline.py --relax`.

## Main Scripts

| Script | Purpose |
|--------|---------|
| `run_relaxation_pipeline.py` | Stepwise orchestration for rigid-body, filtering, and cell-opt |
| `collect_relaxation_results.py` | Collect final cell-opt outputs and extract top-N for DFT |
| `monitor_pipeline.py` | One-shot monitor-based execution |
| `optimize_batch.py` | Rigid-body batch optimization worker |
| `cell_opt_optimize_batch.py` | Cell + atomic optimization worker |

## Folder Structure

This directory contains the reusable scripts:

```text
uma-s1p1-omc-opt/
├── run_relaxation_pipeline.py
├── collect_relaxation_results.py
├── monitor_pipeline.py
├── optimize_batch.py
├── cell_opt_optimize_batch.py
├── submit_monitor.sh
├── submit_rigid_body_opt.sh
├── submit_cell_opt.sh
└── utils/
```

Typical per-formula runtime directory created under `csp-pipeline/`:

```text
<FORMULA>/results/
├── submit_rigid_body_opt_generated.sh
├── submit_cell_opt_generated.sh
├── opt-results/
├── log-files/
├── filtered_by_inter_bb.xyz
├── filtered_by_inter_bb_indices.npy
├── cell-opt-results/
├── cell-opt-logs/
├── pred_uma_rigid_body_opt.xyz
├── pred_uma_cell_opt_final.xyz
├── cell_opt_all_results.csv
├── top10_for_dft.xyz
└── top10_for_dft.csv
```

## Index Chain

Each structure can be traced through the relaxation pipeline:

| Index | Description |
|-------|-------------|
| `orig_idx` | Index in `<FORMULA>/generated/pred_combined.xyz` |
| `filtered_idx` | Index in `<FORMULA>/results/filtered_by_inter_bb.xyz` |
| `cell_opt_idx` | Index in combined cell-opt results |

The mapping `filtered_idx -> orig_idx` is stored in:

```text
<FORMULA>/results/filtered_by_inter_bb_indices.npy
```

## Output Columns

`<FORMULA>/results/cell_opt_all_results.csv` includes:

| Column | Description |
|--------|-------------|
| `cell_opt_idx` | Index in combined cell-opt output |
| `filtered_idx` | Index in filtered XYZ |
| `orig_idx` | Index in original generated XYZ |
| `n_atoms` | Number of atoms |
| `n_bb` | Number of molecules/building blocks in the cell |
| `formulas` | Building block formulas |
| `dominant_formula` | Single formula if all building blocks match |
| `opt_steps` | BFGS steps |
| `total_energy_eV` | Total potential energy |
| `E_per_mol_eV` | Energy per molecule |
| `max_force_eV_A` | Final maximum force |
| `density_g_cm3` | Crystal density |
| `a, b, c` | Lattice lengths |
| `alpha, beta, gamma` | Lattice angles |
| `volume` | Unit cell volume |

## Notes

- Missing chunk files are detectable from the chunk patterns in `opt-results/` and `cell-opt-results/`
- Energy values are reported in eV; density in g/cm^3
- The collected top-N XYZ/CSV files are the inputs to the DFT step
