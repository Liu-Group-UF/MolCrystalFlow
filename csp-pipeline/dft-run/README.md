# DFT Calculations for Crystal Structure Ranking

This directory contains the VASP DFT setup and collection scripts used after UMA-OMC relaxation.

## Overview

After relaxation, the top-N structures are validated with DFT at two levels:
- `pbe-d3` / theory `d3`
- `pbe-mbd` / theory `mbd`

Current high-level flow:

```text
<FORMULA>/results/top10_for_dft.xyz
        ↓
    [DFT job setup]
        ↓
<FORMULA>/dft_jobs/pbe-d3/
<FORMULA>/dft_jobs/pbe-mbd/
        ↓
    [DFT runs]
        ↓
<FORMULA>/dft_combined_ranking.csv
<FORMULA>/stability_ranking_<FORMULA>.pdf
<FORMULA>/relaxed_structures/
```

## Quick Start

### Via `csp_pipeline.py`

From `csp-pipeline/`:

```bash
# Create DFT job folders from the top-N relaxed XYZ
python csp_pipeline.py --formula <FORMULA> --setup_dft

# Submit all generated jobs
python csp_pipeline.py --formula <FORMULA> --submit_dft

# Or submit and wait
python csp_pipeline.py --formula <FORMULA> --submit_dft --wait

# After jobs finish, collect results and generate the plot
python csp_pipeline.py --formula <FORMULA> --collect_dft --plot
```

Current wrapper behavior:
- DFT setup uses `<FORMULA>/results/top10_for_dft.xyz`, not the CSV
- the single-molecule formula is passed explicitly so job names stay at the molecular formula, not the full unit-cell composition
- setup creates:
  - `<FORMULA>/dft_jobs/pbe-d3/`
  - `<FORMULA>/dft_jobs/pbe-mbd/`
  - `<FORMULA>/dft_jobs/submit_all_dft.sh`

### Manual DFT Setup

From `csp-pipeline/dft-run/`:

```bash
python setup_dft_jobs.py \
    --input ../<FORMULA>/results/top10_for_dft.xyz \
    --output_dir ../<FORMULA>/dft_jobs \
    --formula <FORMULA> \
    --theories d3 mbd
```

### Submit Jobs

```bash
cd ../<FORMULA>/dft_jobs
bash submit_all_dft.sh
```

Or submit individually:

```bash
cd ../<FORMULA>/dft_jobs/pbe-d3 && bash submit_all.sh
cd ../<FORMULA>/dft_jobs/pbe-mbd && bash submit_all.sh
```

### Collect Results Manually

From `csp-pipeline/dft-run/`:

```bash
python collect_dft_results.py --formula_dir ../<FORMULA>
python collect_dft_results.py --formula_dir ../<FORMULA> --plot
python collect_dft_results.py --formula_dir ../<FORMULA> --top_n 4
python collect_dft_results.py --formula_dir ../<FORMULA> --plot \
    --gt_energies ../3rd-csp-competition-ground-truth/gt_energies.json
```

## Scripts

| Script | Purpose |
|--------|---------|
| `setup_dft_jobs.py` | Create VASP job directories for `d3` and/or `mbd` |
| `collect_dft_results.py` | Parse finished VASP jobs, extract structures, and generate rankings/plots |

## Folder Structure

This directory contains the reusable scripts:

```text
dft-run/
├── setup_dft_jobs.py
├── collect_dft_results.py
└── README.md
```

Per-formula job directory created under `csp-pipeline/`:

```text
<FORMULA>/dft_jobs/
├── pbe-d3/
│   ├── 000/
│   │   ├── POSCAR
│   │   ├── INCAR
│   │   ├── KPOINTS
│   │   ├── POTCAR
│   │   ├── CONTCAR
│   │   └── run_vasp.slurm
│   ├── 001/, 002/, ...
│   └── submit_all.sh
├── pbe-mbd/
│   └── ...
└── submit_all_dft.sh
```

Collected outputs written under `<FORMULA>/`:

```text
<FORMULA>/
├── dft_results_pbe_d3.csv
├── dft_results_pbe_mbd.csv
├── dft_combined_ranking.csv
├── relaxed_structures/
│   ├── pbe_d3_all.xyz
│   ├── pbe_mbd_all.xyz
│   └── top4_pbe_mbd.xyz
└── stability_ranking_<FORMULA>.pdf
```

## VASP Settings

### PBE-D3(BJ)

```text
GGA    = PE
IVDW   = 12
ENCUT  = 520
EDIFF  = 1E-7
EDIFFG = -0.01
ISIF   = 3
NSW    = 1500
```

### PBE-MBD

```text
GGA    = PE
IVDW   = 263
ENCUT  = 520
EDIFF  = 1E-7
EDIFFG = -0.01
ISIF   = 3
NSW    = 1500
```

## SLURM Configuration

The DFT account and QoS are currently hardcoded in `setup_dft_jobs.py`. Edit that file if you need different values for generated `run_vasp.slurm` scripts.

## Outputs

Generated result files include:
- `dft_results_pbe_d3.csv`
- `dft_results_pbe_mbd.csv`
- `dft_combined_ranking.csv`
- `relaxed_structures/`
- `stability_ranking_<FORMULA>.pdf`

## Notes

- `setup_dft_jobs.py` expects an XYZ structure file as input
- the DFT wrapper in `csp_pipeline.py` translates `pbe-d3` / `pbe-mbd` to the script's `d3` / `mbd` theory names
- extracted XYZ structures are unwrapped so molecules remain connected across periodic boundaries
