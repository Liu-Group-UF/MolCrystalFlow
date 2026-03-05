# DFT Calculations for Crystal Structure Ranking

This directory contains scripts for setting up and collecting VASP DFT calculations to validate and rank UMA-OMC relaxed crystal structures.

## Overview

After UMA-OMC relaxation, the top-N structures are validated with DFT calculations at two levels of theory:
- **PBE-D3(BJ)**: Fast dispersion-corrected DFT (IVDW=12)
- **PBE-MBD**: More accurate many-body dispersion (IVDW=263)

## Features

- Collect energies, densities, lattice parameters from VASP outputs
- **Unwrap molecules** in extracted structures so atoms are fully connected in Cartesian coordinates
- Include **ground-truth/experimental energies** in CSVs and plots for comparison
- Generate stability ranking plots comparing u-MLIP vs DFT

## Quick Start

### 1. Setup DFT Jobs

```bash
# From the dft-run folder:
python setup_dft_jobs.py --input ../uma-s1p1-omc-opt/C9H9N3O5/results/top10_for_dft.xyz --formula C9H9N3O5

# This creates:
# C9H9N3O5/
# ├── pbe-d3/
# │   ├── 000/, 001/, ... 009/
# │   └── submit_all.sh
# ├── pbe-mbd/
# │   ├── 000/, 001/, ... 009/
# │   └── submit_all.sh
# └── submit_all_dft.sh
```

### 2. Submit Jobs

```bash
cd C9H9N3O5
bash submit_all_dft.sh  # Submits both D3 and MBD jobs
```

Or submit individually:
```bash
cd C9H9N3O5/pbe-d3 && bash submit_all.sh
cd C9H9N3O5/pbe-mbd && bash submit_all.sh
```

### 3. Collect Results and Generate Plots

```bash
# Basic collection
python collect_dft_results.py --input_dir ./C9H9N3O5

# With stability ranking plot
python collect_dft_results.py --input_dir ./C9H9N3O5 --plot

# With ground truth comparison
python collect_dft_results.py --input_dir ./C9H9N3O5 --plot \
    --gt_energies ../3rd-csp-competition-ground-truth/gt_energies.json

# Extract top-4 structures
python collect_dft_results.py --input_dir ./C9H9N3O5 --top_n 4
```

This creates:
- `dft_results_pbe_d3.csv` - PBE-D3 results sorted by energy
- `dft_results_pbe_mbd.csv` - PBE-MBD results sorted by energy
- `dft_combined_ranking.csv` - Combined ranking with both theories and GT energies
- `relaxed_structures/` - Extracted CONTCAR structures as XYZ (molecules unwrapped)
- `stability_ranking_<formula>.pdf` - Stability ranking visualization

## Molecule Unwrapping

Extracted XYZ files have molecules **unwrapped across periodic boundaries**. This ensures:
- All atoms belonging to the same molecule are in the same periodic image
- Molecules appear fully connected in Cartesian coordinates
- No "broken" molecules split across the unit cell edges

This uses the same unwrapping method as the original data preprocessing pipeline.

## Scripts

| Script | Purpose |
|--------|---------|
| `setup_dft_jobs.py` | Create VASP job directories for PBE-D3 and PBE-MBD |
| `collect_dft_results.py` | Parse VASP outputs, extract structures, generate plots |

## Directory Structure

```
dft-run/
├── setup_dft_jobs.py           # Main job setup script
├── collect_dft_results.py      # Results collection and plotting
├── README.md                   # This file
│
├── <FORMULA>/                  # Formula-specific DFT jobs
│   ├── pbe-d3/
│   │   ├── 000/
│   │   │   ├── POSCAR
│   │   │   ├── INCAR
│   │   │   ├── KPOINTS
│   │   │   ├── POTCAR
│   │   │   ├── CONTCAR        # Relaxed structure (after job completes)
│   │   │   └── run_vasp.slurm
│   │   ├── 001/, 002/, ...
│   │   └── submit_all.sh
│   │
│   ├── pbe-mbd/
│   │   └── (same structure)
│   │
│   ├── submit_all_dft.sh       # Master submission script
│   │
│   ├── dft_results_pbe_d3.csv  # Results after collection
│   ├── dft_results_pbe_mbd.csv
│   ├── dft_combined_ranking.csv
│   │
│   ├── relaxed_structures/     # Extracted structures
│   │   ├── pbe_d3_all.xyz
│   │   ├── pbe_mbd_all.xyz
│   │   └── top4_pbe_mbd.xyz
│   │
│   └── stability_ranking_<formula>.pdf  # Ranking plot
│
└── archive/                    # Old scripts and notebooks
```

## Stability Ranking Plot

The `--plot` option generates a visualization comparing relative energies across methods:

```
        u-MLIP    PBE-D3    PBE-MBD
    ─────────────────────────────────
    α ══════════════════════════════  (lowest energy)
         ·····╲
    β ══════════╲═══════════════════
              ·····╲
    γ ══════════════╲═══════════════
```

- **Horizontal bars**: Energy level for each polymorph at each method
- **Dotted lines**: Connect same polymorph across methods
- **Greek letters**: α, β, γ, δ for top-1, 2, 3, 4 structures
- **GT**: Ground truth (experimental) if provided

## Ground Truth Energies

Create a JSON file with experimental/reference energies:

```json
{
  "C9H9N3O5": {
    "PBE-D3": -169.9671220475,
    "PBE-MBD": -169.8013839275,
    "u-MLIP": -169.5819626074171
  }
}
```

## VASP Settings

### PBE-D3(BJ)
```
GGA    = PE
IVDW   = 12       # D3(BJ) dispersion
ENCUT  = 520
EDIFF  = 1E-7
EDIFFG = -0.01
ISIF   = 3        # Full cell relaxation
NSW    = 1500
```

### PBE-MBD
```
GGA    = PE
IVDW   = 263      # MBD@rsSCS dispersion
ENCUT  = 520
EDIFF  = 1E-7
EDIFFG = -0.01
ISIF   = 3        # Full cell relaxation
NSW    = 1500
```

## SLURM Configuration

Default settings (HiPerGator):
- Account: `mingjieliu`
- Time: 2 days (D3), 3 days (MBD)
- Nodes: 1, Tasks: 32
- Memory: 2gb per CPU

Modify in `setup_dft_jobs.py` if needed.

## Output CSV Columns

`dft_results_pbe_*.csv`:

| Column | Description |
|--------|-------------|
| `job_idx` | Index matching input XYZ structure |
| `theory` | DFT level (pbe-d3 or pbe-mbd) |
| `converged` | Whether ionic relaxation converged |
| `n_ionic_steps` | Number of VASP ionic steps |
| `total_energy_eV` | Total DFT energy (eV) |
| `E_per_mol_eV` | Energy per molecule (eV) |
| `max_force_eV_A` | Maximum force at convergence |
| `volume_A3` | Cell volume (Å³) |
| `density_g_cm3` | Crystal density |
| `a, b, c` | Lattice parameters (Å) |
| `alpha, beta, gamma` | Lattice angles (°) |
| `rank` | Ranking by energy |

`dft_combined_ranking.csv` (additional columns when `--gt_energies` is used):

| Column | Description |
|--------|-------------|
| `GT_E_per_mol_d3_eV` | Ground-truth PBE-D3 energy (eV/mol) |
| `GT_E_per_mol_mbd_eV` | Ground-truth PBE-MBD energy (eV/mol) |
| `GT_E_per_mol_umlip_eV` | Ground-truth u-MLIP energy (eV/mol) |
| `rel_E_mbd_to_GT_eV` | Energy relative to GT (eV/mol) |
| `rel_E_d3_to_GT_eV` | Energy relative to GT (eV/mol) |

## Full Pipeline

```
molecule.xyz
    ↓ [MolCrystalFlow generation]
<formula>/generated/pred_combined.xyz
    ↓ [UMA-OMC relaxation]
<formula>/results/top10_for_dft.xyz
    ↓ [DFT validation]
<formula>/dft_combined_ranking.csv
```

## Troubleshooting

### Missing POTCAR
Ensure `potcar_root` points to your VASP pseudopotential library:
```bash
python setup_dft_jobs.py --input top10.xyz --potcar_root /path/to/potpaw_PBE
```

### Collecting incomplete results
The collection script handles incomplete jobs gracefully. Re-run after more jobs finish.
