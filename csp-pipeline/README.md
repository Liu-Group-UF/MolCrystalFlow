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
<formula>/results/top10_for_dft.csv
        ↓
    [3. DFT Validation (PBE-D3 & PBE-MBD)]
        ↓
<formula>/dft_combined_ranking.csv
<formula>/stability_ranking_<formula>.pdf
<formula>/relaxed_structures/
```

## Directory Structure

```
csp-pipeline-demo/
├── csp_pipeline.py              # High-level pipeline orchestrator
│
├── molcrystalflow-gen/          # MolCrystalFlow generation module
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
│   ├── relaxed_structures/      # Extracted DFT-relaxed structures
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
cd uma-s1p1-omc-opt

# Run relaxation pipeline (from generated structures)
python run_relaxation_pipeline.py \
    --input ../<formula>/generated/pred_combined.xyz \
    --output_dir ../<formula>/results \
    --num_jobs 100 \
    --structures_per_job 50

# Or submit the monitor job
sbatch submit_monitor.sh
```

Output: `<formula>/results/top10_for_dft.csv`

#### Step 3: DFT Validation

```bash
cd dft-run

# Setup DFT jobs (PBE-D3 and PBE-MBD)
python setup_dft_jobs.py \
    --input ../<formula>/results/top10_for_dft.csv \
    --output_dir ../<formula>/dft_jobs

# Submit jobs
cd ../<formula>/dft_jobs
for d in pbe-d3/structure_* pbe-mbd/structure_*; do
    cd $d && sbatch submit.sh && cd ..
done

# After completion, collect results and generate plots
cd ../..
python dft-run/collect_dft_results.py \
    --formula_dir <formula> \
    --plot \
    --extract_structures
```

Output: 
- `<formula>/dft_combined_ranking.csv`
- `<formula>/stability_ranking_<formula>.pdf`
- `<formula>/relaxed_structures/`

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

### 3. DFT Validation
- **PBE-D3(BJ)**: Fast dispersion-corrected DFT
- **PBE-MBD**: Accurate many-body dispersion for molecular crystals
- **Ranking**: Final energy-based ranking for CSP competition

## Output Files

| File | Description |
|------|-------------|
| `<formula>/molecule.xyz` | Input molecule conformer |
| `<formula>/generated/crystals_z*.xyz` | Generated structures by Z value |
| `<formula>/generated/pred_combined.xyz` | All generated structures |
| `<formula>/results/relaxation_results.csv` | All relaxed structures with metadata |
| `<formula>/results/top10_for_dft.csv` | Top 10 for DFT validation |
| `<formula>/dft_jobs/pbe-d3/` | PBE-D3 job directories |
| `<formula>/dft_jobs/pbe-mbd/` | PBE-MBD job directories |
| `<formula>/dft_combined_ranking.csv` | Final DFT rankings |
| `<formula>/stability_ranking_<formula>.pdf` | Stability ranking plot |
| `<formula>/relaxed_structures/` | Extracted DFT-relaxed structures |

## Code Organization

### csp_pipeline.py (High-Level Orchestrator)
The main entry point that coordinates all pipeline steps:
- `run_generation()` - Step 1: MolCrystalFlow structure generation
- `submit_relaxation()` - Step 2: UMA-OMC relaxation submission
- `setup_dft_jobs()` - Step 3: DFT job preparation
- `collect_dft_results()` - Step 4: Result collection and ranking
- `run_full_pipeline()` - Run all steps in sequence

### molcrystalflow-gen/packing_gen.py (Generation Module)
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
