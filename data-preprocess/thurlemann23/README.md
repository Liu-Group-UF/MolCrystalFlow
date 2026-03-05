# Data Preprocessing: Thurlemann2023

Scripts for processing the Thurlemann et al. 2023 molecular crystal dataset.

## Dataset Description

The Thurlemann2023 dataset contains DFT-computed potential energies for 11,489 molecular crystals from the Cambridge Structural Database (CSD). Each crystal has 5 configurations corresponding to scaled lattice parameters (0.95, 0.975, 1.0, 1.05, 1.1).

### HDF5 Structure

Each CSD code entry contains:
- `z`: Number of molecules in the unit cell
- `elements`: Atomic element symbols of the unit cell (N,)
- `elements_monomer`: Atomic element symbols of the isolated molecule
- `coordinates`: Atomic coordinates in the unit cell [Å], (5, N, 3)
- `coordinates_monomer`: Coordinates of the isolated molecule [Å], (5, N_mono, 3)
- `cells`: Unit cell vectors [Å], (5, 3, 3)
- `total_energies`: Total energy of the unit cell [kJ/mol], (5,)
- `intermolecular_energy`: Intermolecular energy [kJ/mol], (5,)
- `total_energy_monomer`: Total energy of the isolated molecule [kJ/mol]

## Scripts

### hdf2xyz.py

Convert the HDF5 dataset to extended XYZ format.

```bash
# Convert entire dataset
python hdf2xyz.py --input crystal_dataset.hdf5 --output molecule_crystals.extxyz

# Convert first 100 structures (for testing)
python hdf2xyz.py --input crystal_dataset.hdf5 --output test.extxyz --max_structures 100
```

### filter_structures.py

Filter and validate molecular crystal structures:
1. Select only unstrained structures (config_index=2, lattice scale=1.0)
2. Validate that all molecules in each crystal have the same chemical formula
3. Validate that the number of detected molecules matches the expected z value
4. Assign building block indices (`bb_indices`) to each atom for downstream processing
5. Optionally output coarse-grained representations and extracted monomers

```bash
# Basic filtering (unstrained only)
python filter_structures.py --input molecule_crystals.extxyz --output valid_molcrystals.extxyz

# With coarse-grained output and bad indices
python filter_structures.py \
    --input molecule_crystals.extxyz \
    --output valid_molcrystals.extxyz \
    --coarse_grained coarse_grained.extxyz \
    --bad_indices bad_indices.npy

# Full output with monomers
python filter_structures.py \
    --input molecule_crystals.extxyz \
    --output valid_molcrystals.extxyz \
    --coarse_grained coarse_grained.extxyz \
    --monomers all_monomers.extxyz \
    --bad_indices bad_indices.npy

# Process all configurations (not just unstrained)
python filter_structures.py --input molecule_crystals.extxyz --output valid_all.extxyz --all_configs
```

### unwrap_and_split.py

Unwrap molecules across periodic boundaries and split data into train/validation/test sets:
1. Unwrap molecules so each is fully connected (not split across periodic boundaries)
2. Split by unique monomer formulas (no formula appears in multiple splits)
3. Output train, validation, and test XYZ files and index arrays

```bash
# Basic usage
python unwrap_and_split.py --input valid_molcrystals.extxyz --output_dir ./splits

# With coarse-grained structures
python unwrap_and_split.py \
    --input valid_molcrystals.extxyz \
    --output_dir ./splits \
    --coarse_grained coarse_grained.extxyz

# Custom split sizes
python unwrap_and_split.py \
    --input valid_molcrystals.extxyz \
    --output_dir ./splits \
    --train_size 8000 \
    --test_size 500 \
    --seed 123
```

**Output files:**
- `train_molcrystal.extxyz`, `val_molcrystal.extxyz`, `test_molcrystal.extxyz`: Split XYZ files
- `train_inds.npy`, `val_inds.npy`, `test_inds.npy`: Indices into original dataset
- `train_monomers.pkl`, `val_monomers.pkl`, `test_monomers.pkl`: Monomer formulas for each split
- `unwrapped_molcrystals.extxyz`: All structures with unwrapped molecules

### generate_features.py

Generate auxiliary molecular features for monomers:
1. Convert XYZ to SDF using Open Babel
2. Fix bond orders and formal charges using RDKit
3. Extract molecular features (basic, chemical, geometric)
4. Save features to pickle files for use in training

```bash
# Basic usage
python generate_features.py --input all_monomers.xyz --output_dir ./features

# With coarse-grained structures (to get num_bbs per structure)
python generate_features.py \
    --input all_monomers.xyz \
    --output_dir ./features \
    --coarse_grained coarse_grained.extxyz
```

**Output files:**
- `all_monomers.sdf`: Raw SDF from Open Babel conversion
- `all_cleaned_monomers.sdf`: Bond-fixed SDF
- `smiles_list.pkl`: SMILES strings for each monomer
- `valid_indices.npy`, `invalid_indices.npy`: Processing status indices
- `extra_molcrystal_features.pkl`: Raw features per molecule
- `agg_extra_features.pkl`: Aggregated feature arrays (basic, chemical, geometric)

### fix_bonds.py

Multi-step bond order and formal charge fixing for SDF files:
1. Try zeroing formal charges and adjusting to neutralize
2. Try fixing nitrogen and carbon valency issues
3. Try InChI round-trip to recover correct bond orders
4. Try rdDetermineBonds to re-determine bonds from 3D structure

This script implements a more comprehensive fixing procedure than the
basic fixing in `generate_features.py`, following the logic from
`multi_step_bond_fix.ipynb`.

```bash
# Basic usage
python fix_bonds.py --input all_monomers.sdf --output all_cleaned_monomers.sdf

# Save indices and SMILES to output directory
python fix_bonds.py \
    --input all_monomers.sdf \
    --output ./fixed/all_cleaned_monomers.sdf \
    --save_indices
```

**Output files:**
- `all_cleaned_monomers.sdf`: Bond-fixed SDF
- `valid_indices.npy`: Indices of successfully fixed molecules
- `invalid_indices.npy`: Indices of molecules that could not be fixed
- `bond_unfixed.npy`: Same as invalid_indices (for compatibility)
- `smiles_list.pkl`: SMILES strings extracted before fixing

### standardize_lattice.py

Standardize lattice vectors for molecular crystal structures:
1. Sort lattice vectors so that |a| <= |b| <= |c|
2. Transform to lower-triangular form via QR decomposition
3. Ensure positive diagonal elements
4. Consistently rotate atomic positions

```bash
# Single file
python standardize_lattice.py --input train_molcrystal.extxyz --output standardized_train.extxyz

# Batch process all split files
python standardize_lattice.py \
    --input_dir ./splits \
    --output_dir ./standardized \
    --pattern "*_molcrystal.extxyz"
```

### generate_training_data.py

Generate training-ready pkl.gz files from standardized XYZ files:
1. Parse XYZ files with bb_indices (building block indices per atom)
2. Compute rotation matrices and local coordinates using PCA-based equivariant axes
3. Compute fractional translations for each building block
4. Optionally add auxiliary molecular features from a pre-generated pickle file
5. Save as compressed pickle files ready for training

```bash
# Single file
python generate_training_data.py \
    --input standardized_train.extxyz \
    --output train_molcrystal.pkl.gz \
    --features agg_extra_features.pkl \
    --indices train_inds.npy

# Batch mode (all splits)
python generate_training_data.py \
    --input_dir ./standardized \
    --output_dir ./processed \
    --features ./features/agg_extra_features.pkl \
    --indices_dir ./splits
```

**Output files:**
- `train_molcrystal.pkl.gz`, `val_molcrystal.pkl.gz`, `test_molcrystal.pkl.gz`: Training-ready data

**Output dictionary keys per structure:**
- `bb_num_vec`: Number of atoms per building block
- `trans_1`: Fractional translations (center of mass) per building block
- `rotmats_1`: Rotation matrices per building block (from PCA-based equivariant axes)
- `atom_types`: Mapped atom type indices
- `local_coords`: Atom coordinates in local (rotated) frame
- `gt_coords`: Ground truth Cartesian coordinates
- `eigenvalues`: PCA eigenvalues per building block
- `lattice_1`: Lattice matrix
- `axis_flips`: Axis flip indicators per building block
- `basic_features`, `chemical_features`, `geometric_features`: Auxiliary features (if provided)

### normalize_data.py

Normalize training pkl.gz files for consistent local coordinate frames:
1. Group building blocks by their axis flip pattern
2. Use first block as reference for each flip pattern group
3. Detect 2-axis flips relative to reference and apply 180° rotation about remaining axis

The output files (`*_normalized.pkl.gz`) are the **final training-ready data files**.

```bash
# Single file
python normalize_data.py --input train_molcrystal.pkl.gz --output train_normalized.pkl.gz

# Batch mode (all splits)
python normalize_data.py --input_dir ./processed --output_dir ./normalized

# In-place normalization (creates *_normalized.pkl.gz in same directory)
python normalize_data.py --input_dir ./processed
```

**Output files:**
- `train_molcrystal_normalized.pkl.gz`: Final training data
- `val_molcrystal_normalized.pkl.gz`: Final validation data  
- `test_molcrystal_normalized.pkl.gz`: Final test data

### run_pipeline.py

All-in-one script that runs the complete preprocessing pipeline from raw HDF5 
to final training-ready pkl.gz files, organizing all outputs into a structured
directory hierarchy.

```bash
# Full pipeline (recommended)
python run_pipeline.py --input crystal_dataset.hdf5 --output_dir ./preprocessed

# Custom split sizes
python run_pipeline.py --input crystal_dataset.hdf5 --output_dir ./preprocessed \
    --train_size 10000 --test_size 750

# Resume from a specific step (if previous steps completed)
python run_pipeline.py --input crystal_dataset.hdf5 --output_dir ./preprocessed \
    --skip_to step4

# Quiet mode (suppress progress output)
python run_pipeline.py --input crystal_dataset.hdf5 --output_dir ./preprocessed -q
```

**Output directory structure:**
```
output_dir/
├── xyz/                              # XYZ files
│   ├── molecule_crystals.extxyz     # Raw from HDF5
│   ├── valid_molcrystals.extxyz     # Filtered/validated
│   ├── coarse_grained.extxyz        # Coarse-grained
│   └── all_monomers.extxyz          # Extracted monomers
├── splits/                           # Train/val/test splits
│   ├── train_molcrystal.extxyz
│   ├── val_molcrystal.extxyz
│   ├── test_molcrystal.extxyz
│   └── *_inds.npy                   # Index arrays
├── standardized/                     # Lattice-standardized
│   ├── train_molcrystal.extxyz
│   ├── val_molcrystal.extxyz
│   └── test_molcrystal.extxyz
├── sdf/                              # SDF and SMILES
│   ├── all_monomers.sdf
│   ├── all_cleaned_monomers.sdf
│   └── smiles_list.pkl
├── features/                         # Molecular features
│   ├── extra_molcrystal_features.pkl
│   └── agg_extra_features.pkl
├── processed/                        # Pre-normalized pkl.gz
│   ├── train_molcrystal.pkl.gz
│   ├── val_molcrystal.pkl.gz
│   └── test_molcrystal.pkl.gz
└── final/                            # FINAL OUTPUT
    ├── train_molcrystal_normalized.pkl.gz
    ├── val_molcrystal_normalized.pkl.gz
    └── test_molcrystal_normalized.pkl.gz
```

**Pipeline steps:**
1. Convert HDF5 to XYZ
2. Filter and validate structures
3. Unwrap molecules and split data
4. Standardize lattice vectors
5. Generate molecular features (bond fixing + extraction)
6. Generate training pkl.gz files
7. Normalize data (final output)

## Complete Pipeline

For most use cases, simply run:

```bash
python run_pipeline.py --input crystal_dataset.hdf5 --output_dir ./preprocessed
```

If you prefer to run each step manually:

```bash
# Step 1: Convert HDF5 to XYZ
python hdf2xyz.py --input crystal_dataset.hdf5 --output molecule_crystals.extxyz

# Step 2: Filter and validate structures, assign bb_indices
python filter_structures.py \
    --input molecule_crystals.extxyz \
    --output valid_molcrystals.extxyz \
    --coarse_grained coarse_grained.extxyz \
    --monomers all_monomers.extxyz \
    --bad_indices bad_indices.npy

# Step 3: Unwrap molecules and create train/val/test splits
python unwrap_and_split.py \
    --input valid_molcrystals.extxyz \
    --output_dir ./splits \
    --coarse_grained coarse_grained.extxyz \
    --train_size 10000 \
    --test_size 750

# Step 4: Standardize lattice vectors for all splits
python standardize_lattice.py \
    --input_dir ./splits \
    --output_dir ./standardized \
    --pattern "*_molcrystal.extxyz"

# Step 5a: Convert monomers XYZ to SDF (using obabel)
obabel all_monomers.extxyz -O ./features/all_monomers.sdf

# Step 5b: Fix bond orders and formal charges (multi-step)
python fix_bonds.py \
    --input ./features/all_monomers.sdf \
    --output ./features/all_cleaned_monomers.sdf \
    --save_indices

# Step 5c: Generate auxiliary molecular features
python generate_features.py \
    --input all_monomers.extxyz \
    --output_dir ./features \
    --coarse_grained coarse_grained.extxyz

# Step 6: Generate training pkl.gz files
python generate_training_data.py \
    --input_dir ./standardized \
    --output_dir ./processed \
    --features ./features/agg_extra_features.pkl \
    --indices_dir ./splits

# Step 7: Normalize data for consistent local coordinate frames (FINAL OUTPUT)
python normalize_data.py --input_dir ./processed --output_dir ./final
```

**Final output files for training:**
- `final/train_molcrystal_normalized.pkl.gz`
- `final/val_molcrystal_normalized.pkl.gz`
- `final/test_molcrystal_normalized.pkl.gz`

**Note:** Step 5a-5c can be replaced by just running `generate_features.py` which
includes basic bond fixing. Use `fix_bonds.py` separately for more comprehensive
multi-step bond fixing when the simpler approach fails on many molecules.

## Requirements

- h5py
- ase
- tqdm
- rdkit
- torch
- numpy
- openbabel (for `obabel` command)

```bash
pip install h5py ase tqdm rdkit torch numpy
conda install -c conda-forge openbabel  # or: apt-get install openbabel
```

## Reference

```bibtex
@article{thurlemann2023,
    title = {Regularized by {Physics}: {Graph} {Neural} {Network} {Parametrized} {Potentials} for the {Description} of {Intermolecular} {Interactions}},
    volume = {19},
    issn = {1549-9618},
    url = {https://doi.org/10.1021/acs.jctc.2c00661},
    doi = {10.1021/acs.jctc.2c00661},
    number = {2},
    journal = {Journal of Chemical Theory and Computation},
    author = {Thürlemann, Moritz and Böselt, Lennard and Riniker, Sereina},
    month = jan,
    year = {2023},
    pages = {562--579},
}
```
