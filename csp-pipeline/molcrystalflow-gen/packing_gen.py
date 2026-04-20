#!/usr/bin/env python
"""
MolCrystalFlow Packing Generation Module.

This module contains all the core logic for generating molecular crystal structures
using MolCrystalFlow, including:
- Auxiliary feature computation
- Molecule feature extraction
- Inference batch creation
- Overlap filtering
- Model loading and inference

This is the low-level packing generation logic, separated from the high-level CSP
pipeline orchestration in csp_pipeline.py.

Usage (as a module):
    from molcrystalflow_gen.packing_gen import (
        load_model, extract_molecule_features, run_inference
    )
    
Usage (standalone):
    python packing_gen.py --xyz molecule.xyz --z_values 2 4 --num_samples 100 \\
        --ckpt_path /path/to/checkpoint --output_dir ./output
"""

import argparse
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols
from ase.io import read, write
from numpy.linalg import svd
from omegaconf import OmegaConf
from scipy.spatial import ConvexHull
from torch_geometric.data import Batch, Data

# Add project root and data-preprocess to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DATA_PREPROCESS_DIR = PROJECT_ROOT / "data-preprocess" / "thurlemann23"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DATA_PREPROCESS_DIR))

from data.dataset import ATOM_TYPE_TO_IDX, get_equivariant_axes

# Import auxiliary feature computation from shared preprocessing scripts
try:
    from generate_features import (
        BASIC_FEATURES,
        CHEMICAL_FEATURES,
        GEOMETRIC_FEATURES,
        calculate_geometric_descriptors,
        calculate_rdkit_descriptors,
        extract_features_single_molecule,
        xyz_to_sdf,
    )
    from common import fix_mol_bonds
    _HAS_PREPROCESS_SCRIPTS = True
except ImportError as e:
    _HAS_PREPROCESS_SCRIPTS = False
    # Fallback: define minimal feature lists
    BASIC_FEATURES = ["num_heavy_atoms", "molecular_weight"]
    CHEMICAL_FEATURES = [
        "is_chiral", "num_hbd", "num_hba", "num_rotatable_bonds",
        "num_rings", "num_aromatic_rings", "logp", "tpsa",
    ]
    GEOMETRIC_FEATURES = ["radius_of_gyration", "asphericity", "eccentricity", "planarity"]

# Reverse mapping
IDX_TO_ATOM_TYPE = {v: k for k, v in ATOM_TYPE_TO_IDX.items()}


# ==============================================================================
# Auxiliary Feature Computation
# ==============================================================================


def _xyz_to_sdf_fallback(xyz_path: str, sdf_path: str) -> bool:
    """Convert XYZ file to SDF using Open Babel (fallback implementation)."""
    try:
        cmd = ["obabel", xyz_path, "-O", sdf_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Warning: Open Babel error: {result.stderr}")
            return False
        return True
    except FileNotFoundError:
        print("  Warning: Open Babel (obabel) not found.")
        print("    Install with: conda install -c conda-forge openbabel")
        return False


def _calculate_geometric_descriptors_fallback(positions: np.ndarray) -> Dict[str, float]:
    """Calculate geometric descriptors from atomic positions (fallback)."""
    positions = np.asarray(positions)
    centered_pos = positions - positions.mean(axis=0)
    
    if len(positions) < 3:
        return {
            "radius_of_gyration": 0.0,
            "asphericity": 0.0,
            "eccentricity": 0.0,
            "planarity": 0.0,
        }
    
    cov_matrix = np.dot(centered_pos.T, centered_pos) / len(centered_pos)
    eigenvals = np.linalg.eigvalsh(cov_matrix)
    eigenvals = np.sort(eigenvals)[::-1]

    I1, I2, I3 = eigenvals
    radius_of_gyration = np.sqrt(np.sum(eigenvals))
    asphericity = I1 - 0.5 * (I2 + I3)
    eccentricity = (I1 - I2) / I1 if I1 > 0 else 0.0
    planarity = 1.0 - (I3 / I1) if I1 > 0 else 0.0

    return {
        "radius_of_gyration": float(radius_of_gyration),
        "asphericity": float(asphericity),
        "eccentricity": float(eccentricity),
        "planarity": float(planarity),
    }


def _calculate_rdkit_descriptors_fallback(mol) -> Dict[str, Union[int, float, bool]]:
    """Calculate RDKit molecular descriptors (fallback)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
    except ImportError:
        print("  Warning: RDKit not available, using default features")
        return {}

    if mol is None:
        return {}

    descriptors = {}
    try:
        descriptors["num_atoms"] = mol.GetNumAtoms()
        descriptors["num_heavy_atoms"] = mol.GetNumHeavyAtoms()
        descriptors["num_rings"] = rdMolDescriptors.CalcNumRings(mol)
        descriptors["num_aromatic_rings"] = rdMolDescriptors.CalcNumAromaticRings(mol)
        descriptors["num_hbd"] = rdMolDescriptors.CalcNumHBD(mol)
        descriptors["num_hba"] = rdMolDescriptors.CalcNumHBA(mol)
        descriptors["num_rotatable_bonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
        descriptors["molecular_weight"] = Descriptors.MolWt(mol)
        descriptors["tpsa"] = rdMolDescriptors.CalcTPSA(mol)
        descriptors["logp"] = Crippen.MolLogP(mol)
        descriptors["is_chiral"] = float(
            bool(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        )
    except Exception as e:
        print(f"  Warning: Error computing RDKit descriptors: {e}")

    return descriptors


def _fix_mol_bonds_fallback(mol):
    """Attempt to fix bond orders using rdDetermineBonds (fallback)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
    except ImportError:
        return mol

    if mol is None:
        return None

    try:
        mol_copy = Chem.RWMol(mol)
        # Zero formal charges
        for atom in mol_copy.GetAtoms():
            atom.SetFormalCharge(0)
        rdDetermineBonds.DetermineBonds(mol_copy, charge=0)
        Chem.SanitizeMol(mol_copy)
        return mol_copy.GetMol()
    except Exception:
        return mol


def compute_auxiliary_features(
    atoms: Atoms, 
    temp_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Compute auxiliary molecular features for a single molecule.
    
    Uses shared functions from data-preprocess/thurlemann23/generate_features.py
    if available, otherwise falls back to local implementations.
    
    Args:
        atoms: ASE Atoms object
        temp_dir: Temporary directory for intermediate files
        verbose: Whether to print progress
        
    Returns:
        Dictionary with basic_features, chemical_features, geometric_features tensors
    """
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        RDKIT_AVAILABLE = True
    except ImportError:
        RDKIT_AVAILABLE = False
        if verbose:
            print("  Warning: RDKit not available, using default features")

    # Create temp directory if not provided
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Write XYZ file
    xyz_path = temp_dir / "molecule.xyz"
    write(xyz_path, atoms, format="xyz")

    # Initialize features with defaults
    features = {k: 0.0 for k in BASIC_FEATURES + CHEMICAL_FEATURES + GEOMETRIC_FEATURES}

    # Use shared scripts if available
    if _HAS_PREPROCESS_SCRIPTS and RDKIT_AVAILABLE:
        # Preferred path: use shared generate_features.py functions
        positions = atoms.get_positions()
        sdf_path = temp_dir / "molecule.sdf"
        
        if xyz_to_sdf(str(xyz_path), str(sdf_path)):
            try:
                supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
                mol = next(iter(supplier), None)
                if mol is not None:
                    mol = fix_mol_bonds(mol, smiles=None)
                    # Use the shared extract_features_single_molecule
                    features = extract_features_single_molecule(positions, mol)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Error using shared feature extraction: {e}")
                # Fall through to fallback
                features.update(calculate_geometric_descriptors(positions))
        else:
            # Just compute geometric features if SDF conversion fails
            features.update(calculate_geometric_descriptors(positions))
    else:
        # Fallback path: use local implementations
        # Compute geometric features (no RDKit needed)
        positions = atoms.get_positions()
        features.update(_calculate_geometric_descriptors_fallback(positions))

        # Compute RDKit features if available
        if RDKIT_AVAILABLE:
            sdf_path = temp_dir / "molecule.sdf"
            if _xyz_to_sdf_fallback(str(xyz_path), str(sdf_path)):
                try:
                    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
                    mol = next(iter(supplier), None)
                    if mol is not None:
                        mol = _fix_mol_bonds_fallback(mol)
                        features.update(_calculate_rdkit_descriptors_fallback(mol))
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Error loading SDF: {e}")

    # Create feature tensors
    basic_feats = torch.tensor(
        [float(features.get(k, 0)) for k in BASIC_FEATURES], 
        dtype=torch.float32
    )
    chemical_feats = torch.tensor(
        [float(features.get(k, 0)) for k in CHEMICAL_FEATURES],
        dtype=torch.float32
    )
    geometric_feats = torch.tensor(
        [float(features.get(k, 0)) for k in GEOMETRIC_FEATURES],
        dtype=torch.float32
    )

    return {
        "basic_features": basic_feats,
        "chemical_features": chemical_feats,
        "geometric_features": geometric_feats,
        "raw_features": features,
    }


# ==============================================================================
# Molecule Feature Extraction
# ==============================================================================


def extract_molecule_features(
    atoms: Atoms, 
    compute_aux_features: bool = True,
    verbose: bool = True
) -> dict:
    """
    Extract all features from a molecule conformer for MolCrystalFlow inference.
    
    Args:
        atoms: ASE Atoms object containing the molecule conformer
        compute_aux_features: Whether to compute auxiliary features
        verbose: Whether to print progress
        
    Returns:
        Dictionary with molecule features
    """
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    atom_types = np.array([atomic_numbers[s] for s in symbols])

    # Center the molecule
    centroid = positions.mean(axis=0)
    centered_coords = positions - centroid

    # Compute equivariant axes and local coordinates
    rotmat, eigenvalues, flips = get_equivariant_axes(centered_coords, atom_types)
    local_coords = centered_coords @ rotmat

    # Map atom types to indices
    mapped_atom_types = np.array([ATOM_TYPE_TO_IDX.get(t, -1) for t in atom_types])
    if -1 in mapped_atom_types:
        unknown_types = [t for t in atom_types if t not in ATOM_TYPE_TO_IDX]
        raise ValueError(
            f"Unknown atom types: {unknown_types}. Supported: {list(ATOM_TYPE_TO_IDX.keys())}"
        )

    features = {
        "local_coords": torch.tensor(local_coords, dtype=torch.float32),
        "atom_types": torch.tensor(mapped_atom_types, dtype=torch.int32),
        "eigenvalues": torch.tensor(eigenvalues, dtype=torch.float32),
        "axis_flips": torch.tensor(flips, dtype=torch.float32),
        "num_atoms": len(atoms),
    }

    # Compute auxiliary features
    if compute_aux_features:
        if verbose:
            print("  Computing auxiliary features...")
        aux_features = compute_auxiliary_features(atoms, verbose=verbose)
        features.update(aux_features)
        if verbose:
            print(f"    Basic features: {aux_features['basic_features'].tolist()}")
            print(f"    Chemical features: {aux_features['chemical_features'].tolist()}")
            print(f"    Geometric features: {aux_features['geometric_features'].tolist()}")

    return features


def create_inference_batch(
    mol_features: dict,
    z_value: int,
    has_axis_flip: bool = False,
    axis_flip_state: int = 0,
    device: str = "cuda",
) -> Batch:
    """
    Create a batch for MolCrystalFlow inference from molecule features.
    
    When axis_flip_state=1, the local coordinates are flipped along the x-axis
    (first axis) by multiplying x-coordinates by -1. This corresponds to the
    axis_flips indicator [1, 0, 0].
    """
    n_atoms = mol_features["num_atoms"]

    # Get base local coordinates
    base_local_coords = mol_features["local_coords"].clone()
    
    # Apply axis flip to local coordinates if needed
    # Flip along x-axis (first axis) when axis_flip_state=1
    if has_axis_flip and axis_flip_state == 1:
        base_local_coords[:, 0] = -base_local_coords[:, 0]  # Flip x-coordinates
    
    # Replicate molecule features for Z molecules
    local_coords = base_local_coords.repeat(z_value, 1)
    atom_types = mol_features["atom_types"].repeat(z_value)
    eigenvalues = mol_features["eigenvalues"].unsqueeze(0).repeat(z_value, 1)

    # Handle axis flips indicator
    if has_axis_flip and axis_flip_state == 1:
        axis_flips = torch.tensor([[1.0, 0.0, 0.0]] * z_value, dtype=torch.float32)
    else:
        axis_flips = torch.tensor([[0.0, 0.0, 0.0]] * z_value, dtype=torch.float32)

    # Building block info
    bb_num_vec = torch.tensor([n_atoms] * z_value, dtype=torch.int32)

    # Dummy lattice (will be sampled during inference)
    lattice_1 = torch.eye(3).unsqueeze(0)

    # Dummy translations and rotations
    trans_1 = torch.rand(z_value, 3)
    rotmats_1 = torch.eye(3).unsqueeze(0).repeat(z_value, 1, 1)

    # Auxiliary features (replicated for each molecule)
    basic_features = None
    chemical_features = None
    geometric_features = None
    
    if "basic_features" in mol_features:
        basic_features = mol_features["basic_features"].unsqueeze(0).repeat(z_value, 1)
    if "chemical_features" in mol_features:
        chemical_features = mol_features["chemical_features"].unsqueeze(0).repeat(z_value, 1)
    if "geometric_features" in mol_features:
        geometric_features = mol_features["geometric_features"].unsqueeze(0).repeat(z_value, 1)

    # Create Data object (single structure)
    data = Data(
        num_nodes=z_value,
        num_atoms=n_atoms * z_value,
        num_bbs=z_value,
        rotmats_1=rotmats_1,
        trans_1=trans_1,
        local_coords=local_coords,
        gt_coords=local_coords.clone(),
        bb_num_vec=bb_num_vec,
        atom_types=atom_types,
        lattice_1=lattice_1,
        eigenvalues=eigenvalues,
        axis_flips=axis_flips,
        basic_features=basic_features,
        chemical_features=chemical_features,
        geometric_features=geometric_features,
    )

    # Create a batch from single data point
    batch = Batch.from_data_list([data])
    batch = batch.to(device)

    return batch


# ==============================================================================
# Overlap Filtering Functions
# ==============================================================================


def split_molecules(atoms: Atoms) -> List[Atoms]:
    """Split structure into molecules by bb_indices."""
    if "bb_indices" not in atoms.arrays:
        raise ValueError("XYZ must contain 'bb_indices' per atom.")
    bb_indices = atoms.get_array("bb_indices")
    return [atoms[bb_indices == i] for i in np.unique(bb_indices)]


def hull_volume(atoms: Atoms) -> float:
    """Compute convex-hull volume."""
    coords = atoms.get_positions()
    if len(coords) < 4:
        return 0.0
    try:
        return ConvexHull(coords).volume
    except Exception:
        return 0.0


def fit_ellipsoid(atoms: Atoms) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit ellipsoid using PCA/SVD."""
    coords = atoms.get_positions()
    center = coords.mean(axis=0)
    centered = coords - center
    _, _, vh = svd(centered, full_matrices=False)
    radii = np.std(centered @ vh.T, axis=0)
    return center, radii, vh


def ellipsoid_overlap_fraction(mol1: Atoms, mol2: Atoms) -> float:
    """Ellipsoid overlap approximation."""
    c1, r1, R1 = fit_ellipsoid(mol1)
    c2, r2, R2 = fit_ellipsoid(mol2)
    d_vec = c1 - c2
    d = np.linalg.norm(d_vec)
    if d == 0:
        return 1.0
    n = d_vec / d
    r_eff1 = 1.0 / np.sqrt(np.sum((n @ R1) ** 2 / (r1 ** 2 + 1e-8)))
    r_eff2 = 1.0 / np.sqrt(np.sum((n @ R2) ** 2 / (r2 ** 2 + 1e-8)))
    return np.exp(-(d ** 2) / (2 * (r_eff1 + r_eff2) ** 2))


def analyze_structure(
    atoms: Atoms, threshold: float = 0.05
) -> Tuple[Atoms, float, bool]:
    """Analyze one structure for overlaps."""
    molecules = split_molecules(atoms)
    n = len(molecules)
    volumes = [hull_volume(m) for m in molecules]
    keep = np.ones(n, dtype=bool)
    max_overlap = 0.0

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            frac = ellipsoid_overlap_fraction(molecules[i], molecules[j])
            max_overlap = max(max_overlap, frac)
            if frac > threshold:
                remove = i if volumes[i] < volumes[j] else j
                keep[remove] = False

    kept = [m for k, m in zip(keep, molecules) if k]
    filtered = sum(kept, Atoms())
    filtered.set_cell(atoms.cell)
    filtered.set_pbc(atoms.pbc)

    if "bb_indices" in atoms.arrays:
        merged_bb_indices = np.concatenate([m.get_array("bb_indices") for m in kept])
        filtered.new_array("bb_indices", merged_bb_indices)

    all_preserved = np.all(keep)
    return filtered, max_overlap, all_preserved


def filter_structures_by_overlap(
    structures: List[Atoms], overlap_threshold: float = 0.05
) -> List[Atoms]:
    """Filter structures using overlap volume filtering."""
    filtered = []
    for atoms in structures:
        try:
            _, max_overlap, all_preserved = analyze_structure(
                atoms, threshold=overlap_threshold
            )
            if all_preserved:
                filtered.append(atoms)
        except Exception:
            continue
    return filtered


# ==============================================================================
# Model Loading and Inference
# ==============================================================================


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_model(ckpt_path: str, device: str = "cuda"):
    """Load MolCrystalFlow model from checkpoint."""
    from models.molcrystalflow import FlowModule

    ckpt_dir = (
        Path(ckpt_path).parent if ckpt_path.endswith(".ckpt") else Path(ckpt_path)
    )
    ckpt_file = ckpt_path if ckpt_path.endswith(".ckpt") else None

    # Find checkpoint file
    if ckpt_file is None:
        ckpt_files = list(ckpt_dir.glob("*.ckpt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No .ckpt file found in {ckpt_dir}")
        ckpt_file = str(ckpt_files[0])
        ckpt_dir = ckpt_dir
    else:
        ckpt_dir = Path(ckpt_file).parent

    # Load config
    config_path = ckpt_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    cfg = OmegaConf.load(config_path)

    # Load model
    model = FlowModule.load_from_checkpoint(
        checkpoint_path=ckpt_file, cfg=cfg, map_location="cpu", weights_only=False
    )
    model.eval()
    model.to(device)

    return model, cfg


def run_inference(
    model,
    mol_features: dict,
    z_value: int,
    num_samples: int,
    has_axis_flip: bool = False,
    num_timesteps: int = 50,
    scaling: float = 9.0,
    exp_rate: float = 3.0,
    device: str = "cuda",
    seed: int = 42,
) -> List[Atoms]:
    """
    Run MolCrystalFlow inference for a single Z value.
    
    Args:
        model: Loaded MolCrystalFlow model.
        mol_features: Dictionary with molecule features.
        z_value: Number of molecules in the unit cell.
        num_samples: Number of samples to generate.
        has_axis_flip: Whether to include axis flip states.
        num_timesteps: Number of integration timesteps.
        scaling: Scaling factor for centroid fractional coordinates. Default: 9.0
        exp_rate: Exponential rate for rotation matrices. Default: 3.0
        device: Device to run inference on.
        seed: Random seed.
        
    Returns:
        List of ASE Atoms objects with generated structures.
    """
    set_seed(seed)

    # Prepare for flip states
    if has_axis_flip:
        samples_per_flip = num_samples // 2
        flip_states = [(0, samples_per_flip), (1, num_samples - samples_per_flip)]
    else:
        flip_states = [(0, num_samples)]

    all_structures = []

    for flip_state, n_samples in flip_states:
        if n_samples == 0:
            continue

        # Create batch
        batch = create_inference_batch(
            mol_features,
            z_value,
            has_axis_flip=has_axis_flip,
            axis_flip_state=flip_state,
            device=device,
        )

        # Set number of samples and lattice sampling parameters in model config
        if hasattr(model, "_inference_cfg") and model._inference_cfg is not None:
            model._inference_cfg.num_samples = n_samples
            model._inference_cfg.scaling = scaling
            model._inference_cfg.exp_rate = exp_rate
        else:
            model._infer_cfg = OmegaConf.create({
                "num_samples": n_samples,
                "scaling": scaling,
                "exp_rate": exp_rate,
            })
            model._inference_cfg = model._infer_cfg

        # Run inference
        with torch.inference_mode():
            results = model(
                batch, 
                num_timesteps=num_timesteps,
            )

        # Convert results to ASE Atoms
        structures = results_to_atoms(results, mol_features, z_value)
        all_structures.extend(structures)

    return all_structures


def results_to_atoms(
    results: dict, mol_features: dict, z_value: int
) -> List[Atoms]:
    """Convert model output to ASE Atoms objects."""
    cart_coords = results["cart_coords"].cpu().numpy()
    atom_types = results["atom_types"].cpu().numpy()
    lattices = results["lattices"].cpu().numpy()
    num_atoms = results["num_atoms"].cpu().numpy()

    num_samples = cart_coords.shape[0]
    structures = []

    for i in range(num_samples):
        n_atoms = int(num_atoms[i, 0])
        coords = cart_coords[i, :n_atoms]
        types = atom_types[i, :n_atoms]
        lattice = lattices[i, 0]

        # Convert atom types back to symbols
        symbols = [chemical_symbols[IDX_TO_ATOM_TYPE[int(t)]] for t in types]

        # Create ASE Atoms
        atoms = Atoms(symbols=symbols, positions=coords, cell=lattice, pbc=True)

        # Add building block indices
        n_atoms_per_mol = mol_features["num_atoms"]
        bb_indices = np.repeat(np.arange(z_value), n_atoms_per_mol)
        atoms.new_array("bb_indices", bb_indices)

        structures.append(atoms)

    return structures


# ==============================================================================
# High-Level Generation Function
# ==============================================================================


def generate_crystal_structures(
    xyz_path: str,
    ckpt_path: str,
    z_values: List[int] = [2],
    num_samples: int = 100,
    has_axis_flip: bool = False,
    num_timesteps: int = 50,
    scaling: float = 9.0,
    exp_rate: float = 3.0,
    compute_aux_features: bool = True,
    filter_overlap: bool = False,
    overlap_threshold: float = 0.05,
    output_dir: Optional[str] = None,
    formula: Optional[str] = None,
    device: str = "cuda",
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, any]:
    """
    High-level function to generate crystal structures.
    
    This is the main entry point for the packing generation module.
    
    Args:
        xyz_path: Path to the input molecule XYZ file.
        ckpt_path: Path to the MolCrystalFlow checkpoint.
        z_values: List of Z values to generate.
        num_samples: Number of samples per Z value.
        has_axis_flip: Whether to include axis flip states.
        num_timesteps: Number of integration timesteps.
        scaling: Scaling factor for centroid fractional coordinates during lattice sampling.
        exp_rate: Exponential rate for rotation matrix sampling.
        compute_aux_features: Whether to compute auxiliary features.
        filter_overlap: Whether to filter by overlap.
        overlap_threshold: Overlap threshold for filtering.
        output_dir: Base output directory (structures saved to <output_dir>/<formula>/generated/).
        formula: Override chemical formula (auto-detected if not provided).
        device: Device for inference.
        seed: Random seed.
        verbose: Print progress.
        
    Returns:
        Dictionary with:
        - structures: Dict[int, List[Atoms]] - structures by Z value
        - combined: List[Atoms] - all structures combined
        - formula: str - chemical formula
        - output_dir: Path - output directory
        - output_files: Dict - paths to saved files
    """
    # Load molecule
    if verbose:
        print("=" * 60)
        print("MolCrystalFlow Packing Generation")
        print("=" * 60)
        print("\n[1/5] Loading molecule conformer...")
    
    mol_atoms = read(xyz_path)
    if verbose:
        print(f"  Loaded molecule with {len(mol_atoms)} atoms")
    
    # Get chemical formula
    if formula is None:
        formula = mol_atoms.get_chemical_formula()
    if verbose:
        print(f"  Formula: {formula}")

    # Setup output directory
    output_files = {}
    if output_dir:
        base_output_dir = Path(output_dir)
        formula_dir = base_output_dir / formula
        generated_dir = formula_dir / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy input molecule
        mol_copy_path = formula_dir / "molecule.xyz"
        if not mol_copy_path.exists():
            write(mol_copy_path, mol_atoms, format="xyz")
            if verbose:
                print(f"  Copied input molecule to {mol_copy_path}")
        output_files["molecule"] = mol_copy_path
    else:
        generated_dir = None

    if verbose:
        print(f"\nConfiguration:")
        print(f"  Input XYZ: {xyz_path}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  Z values: {z_values}")
        print(f"  Samples per Z: {num_samples}")
        print(f"  Axis flip: {has_axis_flip}")
        print(f"  Timesteps: {num_timesteps}")
        print(f"  Scaling: {scaling}")
        print(f"  Exp rate: {exp_rate}")
        print(f"  Seed: {seed}")
        print(f"  Aux features: {compute_aux_features}")
        if generated_dir:
            print(f"  Output: {generated_dir}")
        print("=" * 60)

    # Extract features
    if verbose:
        print("\n[2/5] Extracting molecule features...")
    mol_features = extract_molecule_features(
        mol_atoms,
        compute_aux_features=compute_aux_features,
        verbose=verbose,
    )
    if verbose:
        print(f"  Local coords shape: {mol_features['local_coords'].shape}")
        print(f"  Eigenvalues: {mol_features['eigenvalues'].numpy()}")

    # Load model
    if verbose:
        print("\n[3/5] Loading MolCrystalFlow model...")
    device = device if torch.cuda.is_available() else "cpu"
    if device == "cpu" and verbose:
        print("  Warning: CUDA not available, using CPU")
    model, cfg = load_model(ckpt_path, device=device)
    if verbose:
        print(f"  Model loaded on {device}")

    # Run inference
    if verbose:
        print("\n[4/5] Generating crystal structures...")
    all_structures = {}
    all_atoms_combined = []

    for z in z_values:
        if verbose:
            print(f"\n  Z = {z}:")
        z_seed = seed + z * 1000

        structures = run_inference(
            model=model,
            mol_features=mol_features,
            z_value=z,
            num_samples=num_samples,
            has_axis_flip=has_axis_flip,
            num_timesteps=num_timesteps,
            scaling=scaling,
            exp_rate=exp_rate,
            device=device,
            seed=z_seed,
        )
        if verbose:
            print(f"    Generated {len(structures)} structures")

        # Filter if requested
        if filter_overlap:
            if verbose:
                print(f"    Filtering with overlap threshold {overlap_threshold}...")
            structures = filter_structures_by_overlap(structures, overlap_threshold)
            if verbose:
                print(f"    After filtering: {len(structures)} structures")

        all_structures[z] = structures
        all_atoms_combined.extend(structures)

        # Save structures
        if generated_dir:
            output_file = generated_dir / f"crystals_z{z}.xyz"
            write(output_file, structures, format="extxyz")
            output_files[f"z{z}"] = output_file
            if verbose:
                print(f"    Saved to {output_file}")

    # Save combined file
    if verbose:
        print("\n[5/5] Saving combined output...")
    if generated_dir:
        combined_file = generated_dir / "pred_combined.xyz"
        write(combined_file, all_atoms_combined, format="extxyz")
        output_files["combined"] = combined_file
        if verbose:
            print(f"  Saved combined file: {combined_file}")

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  Molecule: {formula}")
        total = sum(len(s) for s in all_structures.values())
        for z, structures in all_structures.items():
            print(f"  Z = {z}: {len(structures)} structures")
        print(f"  Total: {total} structures")
        if generated_dir:
            print(f"\n  Output directory: {generated_dir.parent}/")
            print(f"    - molecule.xyz")
            print(f"    - generated/")
            for z in z_values:
                print(f"        - crystals_z{z}.xyz")
            print(f"        - pred_combined.xyz")
        print("=" * 60)
        print("Done!")

    return {
        "structures": all_structures,
        "combined": all_atoms_combined,
        "formula": formula,
        "output_dir": generated_dir.parent if generated_dir else None,
        "output_files": output_files,
        "mol_features": mol_features,
    }


# ==============================================================================
# CLI Entry Point
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="MolCrystalFlow Packing Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 structures with Z=2 and Z=4
  python packing_gen.py --xyz molecule.xyz --z_values 2 4 --num_samples 100 \\
      --ckpt_path /path/to/checkpoint

  # Generate with axis flip states
  python packing_gen.py --xyz molecule.xyz --z_values 4 --num_samples 200 \\
      --axis_flip --ckpt_path /path/to/checkpoint

  # With overlap filtering
  python packing_gen.py --xyz molecule.xyz --z_values 2 --num_samples 500 \\
      --filter_overlap --overlap_threshold 0.05 --ckpt_path /path/to/checkpoint

Output Structure:
  <output_dir>/<FORMULA>/
  ├── generated/
  │   ├── crystals_z2.xyz
  │   ├── crystals_z4.xyz
  │   └── pred_combined.xyz
  └── molecule.xyz
        """,
    )

    # Required arguments
    parser.add_argument(
        "--xyz",
        "-i",
        type=str,
        required=True,
        help="Path to XYZ file containing the molecule conformer",
    )
    parser.add_argument(
        "--ckpt_path",
        "-c",
        type=str,
        required=True,
        help="Path to MolCrystalFlow checkpoint directory or .ckpt file",
    )

    # Generation parameters
    parser.add_argument(
        "--z_values",
        "-z",
        type=int,
        nargs="+",
        default=[2],
        help="Z values (number of molecules in unit cell). Default: [2]",
    )
    parser.add_argument(
        "--num_samples",
        "-n",
        type=int,
        default=100,
        help="Number of samples to generate per Z value. Default: 100",
    )
    parser.add_argument(
        "--axis_flip",
        action="store_true",
        help="Enable axis flip states (equally distributed between two states)",
    )
    parser.add_argument(
        "--timesteps",
        "-t",
        type=int,
        default=50,
        help="Number of integration timesteps. Default: 50",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42, help="Random seed. Default: 42"
    )

    # Lattice sampling parameters
    parser.add_argument(
        "--scaling",
        type=float,
        default=9.0,
        help="Scaling factor for centroid fractional coordinates. Default: 9.0",
    )
    parser.add_argument(
        "--exp_rate",
        type=float,
        default=3.0,
        help="Exponential rate for rotation matrix sampling. Default: 3.0",
    )

    # Feature computation
    parser.add_argument(
        "--no_aux_features",
        action="store_true",
        help="Skip auxiliary feature computation (basic, chemical, geometric)",
    )

    # Filtering
    parser.add_argument(
        "--filter_overlap",
        action="store_true",
        help="Filter structures by overlap threshold",
    )
    parser.add_argument(
        "--overlap_threshold",
        type=float,
        default=0.05,
        help="Overlap threshold for filtering (0-1). Default: 0.05",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=".",
        help="Base output directory. Outputs go to <output_dir>/<formula>/generated/. Default: current directory",
    )
    parser.add_argument(
        "--formula",
        type=str,
        default=None,
        help="Override chemical formula for output folder naming (auto-detected if not provided)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu). Default: cuda",
    )

    args = parser.parse_args()

    # Run generation
    result = generate_crystal_structures(
        xyz_path=args.xyz,
        ckpt_path=args.ckpt_path,
        z_values=args.z_values,
        num_samples=args.num_samples,
        has_axis_flip=args.axis_flip,
        num_timesteps=args.timesteps,
        scaling=args.scaling,
        exp_rate=args.exp_rate,
        compute_aux_features=not args.no_aux_features,
        filter_overlap=args.filter_overlap,
        overlap_threshold=args.overlap_threshold,
        output_dir=args.output_dir,
        formula=args.formula,
        device=args.device,
        seed=args.seed,
        verbose=True,
    )

    # Print next step hint
    if result["output_files"].get("combined"):
        print(f"\nNext step: Run UMA-OMC relaxation pipeline on {result['output_files']['combined']}")


if __name__ == "__main__":
    main()
