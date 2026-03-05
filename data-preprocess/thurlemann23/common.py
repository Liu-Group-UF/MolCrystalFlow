"""
Shared constants, mappings, and utility functions for the Thurlemann2023
preprocessing pipeline.

This module is the single source of truth for:
- Atom-type mappings (atomic number ↔ model index ↔ element symbol)
- Covalent-radius cutoffs for neighbor detection
- Graph / molecule detection (BFS, connected components)
- PCA / equivariant-axes computation and structure preprocessing
- Bond-fixing helpers (RDKit)
- Tensor / numpy conversion utilities

All other scripts in this package import from here instead of re-defining
these functions locally.
"""
from __future__ import annotations

import gzip
import pickle
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from ase import Atoms
from ase.data import atomic_numbers
from ase.neighborlist import NeighborList

# ===========================================================================
# Atom-type mappings
# ===========================================================================

# Mapping: atomic number → model input index
ATOM_TYPE_TO_IDX: Dict[int, int] = {
    1: 0,    # H
    6: 1,    # C
    7: 2,    # N
    8: 3,    # O
    9: 4,    # F
    16: 5,   # S
    17: 6,   # Cl
    15: 7,   # P
    35: 8,   # Br
    53: 9,   # I
    5: 10,   # B
    14: 11,  # Si
}

# Inverse mapping: model input index → atomic number
IDX_TO_ATOM_TYPE: Dict[int, int] = {v: k for k, v in ATOM_TYPE_TO_IDX.items()}

# Atomic number → element symbol
ATOMIC_NUM_TO_SYMBOL: Dict[int, str] = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
    14: "Si", 15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I",
}

# ===========================================================================
# Covalent radii cutoffs for neighbor / molecule detection (Å)
# ===========================================================================

ELEMENT_CUTOFFS: Dict[str, float] = {
    "H": 0.29,
    "C": 0.70,
    "N": 0.48,
    "O": 0.46,
    "F": 0.68,
    "S": 1.10,
    "Cl": 1.10,
    "P": 1.20,
    "Br": 1.20,
    "I": 1.40,
    "B": 0.85,
    "Si": 1.20,
}

# ===========================================================================
# Tensor / numpy helpers
# ===========================================================================


def to_numpy(value: Any) -> np.ndarray:
    """Convert a value (tensor, array, list, …) to a numpy array."""
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


# ===========================================================================
# Pickle-gz I/O helpers
# ===========================================================================


def load_pkl_gz(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a gzipped pickle file."""
    with gzip.open(str(filepath), "rb") as f:
        return pickle.load(f)


def save_pkl_gz(data: Any, filepath: Union[str, Path]) -> None:
    """Save data to a gzipped pickle file."""
    with gzip.open(str(filepath), "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


# ===========================================================================
# Graph / molecule detection
# ===========================================================================


def find_all_neighbors(
    atoms: Atoms, nl: NeighborList, start_node: int
) -> Dict[int, np.ndarray]:
    """Find all connected atoms using BFS (breadth-first search).

    Args:
        atoms: ASE Atoms object.
        nl: Pre-computed ASE NeighborList.
        start_node: Index of the starting atom.

    Returns:
        Dictionary mapping atom index → cumulative displacement vector.
    """
    visited: set = set()
    queue = deque([(start_node, np.zeros(3))])
    neighbors_group: Dict[int, np.ndarray] = {start_node: np.zeros(3)}

    while queue:
        node, offset = queue.popleft()
        if node in visited:
            continue
        visited.add(node)

        indices, _offsets = nl.get_neighbors(node)
        for neighbor in indices:
            if neighbor in neighbors_group:
                continue
            vec = atoms.get_distances(node, [neighbor], mic=True, vector=True)[0]
            new_offset = offset + vec
            neighbors_group[neighbor] = new_offset
            queue.append((neighbor, new_offset))

    return neighbors_group


def get_molecule_groups(
    atoms: Atoms,
) -> Tuple[List[List[int]], List[np.ndarray]]:
    """
    Identify molecular groups (connected components) in the crystal structure.

    Args:
        atoms: ASE Atoms object.

    Returns:
        molecule_groups: List of lists containing atom indices for each molecule.
        offset_groups: List of arrays containing position offsets for each molecule.
    """
    symbols = atoms.get_chemical_symbols()

    for symbol in set(symbols):
        if symbol not in ELEMENT_CUTOFFS:
            raise ValueError(
                f"Unknown element '{symbol}' – add cutoff to ELEMENT_CUTOFFS in common.py"
            )

    cutoffs = [ELEMENT_CUTOFFS[s] for s in symbols]
    nl = NeighborList(
        cutoffs=np.array(cutoffs), self_interaction=False, bothways=True
    )
    nl.update(atoms)

    molecule_groups: List[List[int]] = []
    offset_groups: List[np.ndarray] = []
    chosen: set = set()

    for i in range(len(atoms)):
        if i in chosen:
            continue
        neighbors_map = find_all_neighbors(atoms, nl, i)
        chosen.update(neighbors_map.keys())
        molecule_groups.append(list(neighbors_map.keys()))
        offset_groups.append(np.array(list(neighbors_map.values())))

    return molecule_groups, offset_groups


# ===========================================================================
# PCA / equivariant axes
# ===========================================================================


def get_pca_axes(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute PCA axes for a set of 3-D points.

    Args:
        data: (N, 3) array.

    Returns:
        eigenvalues:  (3,)  in descending order.
        eigenvectors: (3, 3) columns are the PCA axes.
    """
    data_mean = np.mean(data, axis=0)
    centered = data - data_mean

    cov = np.cov(centered, rowvar=False)
    if cov.ndim == 0:
        return np.zeros(3), np.eye(3)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def get_equiv_vec(cart_coords: np.ndarray, atom_types: np.ndarray) -> np.ndarray:
    """Compute equivariant vector for a building block.

    Args:
        cart_coords: (N, 3) Cartesian coordinates (centred).
        atom_types:  (N,) atomic numbers.

    Returns:
        equiv_vec: (3,) non-zero equivariant vector.
    """
    centroid = np.mean(cart_coords, axis=0)
    weight = atom_types / atom_types.sum()
    weighted_centroid = np.sum(cart_coords * weight[:, None], axis=0)
    equiv_vec = weighted_centroid - centroid

    if np.allclose(equiv_vec, 0):
        dist = np.linalg.norm(cart_coords, axis=1)
        sorted_indices = np.argsort(dist)
        i = 0
        while i < len(sorted_indices) and np.allclose(equiv_vec, 0):
            equiv_vec = cart_coords[sorted_indices[i]]
            i += 1

    assert not np.allclose(equiv_vec, 0), "Equivariant vector is zero"
    return equiv_vec


def get_equivariant_axes(
    cart_coords: np.ndarray, atom_types: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute equivariant rotation matrix for a building block.

    Projects the equivariant vector onto the PCA axes to determine which
    axes need flipping.  The total number of negative projections
    (``n_flips``) decides the final flip pattern:

    * ``n_flips == 0``: no flip  → ``flips = [0, 0, 0]``
    * ``n_flips == 1``: force first-axis flip  → ``flips = [1, 0, 0]``
    * ``n_flips == 2``: ignore flips  → ``flips = [0, 0, 0]``
    * ``n_flips == 3``: flip first axis only  → ``flips = [1, 0, 0]``

    The third axis of the returned frame is always
    ``cross(axis0, axis1)`` (right-handed).

    Args:
        cart_coords: (N, 3) centred Cartesian coordinates.
        atom_types:  (N,) atomic numbers.

    Returns:
        right_hand:  (3, 3) rotation matrix.
        eigenvalues: (3,) PCA eigenvalues.
        flips:       (3,) flip indicator (``[0,0,0]`` or ``[1,0,0]``).
    """
    if cart_coords.shape[0] == 1:
        return np.eye(3), np.zeros(3), np.zeros(3, dtype=int)

    equiv_vec = get_equiv_vec(cart_coords, atom_types)
    eigenvalues, axes = get_pca_axes(cart_coords)
    ve = equiv_vec @ axes
    flips = (ve < 0).astype(int)

    n_flips = flips.sum()
    if n_flips == 2:
        # keep as-is (ignore flips)
        flips[:] = 0
    elif n_flips == 3:
        # only flip first axis
        flips[:] = 0
        flips[0] = 1
    elif n_flips == 1:
        # force flip to first axis
        flips[:] = 0
        flips[0] = 1
    # else n_flips == 0: do nothing

    # Apply final flips
    axes = np.where(flips[None, :], -axes, axes)

    # Build right-handed frame from first two axes
    right_hand = np.stack(
        [axes[:, 0], axes[:, 1], np.cross(axes[:, 0], axes[:, 1])],
        axis=1,
    )

    return right_hand, eigenvalues, flips


# ===========================================================================
# Structure preprocessing
# ===========================================================================


def preprocess_structure(
    lattice_matrix: np.ndarray,
    atom_positions: np.ndarray,
    atom_types: np.ndarray,
    bb_indices: np.ndarray,
) -> Dict[str, torch.Tensor]:
    """
    Preprocess a single crystal structure into training-ready tensors.

    The atoms are first reordered by building-block index (stable sort) so
    that all atoms of bb 0 come first, then bb 1, etc.

    Args:
        lattice_matrix: (3, 3) lattice matrix (Cartesian).
        atom_positions: (N, 3) Cartesian coordinates.
        atom_types:     (N,) atomic numbers.
        bb_indices:     (N,) building-block index per atom.

    Returns:
        Dictionary of training-ready tensors.
    """
    # Reorder atoms by bb_indices (stable sort keeps intra-bb order)
    order = np.argsort(bb_indices, kind="stable")
    atom_positions = atom_positions[order]
    atom_types = atom_types[order]
    bb_indices = bb_indices[order]

    cell_matrix = lattice_matrix
    cart_coords = atom_positions

    unique_bbs = np.unique(bb_indices)
    gt_trans: List[np.ndarray] = []
    centered_bb_coords: List[np.ndarray] = []
    bb_num_vec: List[int] = []

    for bb_idx in unique_bbs:
        mask = bb_indices == bb_idx
        bb_coords = cart_coords[mask]
        bb_center = bb_coords.mean(axis=0)
        gt_trans.append(bb_center)
        centered_bb_coords.append(bb_coords - bb_center)
        bb_num_vec.append(int(np.sum(mask)))

    gt_trans_arr = np.array(gt_trans)

    gt_coords = np.concatenate(
        [coords + trans for coords, trans in zip(centered_bb_coords, gt_trans_arr)],
        axis=0,
    )

    gt_rotmats: List[np.ndarray] = []
    local_coords_list: List[np.ndarray] = []
    gt_axis_flips: List[np.ndarray] = []
    gt_eigenvalues: List[np.ndarray] = []

    for i, bb_coord in enumerate(centered_bb_coords):
        mask = bb_indices == unique_bbs[i]
        bb_atom_type = atom_types[mask]

        rotmats, eigenvalues, flips = get_equivariant_axes(bb_coord, bb_atom_type)
        local_coord = bb_coord @ rotmats

        gt_rotmats.append(rotmats)
        local_coords_list.append(local_coord)
        gt_axis_flips.append(np.array(flips, dtype=float))
        gt_eigenvalues.append(np.array(eigenvalues))

    gt_rotmats_arr = np.stack(gt_rotmats, axis=0)
    local_coords_arr = np.concatenate(local_coords_list, axis=0)
    bb_num_vec_arr = np.array(bb_num_vec)

    res_mask = np.ones_like(bb_num_vec_arr)
    diffuse_mask = np.ones_like(bb_num_vec_arr)

    frac_trans = gt_trans_arr @ np.linalg.inv(cell_matrix)
    frac_trans = frac_trans - np.floor(frac_trans)

    mapped_atom_types = np.array([ATOM_TYPE_TO_IDX.get(t, -1) for t in atom_types])

    feats: Dict[str, torch.Tensor] = {
        "bb_num_vec": torch.tensor(bb_num_vec_arr).int(),
        "trans_1": torch.tensor(frac_trans).float(),
        "rotmats_1": torch.tensor(gt_rotmats_arr).float(),
        "atom_types": torch.tensor(mapped_atom_types).int(),
        "local_coords": torch.tensor(local_coords_arr).float(),
        "gt_coords": torch.tensor(gt_coords).float(),
        "eigenvalues": torch.tensor(gt_eigenvalues).float(),
        "lattice_1": torch.tensor(lattice_matrix).float(),
        "axis_flips": torch.tensor(gt_axis_flips).float(),
        "res_mask": torch.tensor(res_mask).int(),
        "diffuse_mask": torch.tensor(diffuse_mask).int(),
    }

    return feats


# ===========================================================================
# Bond-fixing helpers (RDKit)
# ===========================================================================

try:
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    from rdkit.Chem.rdchem import BondType

    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False


def _require_rdkit() -> None:
    if not _HAS_RDKIT:
        raise ImportError(
            "RDKit is required for bond-fixing utilities. "
            "Install with: conda install -c conda-forge rdkit"
        )


def zero_formal_charges(mol: "Chem.Mol") -> "Chem.Mol":
    """Set all formal charges to zero."""
    _require_rdkit()
    mol = Chem.RWMol(mol)
    for atom in mol.GetAtoms():
        atom.SetFormalCharge(0)
    return mol.GetMol()


def fix_nitrogen_valency(mol: "Chem.Mol") -> "Chem.Mol":
    """Fix nitrogen valency issues (single → double bond when under-valent)."""
    _require_rdkit()
    mol = Chem.RWMol(mol)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 7:
            valence = atom.GetTotalValence()
            if valence < 3:
                for bond in atom.GetBonds():
                    if bond.GetBondType() == BondType.SINGLE:
                        bond.SetBondType(BondType.DOUBLE)
                        break
    return mol.GetMol()


def fix_carbon_valency(mol: "Chem.Mol") -> "Chem.Mol":
    """Fix carbon valency issues (single → double bond when under-valent)."""
    _require_rdkit()
    mol = Chem.RWMol(mol)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:
            valence = atom.GetTotalValence()
            if valence < 4:
                for bond in atom.GetBonds():
                    if bond.GetBondType() == BondType.SINGLE:
                        neighbor = bond.GetOtherAtom(atom)
                        if neighbor.GetAtomicNum() in [6, 7]:
                            bond.SetBondType(BondType.DOUBLE)
                            break
    return mol.GetMol()


def fix_valency(mol: "Chem.Mol") -> "Chem.Mol":
    """Apply both nitrogen and carbon valency fixes."""
    mol = fix_nitrogen_valency(mol)
    mol = fix_carbon_valency(mol)
    return mol


def adjust_formal_charges_neutralize(
    mol: "Chem.Mol",
) -> Tuple["Chem.Mol", bool]:
    """Adjust formal charges to neutralize the molecule.

    Returns:
        (fixed_mol, was_modified)
    """
    _require_rdkit()
    mol = Chem.RWMol(mol)
    modified = False

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        charge = atom.GetFormalCharge()
        valence = atom.GetTotalValence()

        if symbol == "N" and charge == 1 and valence == 4:
            pass  # quaternary N is OK
        elif symbol == "N" and charge == 0 and valence == 4:
            atom.SetFormalCharge(1)
            modified = True
        elif symbol == "O" and charge == -1 and valence == 1:
            pass  # carboxylate is OK
        elif symbol == "S" and charge == 0 and valence == 6:
            pass  # sulfone is OK

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass

    return mol.GetMol(), modified


def fix_mol_bonds(mol: "Chem.Mol", smiles: str = None) -> "Chem.Mol":
    """Attempt to fix bond orders using multiple strategies.

    Args:
        mol: RDKit Mol object with potentially incorrect bonds.
        smiles: Optional SMILES string for reference.

    Returns:
        Fixed Mol object, or original if fixing fails.
    """
    _require_rdkit
    if mol is None:
        return None

    # Strategy 1: Use rdDetermineBonds
    try:
        mol_copy = Chem.RWMol(mol)
        mol_copy = zero_formal_charges(mol_copy)
        rdDetermineBonds.DetermineBonds(mol_copy, charge=0)
        fixed_mol, _ = adjust_formal_charges_neutralize(mol_copy)
        Chem.SanitizeMol(fixed_mol)
        return fixed_mol
    except Exception:
        pass

    # Strategy 2: Fix valency issues manually
    try:
        mol_copy = Chem.RWMol(mol)
        mol_copy = zero_formal_charges(mol_copy)
        mol_copy = fix_nitrogen_valency(mol_copy)
        mol_copy = fix_carbon_valency(mol_copy)
        fixed_mol, _ = adjust_formal_charges_neutralize(mol_copy)
        Chem.SanitizeMol(fixed_mol)
        return fixed_mol
    except Exception:
        pass

    # Strategy 3: Use InChI round-trip if SMILES available
    if smiles is not None:
        try:
            ref_mol = Chem.MolFromSmiles(smiles)
            if ref_mol is not None:
                Chem.RemoveStereochemistry(ref_mol)
                inchi = Chem.MolToInchi(ref_mol)
                inchi_mol = Chem.MolFromInchi(inchi)
                if inchi_mol is not None:
                    inchi_mol = Chem.AddHs(inchi_mol)
                    Chem.SanitizeMol(inchi_mol)
                    return inchi_mol
        except Exception:
            pass

    # Return original if all strategies fail
    return mol


# ===========================================================================
# Coordinate reconstruction (for validation / check_data)
# ===========================================================================


def assemble_coords(sample: Dict[str, Any]) -> np.ndarray:
    """Reassemble Cartesian coordinates from decomposed representation.

    Reconstruction formula (matches ``data/interpolant.py:_assemble_coords``):
        ``cart_trans = frac_trans @ lattice``  →  (M, 3)
        For each building block *i*:
            ``global_coords_i = rotmats_i @ local_coords_i + cart_trans_i``

    Args:
        sample: Data sample dict with keys ``local_coords``, ``rotmats_1``,
                ``trans_1``, ``lattice_1``, ``bb_num_vec``.

    Returns:
        Reconstructed Cartesian coordinates (N, 3).
    """
    local_coords = to_numpy(sample["local_coords"])   # (N, 3)
    rotmats = to_numpy(sample["rotmats_1"])            # (M, 3, 3)
    frac_trans = to_numpy(sample["trans_1"])            # (M, 3)
    lattice = to_numpy(sample["lattice_1"])             # (3, 3)
    bb_num_vec = to_numpy(sample["bb_num_vec"])         # (M,)

    cart_trans = frac_trans @ lattice                   # (M, 3)

    splits = np.cumsum(bb_num_vec.astype(int))[:-1]
    local_chunks = np.split(local_coords, splits)

    assembled = []
    for R, t, lc in zip(rotmats, cart_trans, local_chunks):
        global_chunk = (R @ lc.T).T + t
        assembled.append(global_chunk)

    return np.concatenate(assembled, axis=0)


# ===========================================================================
# Normalization helpers (for normalize_data)
# ===========================================================================


def rotation_180_about_axis(axis: int) -> "torch.Tensor":
    """Return 180° rotation matrix about x/y/z axis.

    Args:
        axis: 0 for x, 1 for y, 2 for z.

    Returns:
        3×3 rotation matrix as a torch tensor.
    """
    if axis == 0:
        return torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32))
    elif axis == 1:
        return torch.diag(torch.tensor([-1.0, 1.0, -1.0], dtype=torch.float32))
    elif axis == 2:
        return torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float32))
    else:
        raise ValueError("Axis must be 0, 1, or 2")


def renormalize_single_datapoint(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize local coordinates and rotation matrices for one structure.

    Ensures consistent local coordinate frames across building blocks:
    1.  Group blocks by their axis-flip pattern.
    2.  For each group, use the first block as reference.
    3.  If another block has a 2-axis flip relative to the reference,
        rotate 180° about the remaining axis.

    Args:
        data: Dictionary containing structure data.

    Returns:
        Normalized data dictionary (modified in-place).
    """
    if isinstance(data["local_coords"], torch.Tensor):
        coords = data["local_coords"].clone().float()
    else:
        coords = torch.tensor(data["local_coords"], dtype=torch.float32)

    if isinstance(data["axis_flips"], torch.Tensor):
        flips = data["axis_flips"].clone().float()
    else:
        flips = torch.tensor(data["axis_flips"], dtype=torch.float32)

    if isinstance(data["rotmats_1"], torch.Tensor):
        rotmats = data["rotmats_1"].clone().float()
    else:
        rotmats = torch.tensor(data["rotmats_1"], dtype=torch.float32)

    if isinstance(data["bb_num_vec"], torch.Tensor):
        bb_sizes = data["bb_num_vec"].tolist()
    else:
        bb_sizes = list(data["bb_num_vec"])

    corrected_coords = coords.clone()
    corrected_rots = rotmats.clone()

    valid_flips = [
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1.0, 0.0, 0.0]),
    ]
    n_blocks = len(bb_sizes)

    for uf in valid_flips:
        indices = [b for b in range(n_blocks) if torch.equal(flips[b], uf)]

        if len(indices) <= 1:
            continue

        ref_b = indices[0]
        ref_start = sum(bb_sizes[:ref_b])
        ref_atom = corrected_coords[ref_start]

        for b in indices[1:]:
            b_start = sum(bb_sizes[:b])
            rep_atom = corrected_coords[b_start]

            flip_axes = (rep_atom * ref_atom) < 0
            n_axis_flips = flip_axes.sum().item()

            if n_axis_flips == 2:
                axis_to_rotate = (~flip_axes).nonzero(as_tuple=False).item()
                R = rotation_180_about_axis(axis_to_rotate)

                block_end = b_start + bb_sizes[b]
                corrected_coords[b_start:block_end] = (
                    corrected_coords[b_start:block_end] @ R.T
                )
                corrected_rots[b] = corrected_rots[b] @ R.T

    data["local_coords"] = corrected_coords
    data["rotmats_1"] = corrected_rots

    return data
