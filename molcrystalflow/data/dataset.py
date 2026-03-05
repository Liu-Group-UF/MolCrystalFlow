import gzip
import logging
import os
import pickle

import numpy as np
import torch
from ase.data import atomic_numbers
from torch.utils.data import Dataset
from torch_geometric.data import Data

# Mapping from atomic number to index for model input
ATOM_TYPE_TO_IDX = {
    1: 0,   # H
    6: 1,   # C
    7: 2,   # N
    8: 3,   # O
    9: 4,   # F
    16: 5,  # S
    17: 6,  # Cl
    15: 7,  # P
    35: 8,  # Br
    53: 9,  # I
    5: 10,  # B
    14: 11, # Si
}


def get_pca_axes(data):
    """
    Compute PCA axes for a set of points.
    
    Args:
        data: numpy array of shape (N, 3) with Cartesian coordinates

    Returns:
        eigenvalues: numpy array of shape (3,)
        eigenvectors: numpy array of shape (3, 3)
    """
    # Center the data
    data_mean = np.mean(data, axis=0)
    centered_data = data - data_mean

    # Compute the covariance matrix
    covariance_matrix = np.cov(centered_data, rowvar=False)
    if covariance_matrix.ndim == 0:
        return np.zeros(3), np.eye(3)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    return eigenvalues, eigenvectors

def get_equiv_vec(cart_coords, atom_types):
    """
    Compute equivariant vector for a building block.
    
    Args:
        cart_coords: numpy array of shape (N, 3) with Cartesian coordinates
        atom_types: numpy array of shape (N,) with atomic numbers
        
    Returns:
        equiv_vec: numpy array of shape (3,) representing equivariant vector
    """
    centroid = np.mean(cart_coords, axis=0)

    # Center of mass weighted by atomic number
    weight = atom_types / atom_types.sum()
    weighted_centroid = np.sum(cart_coords * weight[:, None], axis=0)

    # Equivariant vector
    equiv_vec = weighted_centroid - centroid

    # If v = 0 (symmetric), take the closest non-zero atom
    if np.allclose(equiv_vec, 0):
        dist = np.linalg.norm(cart_coords, axis=1)
        sorted_indices = np.argsort(dist)

        i = 0
        while i < len(sorted_indices) and np.allclose(equiv_vec, 0):
            equiv_vec = cart_coords[sorted_indices[i]]
            i += 1
    
    assert not np.allclose(equiv_vec, 0), "Equivariant vector is zero"
    return equiv_vec

def get_equivariant_axes(cart_coords, atom_types):
    """
    Compute equivariant rotation matrix for a building block.

    Flips PCA axes so their projections align with the equivariant vector,
    then builds a right-handed frame from the first two axes. The canonical
    flip indicator is [0,0,0] if cross(axis0, axis1) aligns with equiv_vec,
    or [1,0,0] otherwise.
    
    Args:
        cart_coords: numpy array of shape (N, 3) with Cartesian coordinates
        atom_types: numpy array of shape (N,) with atomic numbers
        
    Returns:
        right_hand: numpy array of shape (3, 3) representing rotation matrix
        eigenvalues: numpy array of shape (3,) with PCA eigenvalues
        flips: numpy array of shape (3,) indicating axis flips
    """
    if cart_coords.shape[0] == 1:
        return np.eye(3), np.zeros(3), np.zeros(3, dtype=int)

    equiv_vec = get_equiv_vec(cart_coords, atom_types)
    eigenvalues, axes = get_pca_axes(cart_coords)
    ve = equiv_vec @ axes
    flips = (ve < 0).astype(int)

    # Flip PCA axes to align with equiv_vec
    axes = np.where(flips[None, :], -axes, axes)

    # Build right-handed frame from first two flipped axes
    right_hand = np.stack(
        [axes[:, 0], axes[:, 1], np.cross(axes[:, 0], axes[:, 1])],
        axis=1
    )

    # Canonical flip: does cross(axis0, axis1) align with equiv_vec?
    cross_axis = np.cross(axes[:, 0], axes[:, 1])
    if cross_axis @ equiv_vec > 0:
        flips = np.array([0, 0, 0], dtype=int)
    else:
        flips = np.array([1, 0, 0], dtype=int)

    return right_hand, eigenvalues, flips

def preprocess_structure(lattice_matrix, atom_positions, atom_types, bb_indices):
    """
    Preprocess a single structure with cell parameters, atom positions, atom types, and building block indices.

    Args:
        lattice_matrix: 3x3 lattice matrix (Cartesian).
        atom_positions: numpy array of shape (N, 3) with Cartesian coordinates of atoms.
        atom_types: numpy array of shape (N,) with atomic numbers of atoms.
        bb_indices: numpy array of shape (N,) with building block indices for each atom.

    Returns:
        feats: Dictionary containing processed features for the structure.
    """
    # Reorder atoms by bb_indices so all atoms of bb 0 come first, then bb 1, etc.
    order = np.argsort(bb_indices, kind="stable")
    atom_positions = atom_positions[order]
    atom_types = atom_types[order]
    bb_indices = bb_indices[order]

    # Convert cell parameters to lattice matrix (already provided as lattice_matrix)
    cell_matrix = lattice_matrix

    # Use given atom_positions as Cartesian coordinates
    cart_coords = atom_positions

    # Build per-building-block data
    unique_bbs = np.unique(bb_indices)
    gt_trans = []
    centered_bb_coords = []
    bb_num_vec = []

    for bb_idx in unique_bbs:
        mask = bb_indices == bb_idx
        bb_coords = cart_coords[mask]

        # Center the building block
        bb_center = bb_coords.mean(axis=0)
        gt_trans.append(bb_center)
        centered_bb_coords.append(bb_coords - bb_center)
        # Store the number of atoms in the building block
        bb_num_vec.append(np.sum(mask))

    gt_trans = np.array(gt_trans)

    # Compute global coordinates (centered blocks plus their translations)
    gt_coords = np.concatenate(
        [coords + trans for coords, trans in zip(centered_bb_coords, gt_trans)],
        axis=0
    )

    # Compute rotation matrices and local coordinates for each building block
    gt_rotmats = []
    local_coords = []
    gt_axis_flips = []
    gt_eigenvalues = []
    for i, bb_coord in enumerate(centered_bb_coords):
        mask = bb_indices == unique_bbs[i]
        bb_atom_type = atom_types[mask]

        rotmats, eigenvalues, flips = get_equivariant_axes(bb_coord, bb_atom_type)
        local_coord = bb_coord @ rotmats

        gt_rotmats.append(rotmats)
        local_coords.append(local_coord)
        gt_axis_flips.append(np.array(flips, dtype=float))
        gt_eigenvalues.append(np.array(eigenvalues))

    gt_rotmats = np.stack(gt_rotmats, axis=0)
    local_coords = np.concatenate(local_coords, axis=0)
    bb_num_vec = np.array(bb_num_vec)

    res_mask = np.ones_like(bb_num_vec)
    diffuse_mask = np.ones_like(bb_num_vec)

    frac_trans = gt_trans @ np.linalg.inv(cell_matrix)
    frac_trans = frac_trans - np.floor(frac_trans)

    mapped_atom_types = np.array([ATOM_TYPE_TO_IDX.get(t, -1) for t in atom_types])

    feats = {
        'bb_num_vec': torch.tensor(bb_num_vec).int(),
        'trans_1': torch.tensor(frac_trans).float(),
        'rotmats_1': torch.tensor(gt_rotmats).float(),
        'atom_types': torch.tensor(mapped_atom_types).int(),
        'local_coords': torch.tensor(local_coords).float(),
        'gt_coords': torch.tensor(gt_coords).float(),
        'eigenvalues': torch.tensor(gt_eigenvalues).float(),
        'lattice_1': torch.tensor(lattice_matrix).float(),
        'axis_flips': torch.tensor(gt_axis_flips).float(),
        'res_mask': torch.tensor(res_mask).int(),
        'diffuse_mask': torch.tensor(diffuse_mask).int(),
    }

    return feats


def preprocess_all_structures(xyz_file, output_file, pkl_file=None, output_xyz=None,
                              existing_inds=None, replicas=None):
    """
    Preprocess all structures in an XYZ file and save them into a compressed pickle file.

    Args:
        xyz_file: Path to the XYZ file containing multiple structures.
        output_file: Path to save the processed data (.pkl.gz).
        pkl_file: Optional path to pickle file with extra features.
        output_xyz: Optional path to save output XYZ file.
        existing_inds: Optional indices for existing data.
        replicas: Optional replicas array to handle augmented dataset.
    """
    processed_data = []

    with open(xyz_file, 'r') as f:
        lines = f.readlines()

    if pkl_file is not None:
        with open(pkl_file, 'rb') as pf:
            extra_features = pickle.load(pf)
        basic_feats = extra_features['basic_features']
        chemical_feats = extra_features['chemical_features']
        geometric_feats = extra_features['geometric_features']
        num_bbs = np.int_(extra_features['num_bbs'])
        split_indices = np.cumsum(num_bbs)[:-1]
        full_basic_chunks = np.split(basic_feats, split_indices)
        full_chemical_chunks = np.split(chemical_feats, split_indices)
        full_geometric_chunks = np.split(geometric_feats, split_indices)

    i = 0
    while i < len(lines):
        try:
            # Read number of atoms
            n_atoms = int(lines[i].strip())
            comment = lines[i + 1].strip()

            # Extract lattice matrix from the comment line
            lattice_str = comment.split('Lattice="')[1].split('"')[0]
            lattice_matrix = np.array([float(x) for x in lattice_str.split()]).reshape(3, 3)

            # Read atomic positions and elements
            atom_positions = []
            atom_elements = []
            bb_indices = []
            for j in range(n_atoms):
                parts = lines[i + 2 + j].strip().split()
                atom_elements.append(parts[0])  # Element symbol
                atom_positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                bb_indices.append(int(parts[4]))  # Building block index

            atom_positions = np.array(atom_positions)
            bb_indices = np.array(bb_indices)

            # Convert element symbols to atomic numbers using ASE
            atom_types = np.array([atomic_numbers[element] for element in atom_elements])

            # Preprocess the structure
            feats = preprocess_structure(lattice_matrix, atom_positions, atom_types, bb_indices)
            processed_data.append(feats)

            # Move to the next structure
            i += n_atoms + 2

        except Exception as e:
            print(f"Error processing structure at line {i}: {e}")
            break

    if output_xyz is None:
        output_xyz = output_file.replace(".pkl.gz", ".xyz")

    # Add extra features if provided
    indices_to_delete = []
    if existing_inds is not None and pkl_file is not None:
        for local_idx, global_idx in enumerate(existing_inds):
            basic_tensor = torch.tensor(full_basic_chunks[global_idx]).float()
            chemical_tensor = torch.tensor(full_chemical_chunks[global_idx]).float()
            geometric_tensor = torch.tensor(full_geometric_chunks[global_idx]).float()

            # Expand tensors according to replicas if provided
            if replicas is not None:
                n_rep = replicas[local_idx]
                basic_tensor = basic_tensor.repeat(n_rep, 1)
                chemical_tensor = chemical_tensor.repeat(n_rep, 1)
                geometric_tensor = geometric_tensor.repeat(n_rep, 1)
            try:
                assert basic_tensor.shape[0] == processed_data[local_idx]['trans_1'].shape[0]
                processed_data[local_idx].update({
                    "basic_features": basic_tensor,
                    "chemical_features": chemical_tensor,
                    "geometric_features": geometric_tensor,
                })
            except AssertionError as e:
                print(f"Assertion error at local_idx {local_idx}, global_idx {global_idx}: {e}")
                indices_to_delete.append(local_idx)
                print(f"Processed data shapes: {basic_tensor.shape}, {processed_data[local_idx]['trans_1'].shape[0]}")
                break

    # Delete problematic data points in reverse order
    for idx in sorted(indices_to_delete, reverse=True):
        del processed_data[idx]

    # Save processed data to a compressed pickle file
    print(f"Saving {len(processed_data)} processed structures to {output_file}")
    with gzip.open(output_file, 'wb') as f:
        pickle.dump(processed_data, f, protocol=pickle.HIGHEST_PROTOCOL)


class MCDataset(Dataset):
    def __init__(
            self,
            *,
            cache_path,
            dataset_cfg,
            is_training,
        ):

        self._log = logging.getLogger(__name__)
        self._cache_path = cache_path
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg

        # Check if processed data exists
        processed_path = self._cache_path.replace('.pt', '_molcrystal_normalized.pkl.gz')
        if os.path.exists(processed_path):
            print(f"INFO:: Loading processed data from {processed_path}")
            with gzip.open(processed_path, 'rb') as f:
                self.processed_data = pickle.load(f)
        else:
            print(f"INFO:: No processed data found at {processed_path}")
            raise FileNotFoundError(
                f"Processed data not found at {processed_path}. "
                "Please run preprocess_all_structures() first."
            )

    @property
    def dataset_cfg(self):
        return self._dataset_cfg

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        """
        Returns: Data object with following keys:
            - rotmats_1: [M, 3, 3]
            - trans_1: [M, 3]
            - local_coords: [N, 3]
            - gt_coords: [N, 3]
            - bb_num_vec: [M,]
            - atom_types: [N,]
            - lattice_1: [1, 3, 3]
            - eigenvalues: [M, 3]
            - axis_flips: [M, 3]
        """
        data = self.processed_data[idx]
        return Data(
            num_nodes=data['rotmats_1'].shape[0],
            num_atoms=data['local_coords'].shape[0],
            num_bbs=data['rotmats_1'].shape[0],
            rotmats_1=data['rotmats_1'],
            trans_1=data['trans_1'],
            local_coords=data['local_coords'],
            gt_coords=data['gt_coords'],
            bb_num_vec=data['bb_num_vec'],
            atom_types=data['atom_types'],
            lattice_1=data['lattice_1'].unsqueeze(0),
            eigenvalues=data['eigenvalues'],
            axis_flips=data['axis_flips'],
            basic_features=data.get('basic_features'),
            chemical_features=data.get('chemical_features'),
            geometric_features=data.get('geometric_features'),
        )