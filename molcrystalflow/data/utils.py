import torch
import numpy as np
from pathlib import Path
from molcrystalflow.openfold.utils import rigid_utils as ru
from ase.data import chemical_symbols
from torch import Tensor

# Global map from chain characters to integers.
NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE

# Define atomic number to compact index mapping
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
    # Add more elements as needed
}

# Define the reverse mapping: compact index to atomic number
IDX_TO_ATOM_TYPE = {v: k for k, v in ATOM_TYPE_TO_IDX.items()}
DEFAULT_ATOM_TYPE = 1  # Default to H if unknown index

def to_numpy(x):
    return x.detach().cpu().numpy()

def create_rigid(rots, trans):
    rots = ru.Rotation(rot_mats=rots)
    return ru.Rigid(rots=rots, trans=trans)

def split_data_with_bb_indices(cart_coords, atom_types, lattices, num_atoms, num_bbs_per_struct):
    """
    Splits concatenated data into a list of samples with building block indices.
    Args:
        cart_coords (torch.Tensor): [num_atoms, 3]
        atom_types (torch.Tensor): [num_atoms]
        lattices (torch.Tensor): [num_samples, 3, 3]
        num_atoms (list): [num_samples] each element is the number of atoms in a sample
        num_bbs_per_struct (list): [num_samples] each element is the number of building blocks in a sample
    Returns:
        split_list (list): [num_samples] each element is a dict with keys 'cart_coords', 'atom_types', 'lattice', 'bb_indices'
    """
    split_list = []
    
    cart_coords_split = cart_coords.split(num_atoms)
    atom_types_split = atom_types.split(num_atoms)
    
    for coords, atoms, lattice, num_bbs in zip(cart_coords_split, atom_types_split, lattices, num_bbs_per_struct):
        # Create building block indices for this structure
        # Assume atoms are grouped by building blocks and evenly distributed
        num_atoms_in_struct = len(coords)
        atoms_per_bb = num_atoms_in_struct // num_bbs if num_bbs > 0 else num_atoms_in_struct
        
        bb_indices = []
        for bb_idx in range(num_bbs):
            start_atom = bb_idx * atoms_per_bb
            end_atom = (bb_idx + 1) * atoms_per_bb if bb_idx < num_bbs - 1 else num_atoms_in_struct
            bb_indices.extend([bb_idx] * (end_atom - start_atom))
        
        bb_indices_tensor = torch.tensor(bb_indices, dtype=torch.long)
        
        split_list.append({
            'cart_coords': coords,
            'atom_types': atoms,
            'lattice': lattice,
            'bb_indices': bb_indices_tensor
        })
    
    return split_list


def split_data(cart_coords, atom_types, lattices, num_atoms):
    """
    Splits concatenated data into a list of samples.
    
    Args:
        cart_coords (torch.Tensor): [total_atoms, 3]
        atom_types (torch.Tensor): [total_atoms]
        lattices (torch.Tensor): [num_samples, 3, 3]
        num_atoms (list): [num_samples] each element is the number of atoms in a sample
        
    Returns:
        split_list (list): [num_samples] each element is a dict with keys 
                          'cart_coords', 'atom_types', 'lattice'
    """
    split_list = []
    
    cart_coords_split = cart_coords.split(num_atoms)
    atom_types_split = atom_types.split(num_atoms)
    
    for coords, atoms, lattice in zip(cart_coords_split, atom_types_split, lattices):
        split_list.append({
            'cart_coords': coords,
            'atom_types': atoms,
            'lattice': lattice,
        })
    
    return split_list


def save_combined_xyz(structures, filename, structure_type="", use_compact_indices=False):
    """Save multiple structures into a single extended XYZ file with lattice information.
    
    Format follows ASE's extended XYZ:
    - Line 1: Number of atoms
    - Line 2: Lattice="a11 a12 a13 a21 a22 a23 a31 a32 a33" Properties=species:S:1:pos:R:3:bb_idx:I:1 pbc="T T T"
    - Following lines: atom_type x y z bb_idx
    
    Args:
        structures: List of structure dictionaries
        filename: Path to save the XYZ file
        structure_type: Optional label for each structure
        use_compact_indices: If True, assumes atom_types are compact indices that need to be converted
    """
    with open(filename, 'w') as f:
        for idx, struct in enumerate(structures):
            # Convert tensors to numpy
            atomic_numbers = to_numpy(struct['atom_types'])
            cart_coords = to_numpy(struct['cart_coords'])
            lattice = to_numpy(struct['lattice'])  # Now expects [3, 3] matrix
            
            # Convert compact indices to atomic numbers if needed
            if use_compact_indices:
                # Convert compact indices to atomic numbers
                real_atomic_numbers = np.zeros_like(atomic_numbers)
                for i, idx in enumerate(atomic_numbers):
                    real_atomic_numbers[i] = IDX_TO_ATOM_TYPE.get(int(idx), DEFAULT_ATOM_TYPE)
                atom_types = [chemical_symbols[int(z)] for z in real_atomic_numbers]
            else:
                atom_types = [chemical_symbols[int(z)] for z in atomic_numbers]
            
            # Get building block indices if available
            bb_indices = None
            if 'bb_indices' in struct:
                bb_indices = to_numpy(struct['bb_indices'])

            # lattice is already a 3x3 matrix
            if lattice.shape == (3, 3):
                v = lattice  # Use matrix directly
            else:
                # Fallback: assume it's 6-parameter format and convert
                a, b, c = lattice[0:3]
                alpha, beta, gamma = lattice[3:6]
                # Convert angles to radians
                alpha = np.radians(alpha)
                beta = np.radians(beta)
                gamma = np.radians(gamma)
                
                # Compute lattice vectors
                v = np.zeros((3, 3))
                v[0][0] = a
                v[1][0] = b * np.cos(gamma)
                v[1][1] = b * np.sin(gamma)
                v[2][0] = c * np.cos(beta)
                v[2][1] = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
                v[2][2] = np.sqrt(c * c - v[2][0] * v[2][0] - v[2][1] * v[2][1])
            
            num_atoms = len(atom_types)
            f.write(f"{num_atoms}\n")
            
            # Write extended XYZ comment line with or without bb_idx property
            if bb_indices is not None:
                properties_str = "Properties=species:S:1:pos:R:3:bb_idx:I:1"
            else:
                properties_str = "Properties=species:S:1:pos:R:3"
            
            f.write(f"{structure_type} {idx} Lattice=\"{v[0][0]:.6f} {v[0][1]:.6f} {v[0][2]:.6f} "
                   f"{v[1][0]:.6f} {v[1][1]:.6f} {v[1][2]:.6f} "
                   f"{v[2][0]:.6f} {v[2][1]:.6f} {v[2][2]:.6f}\" "
                   f"{properties_str} pbc=\"T T T\"\n")
            
            # Write atomic positions with element symbols and optionally building block indices
            for i, (element, pos) in enumerate(zip(atom_types, cart_coords)):
                if bb_indices is not None:
                    f.write(f"{element} {float(pos[0]):.6f} {float(pos[1]):.6f} {float(pos[2]):.6f} {int(bb_indices[i])}\n")
                else:
                    f.write(f"{element} {float(pos[0]):.6f} {float(pos[1]):.6f} {float(pos[2]):.6f}\n")

def visualize_predictions(pred_file, save_dir=None, gt_xyz="ground_truth.xyz", 
                          pred_xyz="predictions.xyz", cg_xyz=None, cg_gt_xyz=None, 
                          cg_element="C", use_compact_indices=True):
    """Visualize all predictions from MOFFlow model.
    
    Args:
        pred_file (str): Path to predictions.pt file
        save_dir (str, optional): Directory to save XYZ files. Defaults to None.
        gt_xyz (str, optional): Filename for ground truth structures. Defaults to "ground_truth.xyz".
        pred_xyz (str, optional): Filename for predicted structures. Defaults to "predictions.xyz".
        cg_xyz (str, optional): Filename for coarse-grained crystals. Defaults to None.
        cg_gt_xyz (str, optional): Filename for ground truth coarse-grained crystals. Defaults to None.
        cg_element (str, optional): Element symbol for coarse-grained sites. Defaults to "C".
        use_compact_indices (bool, optional): If True, assumes atom_types are compact indices. Defaults to True.
    """
    # Load predictions
    results = torch.load(pred_file, weights_only=False)
    
    # Set up save directory
    if save_dir is None:
        save_dir = Path(pred_file).parent / 'xyz_files'
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Process ground truth
    gt_batch = results['gt_data_batch']
    
    # gt_lattice is already in matrix format [B, 3, 3]
    gt_lattice = gt_batch['lattice_1']  # Keep as matrix format
    
    # Get reference to trans_1 for later use
    trans_1 = gt_batch['trans_1']
    
    # Get building block information if available
    if 'num_bbs' in results:
        # Use building block information from the first sample (assuming same structure across samples)
        num_bbs_per_struct = results['num_bbs'][0].tolist() if len(results['num_bbs'].shape) > 1 else [results['num_bbs'][0].item()]
        gt_list = split_data_with_bb_indices(
            gt_batch['gt_coords'],
            gt_batch['atom_types'],
            gt_lattice,  # Use matrix format directly
            gt_batch['num_atoms'].tolist(),
            num_bbs_per_struct
        )
    else:
        # Fallback to regular split without building block indices
        gt_list = split_data(
            gt_batch['gt_coords'],
            gt_batch['atom_types'],
            gt_lattice,  # Use matrix format directly
            gt_batch['num_atoms'].tolist()
        )
    trans_1 = gt_batch['trans_1']
    # Save all ground truth structures to one file
    save_combined_xyz(
        gt_list,
        save_dir / gt_xyz,
        structure_type="GT",
        use_compact_indices=use_compact_indices
    )
    
    # Process all predictions and combine into one list
    num_samples = results['cart_coords'].shape[0]
    all_predictions = []
    for k in range(num_samples):
        # lattice matrices are already in [num_mofs, 3, 3] format
        lattice_matrices = results['lattices'][k]  # [num_mofs, 3, 3]
        
        # Use building block information if available
        if 'num_bbs' in results:
            num_bbs_per_struct = results['num_bbs'][k].tolist()
            pred_list = split_data_with_bb_indices(
                results['cart_coords'][k],
                results['atom_types'][k],
                lattice_matrices,  # Use matrix format directly
                results['num_atoms'][k].tolist(),
                num_bbs_per_struct
            )
        else:
            # Fallback to regular split without building block indices
            pred_list = split_data(
                results['cart_coords'][k],
                results['atom_types'][k],
                lattice_matrices,  # Use matrix format directly
                results['num_atoms'][k].tolist()
            )
        all_predictions.extend(pred_list)
    
    # Save all predictions to one file
    save_combined_xyz(
        all_predictions,
        save_dir / pred_xyz,
        structure_type="Pred",
        use_compact_indices=use_compact_indices
    )

    # --- Save coarse-grained crystals using pred_trans and lattice ---
    if cg_xyz is not None and cg_gt_xyz is not None and "pred_trans" in results and "num_bbs" in results:
        all_cg_structures = []
        all_gt_structures = []
        for k in range(num_samples):
            pred_trans = results['pred_trans'][k]  # [total_num_bbs, 3] (fractional)
            lattice_matrices = results['lattices'][k]  # [num_mofs, 3, 3]
            num_bbs = results['num_bbs'][k]        # [num_mofs]
            offset = 0
            for n, lat_mat, gt_lat in zip(num_bbs, lattice_matrices, gt_batch['lattice_1']):
                # lat_mat and gt_lat are already 3x3 matrices
                
                # Convert fractional to cartesian using matrix form
                pred_cart = frac_to_cart(pred_trans[offset:offset+n], lat_mat)
                gt_cart = frac_to_cart(trans_1[offset:offset+n], gt_lat)
                cg_struct = {
                    "cart_coords": pred_cart,
                    "atom_types": torch.full((n,), chemical_symbols.index(cg_element), dtype=torch.long),
                    "lattice": lat_mat  # Store as 3x3 matrix
                }
                gt_struct = {
                    "cart_coords": gt_cart,
                    "atom_types": torch.full((n,), chemical_symbols.index(cg_element), dtype=torch.long),
                    "lattice": gt_lat  # Store as 3x3 matrix
                }
                all_cg_structures.append(cg_struct)
                if k == 0:  # Only save GT structures once
                    all_gt_structures.append(gt_struct)
                offset += n
        save_combined_xyz(
            all_cg_structures,
            save_dir / cg_xyz,
            structure_type=f"CG_{cg_element}",
            use_compact_indices=False  # Always use regular atomic numbers for CG structures
        )
        save_combined_xyz(
            all_gt_structures,
            save_dir / cg_gt_xyz,
            structure_type=f"CG_{cg_element}",
            use_compact_indices=False  # Always use regular atomic numbers for CG structures
        )

    print(f"Saved combined XYZ files to {save_dir}")
    print(f"Ground truth: {gt_xyz}")
    print(f"All predictions: {pred_xyz}")
    if cg_xyz is not None and cg_gt_xyz is not None:
        print(f"Coarse-grained predictions: {cg_xyz}")
        print(f"Coarse-grained ground truth: {cg_gt_xyz}")
    
    return {
        'save_dir': save_dir,
        'num_samples': num_samples
    }

def rotation_angle_diff(R1, R2):
    R = R1.T @ R2
    trace = np.trace(R)
    cos_theta = (trace - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # for numerical stability
    theta = np.arccos(cos_theta)
    return np.degrees(theta)  # in degrees if desired

def lattice6_to_mat33(lattice_6):
    # lattice_6: [B, 6] (a, b, c, alpha, beta, gamma)
    # returns: [B, 3, 3]
    import math
    a, b, c, alpha, beta, gamma = [lattice_6[:, i] for i in range(6)]
    alpha = torch.deg2rad(alpha)
    beta = torch.deg2rad(beta)
    gamma = torch.deg2rad(gamma)
    vec_a = torch.stack([a, torch.zeros_like(a), torch.zeros_like(a)], dim=-1)
    vec_b = torch.stack([b * torch.cos(gamma), b * torch.sin(gamma), torch.zeros_like(b)], dim=-1)
    cx = c * torch.cos(beta)
    cy = c * (torch.cos(alpha) - torch.cos(beta) * torch.cos(gamma)) / torch.sin(gamma)
    cz = torch.sqrt(c**2 - cx**2 - cy**2)
    vec_c = torch.stack([cx, cy, cz], dim=-1)
    return torch.stack([vec_a, vec_b, vec_c], dim=1)  # [B, 3, 3]

def frac_to_cart(frac_coords: Tensor, lattice: Tensor) -> Tensor:
    """
    Convert fractional coordinates to cartesian coordinates.
    Args:
        frac_coords: [N, 3] fractional coordinates
        lattice: [3, 3] lattice matrix
    Returns:
        [N, 3] cartesian coordinates
    """
    return torch.matmul(frac_coords, lattice)

def lattice_mat33_to_6(lattice_mat):
    """
    Convert batch of 3x3 lattice matrices to 6-parameter vectors (a, b, c, alpha, beta, gamma).
    Args:
        lattice_mat: [B, 3, 3] tensor
    Returns:
        lattice_6: [B, 6] tensor
    """
    a_vec = lattice_mat[:, 0, :]
    b_vec = lattice_mat[:, 1, :]
    c_vec = lattice_mat[:, 2, :]

    a = torch.norm(a_vec, dim=1)
    b = torch.norm(b_vec, dim=1)
    c = torch.norm(c_vec, dim=1)

    def angle(v1, v2):
        cos_angle = (v1 * v2).sum(dim=1) / (torch.norm(v1, dim=1) * torch.norm(v2, dim=1))
        # Clamp for numerical stability
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        return torch.acos(cos_angle) * 180.0 / torch.pi

    alpha = angle(b_vec, c_vec)  # angle between b and c
    beta  = angle(a_vec, c_vec)  # angle between a and c
    gamma = angle(a_vec, b_vec)  # angle between a and b

    lattice_6 = torch.stack([a, b, c, alpha, beta, gamma], dim=1)
    return lattice_6

_quat_elements = ["a", "b", "c", "d"]
_qtr_keys = [l1 + l2 for l1 in _quat_elements for l2 in _quat_elements]
_qtr_ind_dict = {key: ind for ind, key in enumerate(_qtr_keys)}
def _to_mat(pairs):
    mat = np.zeros((4, 4))
    for pair in pairs:
        key, value = pair
        ind = _qtr_ind_dict[key]
        mat[ind // 4][ind % 4] = value

    return mat

_QTR_MAT = np.zeros((4, 4, 3, 3))
_QTR_MAT[..., 0, 0] = _to_mat([("aa", 1), ("bb", 1), ("cc", -1), ("dd", -1)])
_QTR_MAT[..., 0, 1] = _to_mat([("bc", 2), ("ad", -2)])
_QTR_MAT[..., 0, 2] = _to_mat([("bd", 2), ("ac", 2)])
_QTR_MAT[..., 1, 0] = _to_mat([("bc", 2), ("ad", 2)])
_QTR_MAT[..., 1, 1] = _to_mat([("aa", 1), ("bb", -1), ("cc", 1), ("dd", -1)])
_QTR_MAT[..., 1, 2] = _to_mat([("cd", 2), ("ab", -2)])
_QTR_MAT[..., 2, 0] = _to_mat([("bd", 2), ("ac", -2)])
_QTR_MAT[..., 2, 1] = _to_mat([("cd", 2), ("ab", 2)])
_QTR_MAT[..., 2, 2] = _to_mat([("aa", 1), ("bb", -1), ("cc", -1), ("dd", 1)])

def quat_to_rot(quat: torch.Tensor) -> torch.Tensor:
    """
    Converts a quaternion to a rotation matrix.

    Args:
        quat: [*, 4] quaternions
    Returns:
        [*, 3, 3] rotation matrices
    """
    # [*, 4, 4]
    quat = quat[..., None] * quat[..., None, :]

    # [4, 4, 3, 3]
    mat = quat.new_tensor(_QTR_MAT, requires_grad=False)

    # [*, 4, 4, 3, 3]
    shaped_qtr_mat = mat.view((1,) * len(quat.shape[:-2]) + mat.shape)
    quat = quat[..., None, None] * shaped_qtr_mat

    # [*, 3, 3]
    return torch.sum(quat, dim=(-3, -4))


def rot_to_quat(rot: torch.Tensor):
    rot = [[rot[..., i, j] for j in range(3)] for i in range(3)]
    [[R11, R12, R13], [R21, R22, R23], [R31, R32, R33]] = rot

    K = [
        [
            R11 + R22 + R33,
            R32 - R23,
            R13 - R31,
            R21 - R12,
        ],
        [
            R32 - R23,
            R11 - R22 - R33,
            R12 + R21,
            R13 + R31,
        ],
        [
            R13 - R31,
            R12 + R21,
            R22 - R11 - R33,
            R23 + R32,
        ],
        [
            R21 - R12,
            R13 + R31,
            R23 + R32,
            R33 - R11 - R22,
        ],
    ]

    K = (1.0 / 3.0) * torch.stack([torch.stack(t, dim=-1) for t in K], dim=-2)

    _, vectors = torch.linalg.eigh(K)
    quaternion = vectors[..., -1]
    
    mask = quaternion[..., 0] < 0
    quaternion[mask] *= -1
    return quaternion

def LERP(translation_0, translation_1, timestep):
    translation_t = (1 - timestep) * translation_0 + timestep * translation_1  # [num_cluster*17, 3]
    return translation_t

# Function to perform quaternion multiplication for batches
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)
