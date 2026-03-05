from collections import defaultdict

import copy

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from torch.distributions import LogNormal, Uniform
from torch_scatter import scatter_mean

from molcrystalflow.data import so3_utils
from molcrystalflow.data import utils as du
from molcrystalflow.data.utils import lattice6_to_mat33


def _centered_gaussian(num_batch, batch_vec, num_bbs, device):
    noise = torch.randn(num_batch, 3, device=device)
    bb_mean = scatter_mean(noise, batch_vec, dim=0)
    bb_mean = bb_mean.repeat_interleave(num_bbs, dim=0)
    return noise - bb_mean


def _uniform_so3(num_batch, num_bbs, device):
    return torch.tensor(
        Rotation.random(num_batch).as_matrix(),
        device=device,
        dtype=torch.float32,
    )


def _uniform_cell_trans(num_batch, batch_vec, num_bbs, device):
    noise = torch.rand(num_batch, 3, device=device)
    return noise


def _symmetric_so3(num_batch, num_bbs, device):
    """Generate symmetric SO(3) rotations by mirroring base axis-angles.
    
    Args:
        num_batch: Total number of building blocks (unused, kept for API compatibility)
        num_bbs: Number of building blocks per batch
        device: Torch device
        
    Returns:
        Rotation matrices [sum(num_bbs), 3, 3]
    """
    def mirror_axis(axis, i):
        if i % 4 == 0:
            return axis  # no mirroring
        elif i % 4 == 1:
            return torch.tensor([-axis[0], -axis[1], axis[2]], device=device)
        elif i % 4 == 2:
            return torch.tensor([-axis[0], axis[1], -axis[2]], device=device)
        else:  # i % 4 == 3
            return torch.tensor([axis[0], -axis[1], -axis[2]], device=device)

    # Generate base axis-angle for each bb
    axis_angles = torch.tensor(
        Rotation.random(num_bbs.shape[0]).as_rotvec(),
        device=device,
        dtype=torch.float32,
    )

    angles = axis_angles.norm(dim=1, keepdim=True)
    axes = axis_angles / (angles + 1e-8)  # normalize to unit vectors

    rotations = []

    for bb_idx in range(num_bbs.shape[0]):
        num_bb = num_bbs[bb_idx].item()
        base_axis = axes[bb_idx]
        base_angle = angles[bb_idx].item()

        for i in range(num_bb):
            # Apply mirroring to axis
            mirrored_axis = mirror_axis(base_axis, i)

            # Convert axis-angle to rotation matrix
            rotvec = (mirrored_axis * base_angle).cpu().numpy()
            R = Rotation.from_rotvec(rotvec).as_matrix()
            R_tensor = torch.tensor(R, dtype=torch.float32, device=device)

            rotations.append(R_tensor.unsqueeze(0))

    rotations = torch.cat(rotations, dim=0)  # [sum(num_bbs), 3, 3]
    return rotations


class Interpolant:

    def __init__(self, cfg):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._lattice_cfg = cfg.lattice
        self._device = None
        self._lognormal = LogNormal(
            loc=torch.Tensor(self._lattice_cfg.lognormal.loc),
            scale=torch.Tensor(self._lattice_cfg.lognormal.scale)
        )
        self._uniform = Uniform(
            low=self._lattice_cfg.uniform.low - self._lattice_cfg.uniform.eps,
            high=self._lattice_cfg.uniform.high + self._lattice_cfg.uniform.eps
        )

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2*self._cfg.min_t) + self._cfg.min_t

    def unwrap(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        separation_vector = x_1 - x_0
        length_over_two = 0.5
        shortest_separation_vector = torch.remainder(separation_vector + length_over_two,
                                                     1.0) - length_over_two
        return x_0 + shortest_separation_vector

    def _find_optimal_rotation_permutation(self, rotmats_0_batch, rotmats_1_batch, axis_flips_batch=None):
        """Find optimal permutation of rotation matrices to minimize total geodesic distance.
        Groups by axis_flips if provided - only BBs in same group can be matched.
        
        Args:
            rotmats_0_batch: Prior rotation matrices for one batch [N, 3, 3]
            rotmats_1_batch: Target rotation matrices for one batch [N, 3, 3]
            axis_flips_batch: Axis flips for each BB [N, 3] or None
            
        Returns:
            Optimal permutation indices [N]
        """
        N = rotmats_0_batch.shape[0]
        
        def _compute_permutation(rotmats_0, rotmats_1):
            """Compute optimal permutation using Hungarian algorithm."""
            n = rotmats_0.shape[0]
            
            # Compute pairwise geodesic distances [n, n]
            rotmats_0_expanded = rotmats_0.unsqueeze(1).expand(-1, n, -1, -1)
            rotmats_1_expanded = rotmats_1.unsqueeze(0).expand(n, -1, -1, -1)
            
            # Relative rotation: R_rel = R_1^T @ R_0
            rotmats_rel = torch.matmul(
                rotmats_1_expanded.transpose(-2, -1),
                rotmats_0_expanded
            )
            
            # Extract angles from rotation matrices
            traces = torch.diagonal(rotmats_rel, dim1=-2, dim2=-1).sum(dim=-1)
            cos_theta = torch.clamp((traces - 1) / 2, -1 + 1e-7, 1 - 1e-7)
            pairwise_distances = torch.acos(cos_theta)
            
            # Use Hungarian algorithm
            cost_matrix = pairwise_distances.cpu().numpy()
            _, col_indices = linear_sum_assignment(cost_matrix)
            return torch.tensor(col_indices, dtype=torch.long, device=rotmats_0.device)
        
        if axis_flips_batch is None:
            return _compute_permutation(rotmats_0_batch, rotmats_1_batch)
        
        # Group by axis flips - convert to hashable tuples for grouping
        axis_flip_tuples = [tuple(axis_flips_batch[i].cpu().numpy()) for i in range(N)]
        groups = defaultdict(list)
        for idx, flip_tuple in enumerate(axis_flip_tuples):
            groups[flip_tuple].append(idx)
        
        # Initialize permutation with identity
        permutation = torch.arange(N, dtype=torch.long, device=rotmats_0_batch.device)
        
        # Process each group independently
        for group_indices in groups.values():
            if len(group_indices) <= 1:
                continue  # No permutation needed for single-element groups
            
            group_indices = torch.tensor(group_indices, device=rotmats_0_batch.device)
            rotmats_0_group = rotmats_0_batch[group_indices]
            rotmats_1_group = rotmats_1_batch[group_indices]
            
            # Find optimal permutation within this group
            group_permutation = _compute_permutation(rotmats_0_group, rotmats_1_group)
            
            # Apply group permutation to the global permutation
            permutation[group_indices] = group_indices[group_permutation]
        
        return permutation

    def _find_optimal_permutation(self, trans_0_batch, trans_1_batch, axis_flips_batch=None):
        """Find optimal permutation of building blocks to minimize total unwrapped distance.
        Groups by axis_flips if provided - only BBs in same group can be matched.
        
        Args:
            trans_0_batch: Prior translations for one batch [N, 3]
            trans_1_batch: Target translations for one batch [N, 3]
            axis_flips_batch: Axis flips for each BB [N, 3] or None
            
        Returns:
            Optimal permutation indices [N]
        """
        N = trans_0_batch.shape[0]
        
        def _compute_permutation(trans_0, trans_1):
            """Compute optimal permutation using Hungarian algorithm."""
            n = trans_0.shape[0]
            
            # Compute pairwise unwrapped distances [n, n]
            trans_0_expanded = trans_0.unsqueeze(1).expand(-1, n, -1)  # [n, n, 3]
            trans_1_expanded = trans_1.unsqueeze(0).expand(n, -1, -1)  # [n, n, 3]
            
            # Compute unwrapped distances for all pairs
            separation_vectors = trans_1_expanded - trans_0_expanded  # [n, n, 3]
            length_over_two = 0.5
            shortest_separation_vectors = torch.remainder(separation_vectors + length_over_two,
                                                         1.0) - length_over_two
            pairwise_distances = torch.norm(shortest_separation_vectors, dim=-1)  # [n, n]
            
            # Use Hungarian algorithm
            cost_matrix = pairwise_distances.cpu().numpy()
            _, col_indices = linear_sum_assignment(cost_matrix)
            return torch.tensor(col_indices, dtype=torch.long, device=trans_0.device)
        
        if axis_flips_batch is None:
            return _compute_permutation(trans_0_batch, trans_1_batch)
        
        # Group by axis flips - convert to hashable tuples for grouping
        axis_flip_tuples = [tuple(axis_flips_batch[i].cpu().numpy()) for i in range(N)]
        groups = defaultdict(list)
        for idx, flip_tuple in enumerate(axis_flip_tuples):
            groups[flip_tuple].append(idx)
        
        # Initialize permutation with identity
        permutation = torch.arange(N, dtype=torch.long, device=trans_0_batch.device)
        
        # Process each group independently
        for group_indices in groups.values():
            if len(group_indices) <= 1:
                continue  # No permutation needed for single-element groups
            
            group_indices = torch.tensor(group_indices, device=trans_0_batch.device)
            trans_0_group = trans_0_batch[group_indices]
            trans_1_group = trans_1_batch[group_indices]
            
            # Find optimal permutation within this group
            group_permutation = _compute_permutation(trans_0_group, trans_1_group)
            # Apply group permutation to the global permutation
            permutation[group_indices] = group_indices[group_permutation]
        
        return permutation

    def _corrupt_trans(self, trans_1, batch_vec, num_bbs, t, prior_type='uniform', ot=False, axis_flips=None):
        """Corrupt translations with optimal permutation for shortest paths.
        
        Args:
            trans_1: Target translations [M, 3]
            batch_vec: Batch assignment vector [M]
            num_bbs: Number of building blocks per MOF
            t: Time parameter [M, 1]
            prior_type: Type of prior distribution ('uniform', 'gaussian', 'spg_aware')
            ot: Whether to use optimal transport
            axis_flips: Axis flips for each building block [M, 3] or None
        """
        # Generate translations based on prior type
        if prior_type == 'uniform':
            trans_0 = _uniform_cell_trans(trans_1.shape[0], batch_vec, num_bbs, self._device)
        elif prior_type == 'gaussian':
            trans_nm_0 = _centered_gaussian(trans_1.shape[0], batch_vec, num_bbs, self._device)
            trans_0 = trans_nm_0 * du.NM_TO_ANG_SCALE
        else:
            raise ValueError(f'Unknown prior_type: {prior_type}. Supported types: uniform, gaussian')

        if ot:
            # Process each batch separately to find optimal permutations
            trans_0_permuted = torch.zeros_like(trans_0)
            current_idx = 0
            
            for batch_idx in range(len(num_bbs)):
                batch_size = num_bbs[batch_idx].item()
                
                # Extract building blocks for this batch
                trans_0_batch = trans_0[current_idx:current_idx + batch_size]  # [N, 3]
                trans_1_batch = trans_1[current_idx:current_idx + batch_size]  # [N, 3]
                
                # Extract axis flips for this batch if available
                axis_flips_batch = None
                if axis_flips is not None:
                    axis_flips_batch = axis_flips[current_idx:current_idx + batch_size]
                
                # Find optimal permutation for this batch (grouped by axis_flips if available)
                permutation = self._find_optimal_permutation(trans_0_batch, trans_1_batch, axis_flips_batch)
                
                # Apply permutation to trans_0
                trans_0_permuted[current_idx:current_idx + batch_size] = trans_0_batch[permutation]
                
                current_idx += batch_size
            trans_0 = trans_0_permuted
        
        # Now use the permuted trans_0 for interpolation
        trans_1prime = self.unwrap(trans_0, trans_1)
        trans_t = (1 - t) * trans_0 + t * trans_1prime
        trans_t = trans_t % 1.0

        b_trans = trans_1prime - trans_0
        b_trans_mean = scatter_mean(b_trans, batch_vec, dim=0)
        b_trans_mean = b_trans_mean.repeat_interleave(num_bbs, dim=0)
        b_trans = b_trans - b_trans_mean
        return trans_t, b_trans

    def _corrupt_rotmats(self, rotmats_1, num_bbs, t, prior_type='symmetric', ot=False, axis_flips=None):
        """Corrupt rotations with optimal permutation for shortest geodesic paths.
        
        Args:
            rotmats_1: Target rotation matrices [M, 3, 3]
            num_bbs: Number of building blocks per MOF
            t: Time parameter [M, 1]
            prior_type: Type of prior distribution ('symmetric', 'uniform', 'spg_aware')
            ot: Whether to use optimal transport
            axis_flips: Axis flips for each building block [M, 3] or None
        """
        # Generate rotations based on prior type
        if prior_type == 'symmetric':
            rotmats_0 = _symmetric_so3(rotmats_1.shape[0], num_bbs, self._device)
        elif prior_type == 'uniform':
            rotmats_0 = _uniform_so3(rotmats_1.shape[0], num_bbs, self._device)
        else:
            raise ValueError(f'Unknown prior_type: {prior_type}. Supported types: symmetric, uniform')

        if ot:
            # Process each batch separately to find optimal permutations
            rotmats_0_permuted = torch.zeros_like(rotmats_0)
            current_idx = 0
            
            for batch_idx in range(len(num_bbs)):
                batch_size = num_bbs[batch_idx].item()
                
                # Extract rotation matrices for this batch
                rotmats_0_batch = rotmats_0[current_idx:current_idx + batch_size]  # [N, 3, 3]
                rotmats_1_batch = rotmats_1[current_idx:current_idx + batch_size]  # [N, 3, 3]
                
                # Extract axis flips for this batch if available
                axis_flips_batch = None
                if axis_flips is not None:
                    axis_flips_batch = axis_flips[current_idx:current_idx + batch_size]
                
                # Find optimal permutation for this batch (grouped by axis_flips if available)
                permutation = self._find_optimal_rotation_permutation(rotmats_0_batch, rotmats_1_batch, axis_flips_batch)
                
                # Apply permutation to rotmats_0
                rotmats_0_permuted[current_idx:current_idx + batch_size] = rotmats_0_batch[permutation]
                
                current_idx += batch_size
            rotmats_0 = rotmats_0_permuted
        
        # Use the permuted rotmats_0 for geodesic interpolation
        rotmats_t = so3_utils.geodesic_t(t, rotmats_1, rotmats_0)
        # rotmats_t = t * rotmats_1 + (1 - t) * rotmats_0
        return rotmats_t
    
    def _corrupt_lattice(self, lattice_1, t):
        """
        Corrupt lattice matrices with random sampling.
        
        Args:
            lattice_1: Target lattice matrices [B, 3, 3]
            t: Time parameter [B, 1]
        
        Returns:
            lattice_t: Corrupted lattice matrices [B, 3, 3]
        """
        batch_size = lattice_1.shape[0]
        
        # Sample random lattice parameters (a, b, c, alpha, beta, gamma)
        # lengths_0: sample 3 separate length distributions for each batch item
        lengths_0 = self._lognormal.sample((batch_size,)).to(self._device)  # [B, 3]
        angles_0 = self._uniform.sample((batch_size, 3)).to(self._device)   # [B, 3]
        lattice_params_0 = torch.cat([lengths_0, angles_0], dim=-1)  # [B, 6]
        
        # Convert sampled parameters to 3x3 matrix
        from molcrystalflow.data import utils as du
        lattice_0 = du.lattice6_to_mat33(lattice_params_0)  # [B, 3, 3]
        
        lattice_t = (1 - t.unsqueeze(-1)) * lattice_0 + t.unsqueeze(-1) * lattice_1
        return lattice_t

    def corrupt_batch(self, batch):
        # Note that you may only need group ot when number of bbs M >= 3
        noisy_batch = copy.deepcopy(batch)

        # [M, 3]
        trans_1 = batch['trans_1']  # Angstrom

        # [M, 3, 3]
        rotmats_1 = batch['rotmats_1']
        # quaternions_1 = batch['quaternions_1']

        # [B, 3, 3]
        lattice_1 = batch['lattice_1']
        
        # [M, 3] - axis flips for grouping in optimal transport
        axis_flips = batch.get('axis_flips', None)
        # print(axis_flips)

        # [M, 1]
        batch_vec = batch.batch

        # [M, 1]
        num_batch = batch.num_graphs
        t = self.sample_t(num_batch)[:, None]
        t_repeat = t.repeat_interleave(batch.num_bbs, dim=0)
        noisy_batch['so3_t'] = t_repeat
        noisy_batch['r3_t'] = t_repeat
        noisy_batch['l_t'] = t
        
        # Apply corruptions
        if self._trans_cfg.corrupt:
            # trans_t = self._corrupt_trans(
            #     trans_1, batch_vec, batch.num_bbs, t_repeat)
            trans_t, b_trans = self._corrupt_trans(
                trans_1, batch_vec, batch.num_bbs, t_repeat, self._trans_cfg.prior_type, self._trans_cfg.batch_ot, axis_flips)
        else:
            trans_t = trans_1
        if torch.any(torch.isnan(trans_t)):
            raise ValueError('NaN in trans_t during corruption')
        noisy_batch['trans_t'] = trans_t
        noisy_batch['b_trans'] = b_trans

        if self._rots_cfg.corrupt:
            rotmats_t = self._corrupt_rotmats(
                rotmats_1, batch.num_bbs, t_repeat, self._rots_cfg.prior_type, self._rots_cfg.batch_ot, axis_flips)
        else:
            rotmats_t = rotmats_1
        if torch.any(torch.isnan(rotmats_t)):
            raise ValueError('NaN in rotmats_t during corruption')
        noisy_batch['rotmats_t'] = rotmats_t

        if self._lattice_cfg.corrupt:
            # Resample if we get NaN in lattice corruption
            max_attempts = 10
            for attempt in range(max_attempts):
                lattice_t = self._corrupt_lattice(lattice_1, t)
                if not torch.any(torch.isnan(lattice_t)):
                    break
                if attempt == max_attempts - 1:
                    raise ValueError(f'NaN in lattice_t during corruption after {max_attempts} attempts')
        else:
            lattice_t = lattice_1
        noisy_batch['lattice_t'] = lattice_t

        return noisy_batch

    def _trans_vector_field(self, t, trans_1, trans_t):
        return (trans_1 - trans_t) / (1 - t)

    def _trans_euler_step(self, d_t, t, trans_1, trans_t, scaling=None):
        assert d_t > 0
        trans_vf = self._trans_vector_field(t, trans_1, trans_t)
        if scaling is None:
            scaling = self._trans_cfg.scaling
        return trans_t + trans_vf  * (1 + t * scaling) * d_t

    def _b_trans_euler_step(self, d_t, t, trans_vf, trans_t, scaling=None):
        assert d_t > 0
        if scaling is None:
            scaling = self._trans_cfg.scaling
        pred_trans = trans_t + trans_vf  * (1 + t * scaling) * d_t
        pred_trans = pred_trans % 1.0
        return pred_trans

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        if self._rots_cfg.sample_schedule == 'linear':
            scaling = 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == 'exp':
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(
                f'Unknown sample schedule {self._rots_cfg.sample_schedule}')
        return so3_utils.geodesic_t(
            (1 + t * scaling) * d_t, rotmats_1, rotmats_t)

    def _assemble_coords(self, local_coords, rotmats, trans, bb_num_vec):
        """
        Args:
            local_coords (torch.Tensor): [N, 3]
                local coordinates of the building block
            rotmats (torch.Tensor): [M, 3, 3]
                rotation matrices for each building block
            trans (torch.Tensor): [M, 3]
                translation vectors for each building block
            bb_num_vec (torch.Tensor): [M, ]
                each entry is the number of atoms in the corresponding building block
        
        Returns:
            global_coords (torch.Tensor): [N, 3]
                global coordinates of the building block
        """
        local_coords = torch.split(local_coords, bb_num_vec.tolist())   # M list of [N_i, 3]
        rigids = du.create_rigid(rotmats, trans)                        # [M,] rigid transformations
        
        assemble_coords = []
        for rigid, bb_local_coords in zip(rigids, local_coords):
            bb_global_coord = rigid.apply(bb_local_coords)
            assemble_coords.append(bb_global_coord)
        
        assemble_coords = torch.cat(assemble_coords, dim=0)
        return assemble_coords