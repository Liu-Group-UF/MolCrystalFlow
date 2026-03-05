import logging
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from pytorch_lightning import LightningModule
from torch_geometric.utils import scatter

from molcrystalflow.data import so3_utils
from molcrystalflow.data import utils as du
from molcrystalflow.data.interpolant import (
    Interpolant,
    _symmetric_so3,
    _uniform_cell_trans,
    _uniform_so3,
)
from molcrystalflow.data.utils import lattice6_to_mat33
from molcrystalflow.models import utils as mu
from molcrystalflow.models.molcrystalnet import FlowModel


class FlowModule(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._interpolant_cfg = cfg.interpolant
        self._inference_cfg = getattr(cfg, 'inference', None)

        # Set-up vector field prediction model
        self.model = FlowModel(cfg.model)

        # Set-up interpolant
        self.interpolant = Interpolant(cfg.interpolant)

        self.results = []
        self.save_hyperparameters()

        self._checkpoint_dir = None
        self._inference_dir = None

        self.logmap = so3_utils.LogSO3()

    @property
    def checkpoint_dir(self):
        if self._checkpoint_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    checkpoint_dir = [self._exp_cfg.checkpointer.dirpath]
                else:
                    checkpoint_dir = [None]
                dist.broadcast_object_list(checkpoint_dir, src=0)
                checkpoint_dir = checkpoint_dir[0]
            else:
                checkpoint_dir = self._exp_cfg.checkpointer.dirpath
            self._checkpoint_dir = checkpoint_dir
            os.makedirs(self._checkpoint_dir, exist_ok=True)
        return self._checkpoint_dir

    @property
    def inference_dir(self):
        if self._inference_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    inference_dir = [self._exp_cfg.inference_dir]
                else:
                    inference_dir = [None]
                dist.broadcast_object_list(inference_dir, src=0)
                inference_dir = inference_dir[0]
            else:
                inference_dir = self._exp_cfg.inference_dir
            self._inference_dir = inference_dir
            os.makedirs(self._inference_dir, exist_ok=True)
        return self._inference_dir

    def _log_scalar(
        self,
        key,
        value,
        on_step=True,
        on_epoch=False,
        prog_bar=True,
        batch_size=None,
        sync_dist=False,
        rank_zero_only=True
    ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )

    def configure_optimizers(self):
        optimizer =  torch.optim.AdamW(
            params=self.model.parameters(),
            **self._exp_cfg.optimizer
        )
        if not self._exp_cfg.use_lr_scheduler:
            return optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **self._exp_cfg.lr_scheduler
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'valid/loss'
        }

    def model_step(self, noisy_batch):
        """Compute losses for a single training/validation step."""
        training_cfg = self._exp_cfg.training

        # Ground truth labels
        gt_rotmats_1 = noisy_batch['rotmats_1']
        gt_lattice_1 = noisy_batch['lattice_1']
        gt_b_trans = noisy_batch['b_trans']
        rotmats_t = noisy_batch['rotmats_t']

        gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, gt_rotmats_1.float())
        if torch.any(torch.isnan(gt_rot_vf)):
            raise ValueError('NaN encountered in gt_rot_vf')

        # Timestep used for normalization
        so3_t = noisy_batch['so3_t']
        l_t = noisy_batch['l_t']
        so3_norm_scale = 1 - torch.min(
            so3_t, torch.tensor(training_cfg.t_normalize_clip)
        )
        l_norm_scale = 1 - torch.min(
            l_t, torch.tensor(training_cfg.t_normalize_clip)
        )

        # Model output predictions
        model_output = self.model(noisy_batch)
        pred_b_trans = model_output['pred_b_trans']
        pred_rotmats_1 = model_output['pred_rotmats']
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
        pred_lattice_1 = model_output['pred_lattice']

        # Translation VF loss (L2)
        trans_error = (gt_b_trans - pred_b_trans) * training_cfg.trans_scale
        trans_loss = training_cfg.translation_loss_weight * torch.mean(
            trans_error ** 2, dim=-1)
        trans_loss = scatter(trans_loss, noisy_batch.batch, reduce='mean')
        trans_loss = torch.clamp(trans_loss, max=5)

        # Rotation VF loss (L2)
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / so3_norm_scale
        rots_vf_loss = training_cfg.rotation_loss_weights * torch.mean(
            rots_vf_error ** 2, dim=-1)
        rots_vf_loss = scatter(rots_vf_loss, noisy_batch.batch, reduce='mean')
        se3_vf_loss = trans_loss + rots_vf_loss
        if torch.any(torch.isnan(se3_vf_loss)):
            raise ValueError('NaN encountered in se3_vf_loss')

        # Lattice loss (L2)
        gt_lattice_flat = gt_lattice_1.view(gt_lattice_1.shape[0], -1)
        pred_lattice_flat = pred_lattice_1.view(pred_lattice_1.shape[0], -1)
        cell_error = (gt_lattice_flat - pred_lattice_flat) / l_norm_scale
        cell_loss = training_cfg.cell_loss_weight * torch.mean(
            cell_error ** 2, dim=-1)

        return {
            'trans_loss': trans_loss,
            'rots_vf_loss': rots_vf_loss,
            'se3_vf_loss': se3_vf_loss,
            'cell_loss': cell_loss,
        }

    def on_train_start(self):
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()

    def training_step(self, batch, stage: int):
        """Execute a single training step."""
        step_start_time = time.time()
        self.interpolant.set_device(batch.trans_1.device)
        noisy_batch = self.interpolant.corrupt_batch(batch)
        batch_losses = self.model_step(noisy_batch)
        num_batch = batch.num_graphs
        total_losses = {k: torch.mean(v) for k, v in batch_losses.items()}

        for k, v in total_losses.items():
            self._log_scalar(f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # Log timestep
        batch_t = torch.squeeze(noisy_batch['l_t'])
        self._log_scalar(
            "train/t", np.mean(du.to_numpy(batch_t)),
            prog_bar=False, batch_size=num_batch
        )

        # Stratified losses
        for loss_name, batch_loss in batch_losses.items():
            stratified_losses = mu.t_stratified_loss(
                batch_t, batch_loss, loss_name=loss_name
            )
            for k, v in stratified_losses.items():
                self._log_scalar(f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # Training throughput
        self._log_scalar("train/batch_size", num_batch, prog_bar=False)
        step_time = time.time() - step_start_time
        self._log_scalar("train/samples_per_second", num_batch / step_time)

        train_loss = total_losses['se3_vf_loss'] + total_losses['cell_loss']
        self._log_scalar("train/loss", train_loss, batch_size=num_batch)
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        """Execute a single validation step."""
        self.interpolant.set_device(batch.trans_1.device)
        noisy_batch = self.interpolant.corrupt_batch(batch)
        batch_losses = self.model_step(noisy_batch)
        total_losses = {k: torch.mean(v) for k, v in batch_losses.items()}

        for k, v in total_losses.items():
            self._log_scalar(
                f"valid/{k}", v,
                on_step=False, on_epoch=True, prog_bar=False,
                sync_dist=True, rank_zero_only=False
            )

        # Log timestep
        batch_t = torch.squeeze(noisy_batch['l_t'])
        self._log_scalar(
            "valid/t", np.mean(du.to_numpy(batch_t)),
            on_step=False, on_epoch=True, prog_bar=False,
            sync_dist=True, rank_zero_only=False
        )

        # Stratified losses
        for loss_name, batch_loss in batch_losses.items():
            stratified_losses = mu.t_stratified_loss(
                batch_t, batch_loss, loss_name=loss_name
            )
            for k, v in stratified_losses.items():
                self._log_scalar(
                    f"valid/{k}", v,
                    on_step=False, on_epoch=True, prog_bar=False,
                    sync_dist=True, rank_zero_only=False
                )

        # Validation loss
        valid_loss = total_losses['se3_vf_loss'] + total_losses['cell_loss']
        self._log_scalar(
            "valid/loss", valid_loss,
            on_step=False, on_epoch=True, prog_bar=True,
            sync_dist=True, rank_zero_only=False
        )
        return valid_loss

    def forward(self, batch, num_timesteps=None):
        """Run inference to generate structures."""
        num_batch = batch['lattice_1'].shape[0]
        num_bbs = batch['num_bbs']
        atom_types = batch.get('atom_types', None)
        local_coords = batch.get('local_coords', None)
        bb_num_vec = batch.get('bb_num_vec', None)
        batch_vec = batch.get('batch', None)

        # Handle both training and inference cases
        if self._inference_cfg is not None:
            num_samples = self._inference_cfg.num_samples
        else:
            num_samples = 1

        all_cart_coords = []
        all_atom_types = []
        all_lattices = []
        all_num_atoms = []
        all_pred_trans = []
        all_pred_rotmats = []

        save_trajectories = getattr(self._inference_cfg, 'save_trajectories', True)
        all_traj_lattices = []
        all_traj_trans = []
        all_traj_rotmats = []
        all_traj_timesteps = []

        for sample_idx in range(num_samples):
            # Sample new initial noise for each sample
            trans_0 = batch.get('trans_0', None)
            rotmats_0 = batch.get('rotmats_0', None)
            lattice_0 = batch.get('lattice_0', None)

            if trans_0 is None:
                trans_0 = _uniform_cell_trans(len(batch_vec), batch_vec, num_bbs, self.device)

            if rotmats_0 is None:
                if self._interpolant_cfg.rots.prior_type == 'uniform':
                    rotmats_0 = _uniform_so3(len(batch_vec), num_bbs, self.device)
                elif self._interpolant_cfg.rots.prior_type == 'symmetric':
                    rotmats_0 = _symmetric_so3(len(batch_vec), num_bbs, self.device)
                else:
                    raise ValueError(
                        f"Unknown prior type: {self._interpolant_cfg.rots.prior_type}. "
                        "Supported types: uniform, symmetric"
                    )

            if lattice_0 is None:
                lengths_0 = self.interpolant._lognormal.sample((num_batch,)).to(self.device)
                angles_0 = self.interpolant._uniform.sample((num_batch, 3)).to(self.device)
                lattice_params_0 = torch.cat([lengths_0, angles_0], dim=-1)
                lattice_0 = lattice6_to_mat33(lattice_params_0)

            # Set up time grid
            if num_timesteps is None:
                num_timesteps = self._interpolant_cfg.sampling.num_timesteps
            ts = torch.linspace(self._interpolant_cfg.min_t, 1.0, num_timesteps, device=self.device)
            t_1 = ts[0]

            mc_traj = [(trans_0, rotmats_0, lattice_0)]
            # Store intermediate coordinates for this trajectory
            traj_lattices = []
            traj_trans = []      # Fractional coordinates
            traj_rotmats = []    # Rotation matrices
            traj_timesteps = []

            for i, t_2 in enumerate(ts[1:]):
                trans_t_1, rotmats_t_1, lattice_t_1 = mc_traj[-1]
                batch['trans_t'] = trans_t_1
                batch['rotmats_t'] = rotmats_t_1
                batch['lattice_t'] = lattice_t_1
                t = torch.ones((num_batch, 1), device=self.device) * t_1
                t_repeat = t.repeat_interleave(num_bbs, dim=0)
                batch['so3_t'] = t_repeat
                batch['r3_t'] = t_repeat
                batch['l_t'] = t
                d_t = t_2 - t_1

                with torch.no_grad():
                    model_out = self.model(batch)

                pred_b_trans = model_out['pred_b_trans']
                pred_rotmats_1 = model_out['pred_rotmats']

                pred_lattice_1 = model_out['pred_lattice']

                # Take reverse step
                trans_t_2 = self.interpolant._b_trans_euler_step(
                    d_t, t_1, pred_b_trans, trans_t_1
                )
                rotmats_t_2 = self.interpolant._rots_euler_step(
                    d_t, t_1, pred_rotmats_1, rotmats_t_1
                )
                lattice_t_2 = self.interpolant._trans_euler_step(
                    d_t, t_1, pred_lattice_1, lattice_t_1, scaling=0
                )

                mc_traj.append((trans_t_2, rotmats_t_2, lattice_t_2))

                # Store intermediate coordinates for trajectory
                if save_trajectories:
                    traj_lattices.append(lattice_t_2)
                    traj_trans.append(trans_t_2)
                    traj_rotmats.append(rotmats_t_2)
                    traj_timesteps.append(t_2)
                t_1 = t_2
            # Final step
            t_1 = ts[-1]
            trans_t_1, rotmats_t_1, lattice_t_1 = mc_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            batch['lattice_t'] = lattice_t_1
            t = torch.ones((num_batch, 1), device=self.device) * t_1
            t_repeat = t.repeat_interleave(num_bbs, dim=0)
            batch['so3_t'] = t_repeat
            batch['r3_t'] = t_repeat
            batch['l_t'] = t
            with torch.no_grad():
                model_out = self.model(batch)
            pred_b_trans = model_out['pred_b_trans']
            pred_trans_1 = self.interpolant._b_trans_euler_step(
                    d_t, t_1, pred_b_trans, trans_t_1
                )
            pred_rotmats_1 = model_out['pred_rotmats']
            pred_lattice_1 = model_out['pred_lattice']
            batch_vec = batch['batch']                 # [num_bbs], values in 0..num_samples-1

            # Gather the correct lattice for each building block
            lattice_per_bb = pred_lattice_1[batch_vec]     # [num_bbs, 3, 3]
            _pred_trans_1 = torch.bmm(pred_trans_1.unsqueeze(1), lattice_per_bb).squeeze(1)  # [num_bbs, 3]

            # Assemble coordinates
            cart_coords = self.interpolant._assemble_coords(
                local_coords, pred_rotmats_1, _pred_trans_1, bb_num_vec)
            all_cart_coords.append(cart_coords)
            all_atom_types.append(atom_types)
            all_lattices.append(pred_lattice_1)
            all_num_atoms.append(batch['num_atoms'])
            all_pred_trans.append(pred_trans_1)
            all_pred_rotmats.append(pred_rotmats_1)
            # Add final step to trajectory
            if save_trajectories:
                traj_lattices.append(pred_lattice_1)
                traj_trans.append(pred_trans_1)
                traj_rotmats.append(pred_rotmats_1)
                traj_timesteps.append(ts[-1])
                all_traj_lattices.append(torch.stack(traj_lattices))  # [num_timesteps, num_bbs, 3, 3] or [num_timesteps, num_structures, 3, 3]
                all_traj_trans.append(torch.stack(traj_trans))        # [num_timesteps, num_bbs, 3]
                all_traj_rotmats.append(torch.stack(traj_rotmats))    # [num_timesteps, num_bbs, 3, 3]
                all_traj_timesteps.append(torch.stack(traj_timesteps))# [num_timesteps]
        # Stack all trajectories if saving
        if save_trajectories and all_traj_lattices:
            all_mc_trajs = {
                'lattices': torch.stack(all_traj_lattices),   # [num_samples, num_timesteps, num_bbs, 3, 3]
                'trans': torch.stack(all_traj_trans),         # [num_samples, num_timesteps, num_bbs, 3]
                'rotmats': torch.stack(all_traj_rotmats),     # [num_samples, num_timesteps, num_bbs, 3, 3]
                'timesteps': torch.stack(all_traj_timesteps)  # [num_samples, num_timesteps]
            }
        else:
            all_mc_trajs = None
        num_bbs = torch.stack([batch.num_bbs for _ in range(num_samples)], dim=0)  # [num_samples, num_mofs]
        # Stack results
        results = {
            'cart_coords': torch.stack(all_cart_coords),  # [num_samples, num_atoms, 3]
            'atom_types': torch.stack(all_atom_types),    # [num_samples, num_atoms]
            'lattices': torch.stack(all_lattices),        # [num_samples, num_mofs, 3, 3]
            'num_atoms': torch.stack(all_num_atoms),      # [num_samples, num_mofs]
            'pred_trans': torch.stack(all_pred_trans),    # [num_samples, num_bbs, 3]
            'pred_rotmats': torch.stack(all_pred_rotmats),    # [num_samples, num_bbs, 3]
            'num_bbs': num_bbs,                          # [num_samples, num_mofs]
            'gt_data_batch': batch,                       # Original batch for reference
            'mc_trajs': all_mc_trajs                   # Stacked trajectories or None
        }
        return results
