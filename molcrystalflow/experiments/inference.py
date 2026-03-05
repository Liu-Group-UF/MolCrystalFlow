"""Script for running inference and evaluation."""

import os
import random
import time

import GPUtil
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from molcrystalflow.data.dataloader import MCDatamodule
from molcrystalflow.data.dataset import MCDataset
from molcrystalflow.experiments import utils as eu
from molcrystalflow.models.molcrystalflow import FlowModule

torch.set_float32_matmul_precision('high')
log = eu.get_pylogger(__name__)


class EvalRunner:
    """Runner class for model inference and evaluation."""

    def __init__(self, cfg: DictConfig):
        """Initialize the evaluation runner.

        Args:
            cfg: Inference configuration.
        """
        ckpt_path = cfg.inference.ckpt_path
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, 'config.yaml'))

        # Set-up config (user config overrides checkpoint config)
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_struct(ckpt_cfg, False)
        cfg = OmegaConf.merge(ckpt_cfg, cfg)
        cfg.experiment.checkpointer.dirpath = './'

        self._cfg = cfg
        self._exp_cfg = cfg.experiment
        self._infer_cfg = cfg.inference
        self._data_cfg = cfg.data
        self._rng = np.random.default_rng(self._infer_cfg.seed)

        if self._infer_cfg.seed is not None:
            log.info(f'Setting seed to {self._infer_cfg.seed}')
            self._set_seed(self._infer_cfg.seed)

        self.device = GPUtil.getAvailable(order='memory')[0]

        # Set-up output directory only on rank 0
        local_rank = os.environ.get('LOCAL_RANK', 0)
        if local_rank == 0:
            inference_dir = self._infer_cfg.inference_dir
            self._exp_cfg.inference_dir = inference_dir
            os.makedirs(inference_dir, exist_ok=True)
            log.info(f'Saving results to {inference_dir}')

            # Save config
            config_path = os.path.join(inference_dir, 'inference.yaml')
            with open(config_path, 'w') as f:
                OmegaConf.save(config=self._cfg, f=f)
            log.info(f'Saving inference config to {config_path}')

        # Load checkpoint and initialize module
        log.info(f'Loading checkpoint from {ckpt_path}')
        self._flow_module = FlowModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            cfg=self._cfg,
            map_location='cpu',
            weights_only=False  # Required for checkpoints with OmegaConf objects
        )
        self._flow_module.eval()
        self._flow_module.to(self.device)
        self._flow_module._infer_cfg = self._infer_cfg
        log.info(pl.utilities.model_summary.ModelSummary(self._flow_module))

    @property
    def inference_dir(self):
        """Return the inference output directory."""
        return self._flow_module.inference_dir

    def _set_seed(self, seed: int = 42):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def run_sampling(self):
        """Run inference on the test dataset."""
        devices = GPUtil.getAvailable(
            order='memory', limit=8)[:self._infer_cfg.num_gpus]
        log.info(f"Using devices: {devices}")
        log.info(f'Evaluating {self._infer_cfg.task}')

        eval_dataset = MCDataset(
            cache_path=os.path.join(self._data_cfg.cache_dir, 'test.pt'),
            dataset_cfg=self._data_cfg,
            is_training=False
        )
        dataloader = MCDatamodule(
            data_cfg=self._data_cfg,
            train_dataset=None,
            valid_dataset=None,
            test_dataset=eval_dataset,
        ).test_dataloader()

        trainer = Trainer(
            accelerator="gpu",
            strategy="ddp",
            devices='auto',
            logger=False,  # Disable Lightning logging during inference
        )

        with torch.inference_mode():
            predictions = trainer.predict(
                self._flow_module,
                dataloaders=dataloader,
                return_predictions=True
            )

        # Combine predictions from all batches
        combined_results = self._combine_predictions(predictions)

        # Save results (rank 0 only)
        if os.environ.get('LOCAL_RANK', '0') == '0':
            num_samples = combined_results['cart_coords'].shape[0]
            save_path = os.path.join(
                self._infer_cfg.output_dir, f'predictions_{num_samples}.pt')
            os.makedirs(self._infer_cfg.output_dir, exist_ok=True)
            torch.save(combined_results, save_path)
            log.info(f"Saved predictions to {save_path}")

    def _combine_predictions(self, predictions):
        """Combine predictions from multiple batches.

        Args:
            predictions: List of prediction dictionaries from each batch.

        Returns:
            Dictionary with combined predictions.
        """
        save_trajectories = getattr(self._infer_cfg, 'save_trajectories', False)

        combined_results = {
            'cart_coords': torch.cat([p['cart_coords'] for p in predictions], dim=1),
            'atom_types': torch.cat([p['atom_types'] for p in predictions], dim=1),
            'lattices': torch.cat([p['lattices'] for p in predictions], dim=1),
            'num_atoms': torch.cat([p['num_atoms'] for p in predictions], dim=1),
            'pred_trans': torch.cat([p['pred_trans'] for p in predictions], dim=1),
            'pred_rotmats': torch.cat([p['pred_rotmats'] for p in predictions], dim=1),
            'num_bbs': torch.cat([p['num_bbs'] for p in predictions], dim=1),
        }

        if save_trajectories:
            keys = list(predictions[0]['mc_trajs'].keys())
            combined_results['mc_trajs'] = {
                k: torch.cat([p['mc_trajs'][k] for p in predictions], dim=2)
                for k in keys if k != 'timesteps'
            }

        # Concatenate ground truth data
        combined_results['gt_data_batch'] = {
            'gt_coords': torch.cat(
                [p['gt_data_batch']['gt_coords'] for p in predictions], dim=0),
            'atom_types': torch.cat(
                [p['gt_data_batch']['atom_types'] for p in predictions], dim=0),
            'lattice_1': torch.cat(
                [p['gt_data_batch']['lattice_1'] for p in predictions], dim=0),
            'trans_1': torch.cat(
                [p['gt_data_batch']['trans_1'] for p in predictions], dim=0),
            'rotmats_1': torch.cat(
                [p['gt_data_batch']['rotmats_1'] for p in predictions], dim=0),
            'num_atoms': torch.cat(
                [p['gt_data_batch']['num_atoms'] for p in predictions], dim=0)
        }

        return combined_results


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs"),
    config_name="inference"
)
def run(cfg: DictConfig) -> None:
    """Main entry point for inference."""
    log.info(f'Starting inference with {cfg.inference.num_gpus} GPUs')
    start_time = time.time()
    sampler = EvalRunner(cfg)
    sampler.run_sampling()
    elapsed_time = time.time() - start_time
    log.info(f'Finished in {elapsed_time:.2f}s')


if __name__ == '__main__':
    run()
