import os
import sys
sys.path.append("/home/c.zeng/blue_mingjieliu/c.zeng/research-projects/MolGen/ref-codebase/molcrystalflow")
import glob
import json
import torch
from omegaconf import OmegaConf
from experiments.inference_backup import EvalRunner  # your EvalRunner class
from data.utils import visualize_predictions
from experiments.xyz_matching import match_structures_multi_samples
# from common.utils import PROJECT_ROOT
os.environ["PROJECT_ROOT"] = "/home/c.zeng/blue_mingjieliu/c.zeng/research-projects/MolGen/ref-codebase/molcrystalflow"
def run_full_inference(
    ckpt_path,
    cache_dir,
    scaling=0,
    exp_rate=20,
    num_samples=3,
    timesteps=50,
    inference_dir="inference_results",
    save_dir="xyz-files",
    seed=42,
    num_gpus=1,
    num_cpus=24,
    stol=1,
    ltol=0.3,
    angle_tol=10,
):
    """
    Run full inference pipeline with tuned parameters and compute matching rate.

    Args:
        ckpt_path (str): Path to model checkpoint (.ckpt).
        cache_dir (str): Path to preprocessed data.
        scaling (float): interpolant.trans.scaling
        exp_rate (float): interpolant.rots.exp_rate
        num_samples (int): number of samples
        timesteps (int): inference timesteps
        inference_dir (str): directory for .pt files
        save_dir (str): directory to save .xyz files
        seed (int): random seed
        num_gpus (int): number of GPUs
        num_cpus (int): CPUs for structure matching
        stol, ltol, angle_tol: matching tolerances

    Returns:
        dict: results including matching rate and summary
    """
    suffix = f"sc{scaling:.0f}_er{exp_rate:.0f}"

    # Load base hydra config
    PROJECT_ROOT = "/home/c.zeng/blue_mingjieliu/c.zeng/research-projects/MolGen/ref-codebase/molcrystalflow"
    cfg_path = f"{PROJECT_ROOT}/configs/inference.yaml"
    inference_cfg = OmegaConf.load(cfg_path)
    ckpt_cfg = OmegaConf.load(os.path.join(ckpt_path, 'config.yaml'))
    cfg = OmegaConf.merge(ckpt_cfg, inference_cfg)
    # Override configs
    # cfg.inference.ckpt_path = ckpt_path
    cfg.inference.num_gpus = num_gpus
    cfg.inference.num_samples = num_samples
    cfg.inference.seed = seed
    cfg.inference.inference_dir = inference_dir

    cfg.data.cache_dir = cache_dir
    cfg.interpolant.sampling.num_timesteps = timesteps
    cfg.interpolant.trans.scaling = scaling
    cfg.interpolant.rots.sample_schedule = "exp"
    cfg.interpolant.rots.exp_rate = exp_rate

    # --- Step 1: Run inference ---
    sampler = EvalRunner(cfg)
    sampler.run_sampling()

    # --- Step 2: Find the generated .pt file ---
    pt_file = os.path.join(
        cfg.inference.inference_dir,
        f"predictions_{cfg.inference.num_samples}.pt"
    )
    if not os.path.exists(pt_file):
        raise FileNotFoundError(f"No predictions file found at {pt_file}")

    # --- Step 3: Convert to xyz ---
    os.makedirs(save_dir, exist_ok=True)
    save_info = visualize_predictions(
        pred_file=pt_file,
        gt_xyz=f'gt_{suffix}.xyz',
        pred_xyz=f'pred_{suffix}.xyz',
        save_dir=save_dir,
        cg_xyz=f'cg_pred_{suffix}.xyz',
        cg_gt_xyz=f'cg_gt_{suffix}.xyz'
    )
    num_samples = save_info["num_samples"]

    # --- Step 4: Matching ---
    rmsd_results, mr, summary = match_structures_multi_samples(
        gt_file=f"{save_dir}/gt_{suffix}.xyz",
        pred_file=f"{save_dir}/pred_{suffix}.xyz",
        num_cpus=num_cpus,
        stol=stol,
        ltol=ltol,
        angle_tol=angle_tol,
        num_samples=num_samples
    )

    results = {
        "suffix": suffix,
        "pt_file": pt_file,
        "num_samples": num_samples,
        "matching_rate": mr,
        "summary": summary,
    }

    # Save results
    os.makedirs("results", exist_ok=True)
    with open(f"results/results_{suffix}.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
