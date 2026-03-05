#!/usr/bin/env python
"""
Run lattice volume comparison analysis on ground truth and predicted XYZ files.

This script:
1. Loads ground truth and predicted XYZ files
2. Extracts lattice volumes from both sets
3. Computes volume deviations (RMAD, per-structure deviations)
4. Generates publication-quality figures (KDE parity plot, boxplot)
5. Saves results to PDF and summary statistics to JSON

Usage:
    python experiments/run_lattice_volume_analysis.py \
        --gt_file path/to/gt.xyz \
        --pred_file path/to/pred.xyz \
        --num_samples K \
        --output_dir path/to/output
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures

from ase.io import read
from ase.cell import Cell
from ase.geometry.cell import cell_to_cellpar

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)


def extract_volumes(xyz_file, num_mols=None):
    """Extract volumes from all structures in an XYZ file."""
    if num_mols is not None:
        structures = read(xyz_file, index=f':{num_mols}')
    else:
        structures = read(xyz_file, index=':')
    return [atoms.get_volume() for atoms in structures]


def extract_volumes_by_samples(xyz_file, num_samples, num_gt_structures):
    """
    Extract volumes grouped by samples from a multi-sample XYZ file.
    
    Args:
        xyz_file (str): Path to XYZ file containing multiple samples
        num_samples (int): Number of samples per ground truth structure
        num_gt_structures (int): Number of ground truth structures
    
    Returns:
        list: List of volume lists, one for each sample
    """
    all_structures = read(xyz_file, index=':')
    expected_total = num_samples * num_gt_structures
    
    if len(all_structures) != expected_total:
        raise ValueError(f"Expected {expected_total} structures ({num_gt_structures} GT × {num_samples} samples), "
                        f"but found {len(all_structures)}")
    
    # Group structures by sample
    samples_volumes = []
    for sample_idx in range(num_samples):
        sample_volumes = []
        for gt_idx in range(num_gt_structures):
            struct_idx = gt_idx + sample_idx * num_gt_structures
            volume = all_structures[struct_idx].get_volume()
            sample_volumes.append(volume)
        samples_volumes.append(sample_volumes)
    
    return samples_volumes


def get_lattice_parameters(xyz_file, num_mols=None):
    """Extract lattice parameters from an XYZ file"""
    if num_mols is not None:
        configs = read(xyz_file, index=f':{num_mols}')
    else:
        configs = read(xyz_file, index=':')
    
    params = {
        'a': [], 'b': [], 'c': [], 
        'alpha': [], 'beta': [], 'gamma': []
    }
    
    for atoms in configs:
        cell = Cell(atoms.cell)
        reduced_cell = cell.niggli_reduce()[0]
        cellpar = cell_to_cellpar(reduced_cell)
        
        for i, key in enumerate(['a', 'b', 'c', 'alpha', 'beta', 'gamma']):
            params[key].append(cellpar[i])
    
    return {k: np.array(v) for k, v in params.items()}


def plot_volume_parity_kde(vols_ref, samples_volumes, num_samples, save_path, bw_adjust=1.0, std_across_samples=None, cmap="viridis"):
    """
    Plot parity KDE of predicted vs ground-truth volumes with RMAD and colormap for std.
    """
    import seaborn as sns
    # Collect all volumes
    V_ref_all = []
    V_pred_all = []
    for sample_idx in range(num_samples):
        vols_pred = samples_volumes[sample_idx]
        for v_ref, v_pred in zip(vols_ref, vols_pred):
            V_ref_all.append(v_ref)
            V_pred_all.append(v_pred)
    V_ref_all = np.asarray(V_ref_all, dtype=float)
    V_pred_all = np.asarray(V_pred_all, dtype=float)
    # RMAD (%): mean(|pred-ref|/ref)*100
    eps = 1e-12
    rel_abs_dev = np.abs(V_pred_all - V_ref_all) / np.maximum(np.abs(V_ref_all), eps)
    rmad = 100.0 * np.mean(rel_abs_dev)
    std_rmad = 100.0 * np.std(rel_abs_dev)
    # Plot
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.4)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    # KDE plot with colormap
    cmap_obj = plt.get_cmap(cmap)
    if len(V_ref_all) > 1:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        sns.kdeplot(
            x=V_ref_all,
            y=V_pred_all,
            ax=ax,
            fill=True,
            cmap=cmap_obj,
            levels=50,
            thresh=0.01,
            bw_adjust=bw_adjust,
        )

        # Inset colorbar (bottom-right), matching requested style.
        if ax.collections:
            ref_collection = ax.collections[-1]
            sm = ScalarMappable(
                norm=ref_collection.norm if ref_collection.norm is not None else Normalize(vmin=0.0, vmax=1.0),
                cmap=ref_collection.cmap if ref_collection.cmap is not None else cmap_obj,
            )
        else:
            sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap_obj)
        sm.set_array([])

        cax = ax.inset_axes([0.62, 0.05, 0.30, 0.035])
        cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.ax.set_xticks([])
        cbar.ax.set_yticks([])
        cbar.outline.set_linewidth(0.6)
        cax.text(
            0.5, 1.55, "Increasing density",
            transform=cax.transAxes,
            ha="center", va="bottom",
            fontsize=16
        )
    # Parity line
    min_vol = min(V_ref_all.min(), V_pred_all.min()) * 0.9
    max_vol = max(V_ref_all.max(), V_pred_all.max()) * 1.1
    ax.plot([min_vol, max_vol], [min_vol, max_vol], ls="--", lw=1.5, color="black", zorder=10)
    ax.set_xlim(min_vol, max_vol)
    ax.set_ylim(min_vol, max_vol)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Ground-truth lattice volume [$\mathrm{\AA}^3$]", fontsize=14)
    ax.set_ylabel(r"Generated-structure lattice volume [$\mathrm{\AA}^3$]", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    # Annotate RMAD and std (use std_across_samples if provided)
    if std_across_samples is not None:
        std_text = "RMAD: %.2f±%.2f%%" % (rmad, std_across_samples)
    else:
        std_text = "RMAD: %.2f±%.2f%%" % (rmad, std_rmad)
    ax.text(0.05, 0.95, std_text, transform=ax.transAxes,
            ha="left", va="top", fontsize=16,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return rmad, std_rmad


def plot_lattice_deviation_boxplot(
    boxplot_data,
    save_path="lattice_volume_boxplot.pdf",
    max_points=100,
):
    """
    Box plot of relative volume deviations per structure (Nature style).
    """
    num_structures = len(boxplot_data)
    # Optional downsampling for readability
    if num_structures > max_points:
        idx = np.linspace(0, num_structures - 1, max_points).astype(int)
        boxplot_data = [np.abs(boxplot_data[i]) for i in idx]
        x_labels = idx
    else:
        x_labels = np.arange(num_structures)
    fig, ax = plt.subplots(figsize=(max(6, 0.12 * len(boxplot_data)), 4.5), dpi=300)
    bp = ax.boxplot(
        boxplot_data,
        patch_artist=True,
        widths=0.6,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2),
        boxprops=dict(edgecolor="black", linewidth=1.2),
    )
    # Nature-style color (soft blue)
    for box in bp["boxes"]:
        box.set_facecolor("#4C72B0")
        box.set_alpha(0.75)
    # Zero reference line
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axhline(5.0, color="brown", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_xlabel("Structure index", fontsize=12)
    ax.set_ylabel("Relative lattice volume deviation [%]", fontsize=12)
    ax.set_xticks(np.arange(1, len(x_labels) + 1))
    ax.set_xticklabels(x_labels, fontsize=8, rotation=90)
    ax.tick_params(axis="y", labelsize=11)
    # Clean Nature-style spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_lattice_comparisons(gt_file, pred_file, save_path, num_gt):
    """Create histogram and scatter plots for lattice parameters."""
    params1 = get_lattice_parameters(gt_file, num_mols=num_gt)
    params2 = get_lattice_parameters(pred_file, num_mols=num_gt)
    labels = ['Ground Truth', 'Predicted']
    param_names = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
    titles = ['a (Å)', 'b (Å)', 'c (Å)', 'α (°)', 'β (°)', 'γ (°)']

    fig, axs = plt.subplots(4, 3, figsize=(15, 20))

    # First row: histograms for lengths
    for ax, param_name, title in zip(axs[0], param_names[:3], titles[:3]):
        ax.hist(params1[param_name], bins=30, alpha=0.6, color='blue', 
                label=labels[0], edgecolor='black', density=True)
        ax.hist(params2[param_name], bins=30, alpha=0.6, color='red',
                label=labels[1], edgecolor='black', density=True)
        ax.set_xlabel(title)
        ax.set_ylabel('Density')
        ax.legend()

    # Second row: histograms for angles
    for ax, param_name, title in zip(axs[1], param_names[3:], titles[3:]):
        ax.hist(params1[param_name], bins=30, alpha=0.6, color='blue',
                label=labels[0], edgecolor='black', density=True)
        ax.hist(params2[param_name], bins=30, alpha=0.6, color='red',
                label=labels[1], edgecolor='black', density=True)
        ax.set_xlabel(title)
        ax.set_ylabel('Density')
        ax.legend()

    # Third row: scatter plots for lengths
    for ax, param_name, title in zip(axs[2], param_names[:3], titles[:3]):
        ax.scatter(params1[param_name], params2[param_name], alpha=0.5)
        minv = min(params1[param_name].min(), params2[param_name].min())
        maxv = max(params1[param_name].max(), params2[param_name].max())
        ax.plot([minv, maxv], [minv, maxv], 'k--', alpha=0.5)
        ax.set_xlabel(f'{title} (Ground Truth)')
        ax.set_ylabel(f'{title} (Predicted)')

    # Fourth row: scatter plots for angles
    for ax, param_name, title in zip(axs[3], param_names[3:], titles[3:]):
        ax.scatter(params1[param_name], params2[param_name], alpha=0.5)
        minv = min(params1[param_name].min(), params2[param_name].min())
        maxv = max(params1[param_name].max(), params2[param_name].max())
        ax.plot([minv, maxv], [minv, maxv], 'k--', alpha=0.5)
        ax.set_xlabel(f'{title} (Ground Truth)')
        ax.set_ylabel(f'{title} (Predicted)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Print statistics of differences
    stats = {}
    print("\nDifference Statistics (Predicted - Ground Truth):")
    for param_name in param_names:
        diff = np.abs(params2[param_name] - params1[param_name])
        stats[param_name] = {
            'mean_diff': float(np.mean(diff)),
            'std_diff': float(np.std(diff)),
            'max_diff': float(np.max(diff)),
            'rmsd': float(np.sqrt(np.mean(diff**2)))
        }
        print(f"\n{param_name}:")
        print(f"  Mean diff: {np.mean(diff):.3f}")
        print(f"  Std diff:  {np.std(diff):.3f}")
        print(f"  Max diff:  {np.max(np.abs(diff)):.3f}")
        print(f"  RMSD:      {np.sqrt(np.mean(diff**2)):.3f}")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Run lattice volume comparison analysis for MolCrystalFlow results."
    )
    parser.add_argument(
        "--gt_file", 
        type=str, 
        required=True,
        help="Path to ground truth XYZ file"
    )
    parser.add_argument(
        "--pred_file", 
        type=str, 
        required=True,
        help="Path to predicted XYZ file"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        required=True,
        help="Number of samples per ground truth structure"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Output directory for figures (default: same as gt_file directory)"
    )
    parser.add_argument(
        "--prefix", 
        type=str, 
        default="lattice_volume",
        help="Prefix for output files (default: lattice_volume)"
    )
    parser.add_argument(
        "--max_boxplot_points", 
        type=int, 
        default=60,
        help="Maximum number of structures to show in boxplot (default: 60)"
    )
    parser.add_argument(
        "--kde_cmap",
        type=str,
        default="viridis",
        help="Colormap for KDE parity plot (default: viridis)"
    )
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.gt_file):
        raise FileNotFoundError(f"Ground truth file not found: {args.gt_file}")
    if not os.path.exists(args.pred_file):
        raise FileNotFoundError(f"Prediction file not found: {args.pred_file}")

    # Set up output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.gt_file)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Lattice Volume Comparison Analysis")
    print("=" * 60)
    print(f"Ground truth file: {args.gt_file}")
    print(f"Prediction file: {args.pred_file}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    # Extract volumes
    print("\n[Step 1] Extracting volumes from XYZ files...")
    vols_ref = extract_volumes(args.gt_file)
    num_gt = len(vols_ref)
    print(f"  Number of ground truth structures: {num_gt}")

    samples_volumes = extract_volumes_by_samples(args.pred_file, args.num_samples, num_gt)
    print(f"  Number of predicted structures: {num_gt * args.num_samples}")

    # Compute deviations
    print("\n[Step 2] Computing volume deviations...")
    boxplot_data = []
    for gt_idx in range(num_gt):
        ref_vol = vols_ref[gt_idx]
        rel_devs = [
            100 * (samples_volumes[s][gt_idx] - ref_vol) / ref_vol
            for s in range(args.num_samples)
        ]
        boxplot_data.append(np.array(rel_devs))

    # Summary statistics
    all_devs = np.concatenate(boxplot_data)
    abs_devs = np.abs(all_devs)
    mean_abs_dev = abs_devs.mean()

    # Compute mean deviation for each sample (across all structures)
    sample_mean_devs = [
        np.mean([boxplot_data[gt][s] for gt in range(num_gt)])
        for s in range(args.num_samples)
    ]
    std_abs_dev = float(np.std(sample_mean_devs))  # std of mean deviations across samples

    # Per-sample statistics
    sample_stats = []
    for s in range(args.num_samples):
        sample_devs = np.abs([boxplot_data[gt][s] for gt in range(num_gt)])
        sample_stats.append({
            'sample': s,
            'mean_abs_deviation': float(sample_devs.mean()),
            'std_abs_deviation': float(sample_devs.std()),
            'max_abs_deviation': float(sample_devs.max())
        })

    print(f"  Mean |relative volume deviation|: {mean_abs_dev:.4f}%")
    print(f"  Std  of mean volume deviation across samples: {std_abs_dev:.4f}%")

    # Generate figures
    print("\n[Step 3] Generating figures...")
    
    # KDE parity plot
    kde_path = os.path.join(args.output_dir, f"{args.prefix}_kde_parity.pdf")
    rmad, std_rmad = plot_volume_parity_kde(
        vols_ref, samples_volumes, args.num_samples, kde_path,
        std_across_samples=std_abs_dev, cmap=args.kde_cmap
    )
    print(f"  KDE parity plot saved: {kde_path}")
    print(f"  RMAD: {rmad:.2f}±{std_abs_dev:.2f}%")

    # Boxplot
    boxplot_path = os.path.join(args.output_dir, f"{args.prefix}_deviation_boxplot.pdf")
    plot_lattice_deviation_boxplot(boxplot_data, boxplot_path, max_points=args.max_boxplot_points)
    print(f"  Deviation boxplot saved: {boxplot_path}")

    # Lattice parameter comparison (first sample vs GT)
    lattice_params_path = os.path.join(args.output_dir, f"{args.prefix}_params_comparison.pdf")
    lattice_stats = plot_lattice_comparisons(args.gt_file, args.pred_file, lattice_params_path, num_gt)
    print(f"  Lattice params comparison saved: {lattice_params_path}")

    # Save summary statistics
    print("\n[Step 4] Saving summary statistics...")
    summary = {
        'num_gt_structures': num_gt,
        'num_samples': args.num_samples,
        'rmad_percent': float(rmad),
        'mean_abs_deviation_percent': float(mean_abs_dev),
        'std_abs_deviation_percent': float(std_abs_dev),
        'max_abs_deviation_percent': float(abs_devs.max()),
        'min_abs_deviation_percent': float(abs_devs.min()),
        'per_sample_stats': sample_stats,
        'lattice_parameter_stats': lattice_stats,
        'output_files': {
            'kde_parity_plot': kde_path,
            'deviation_boxplot': boxplot_path,
            'lattice_params_comparison': lattice_params_path
        }
    }

    summary_path = os.path.join(args.output_dir, f"{args.prefix}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"  Summary saved: {summary_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"RMAD: {rmad:.2f}±{std_abs_dev:.2f}%")
    print(f"Mean |relative deviation|: {mean_abs_dev:.4f}%")
    print(f"Std  |relative deviation|: {std_abs_dev:.4f}%")
    print(f"\nOutput files saved in: {args.output_dir}")
    print(f"  - {os.path.basename(kde_path)}")
    print(f"  - {os.path.basename(boxplot_path)}")
    print(f"  - {os.path.basename(lattice_params_path)}")
    print(f"  - {os.path.basename(summary_path)}")


if __name__ == "__main__":
    main()
