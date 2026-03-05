#!/usr/bin/env python
"""
Exploratory Data Analysis (EDA) for molecular crystal datasets.

This script analyzes filtered molecular crystal datasets and generates four
publication-quality figures:

1. Element frequency bar chart (element_freq.png)
2. Atoms vs molecules violin plot (atoms_vs_monomers_violin.png)
3. Atoms per molecule histogram (natoms_hist.png)
4. Molecules per crystal frequency bar chart (molecules_per_crystal_freq.png)

Usage:
    # For OMC25-MCF dataset
    python run_eda.py --dataset omc25-mcf
    
    # For Thurlemann23 dataset
    python run_eda.py --dataset thurlemann23
    
    # Custom paths
    python run_eda.py \\
        --molcrystals path/to/valid_molcrystals.extxyz \\
        --coarse_grained path/to/coarse_grained.extxyz \\
        --output_dir path/to/output

Output:
    <output_dir>/eda_figures/
    ├── element_freq.png
    ├── atoms_vs_monomers_violin.png
    ├── natoms_hist.png
    ├── molecules_per_crystal_freq.png
    └── eda_summary.txt
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available, using matplotlib for violin plot")

from ase.io import read


# =============================================================================
# Plotting Utilities
# =============================================================================


def setup_nature_style():
    """Apply Nature-style plot settings."""
    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 16,
        "figure.figsize": (5, 4),
        "axes.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.4,
    })


def k_m_formatter(x, pos):
    """Format large numbers as k/M."""
    if x >= 1e6:
        return f'{x/1e6:.0f}M'
    elif x >= 1e4:
        return f'{x/1e3:.0f}k'
    else:
        return f'{int(x)}'


def fmt_km(v):
    """Format value as k/M string."""
    if v >= 1e6:
        return f'{v/1e6:.1f}M'
    elif v >= 1e4:
        return f'{v/1e3:.1f}k'
    else:
        return f'{int(v)}'


def style_axes(ax):
    """Apply Nature-style axis styling."""
    ax.yaxis.set_major_formatter(FuncFormatter(k_m_formatter))
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1.2)
    ax.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.4)


# =============================================================================
# EDA Functions
# =============================================================================


def analyze_elements(molcrystals):
    """Analyze element distribution in the dataset."""
    unique_elements = set()
    elements = []
    
    for atoms in molcrystals:
        symbols = atoms.get_chemical_symbols()
        elements.extend(symbols)
        unique_elements.update(symbols)
    
    ele_counter = Counter(elements)
    total_count = sum(ele_counter.values())
    ele_ratios = {key: value / total_count for key, value in ele_counter.items()}
    
    return ele_counter, ele_ratios, unique_elements


def analyze_molecules(molcrystals, cg_crystals):
    """Analyze molecule counts and sizes."""
    natoms_in_monomer = []
    n_monomers = []
    
    for mc, cg in zip(molcrystals, cg_crystals):
        n_mol = len(cg)
        n_atoms = len(mc)
        atoms_per_mol = n_atoms / n_mol if n_mol > 0 else 0
        
        natoms_in_monomer.append(atoms_per_mol)
        n_monomers.append(n_mol)
    
    return natoms_in_monomer, n_monomers


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_element_frequency(ele_counter, output_path, verbose=True):
    """Plot element frequency bar chart."""
    labels = list(ele_counter.keys())
    values = list(ele_counter.values())
    
    # Sort by frequency (descending)
    sorted_pairs = sorted(zip(labels, values), key=lambda x: -x[1])
    labels, values = zip(*sorted_pairs)
    
    plt.figure(figsize=(5, 4))
    plt.bar(labels, values, color='#66C2A5')  # Nature-style muted green
    
    ax = plt.gca()
    style_axes(ax)
    
    plt.xlabel('Element', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"  Saved: {output_path}")


def plot_atoms_vs_monomers_violin(n_monomers, natoms_in_monomer, output_path, verbose=True):
    """Plot violin plot of atoms per molecule grouped by # molecules."""
    plt.figure(figsize=(5, 4.2))
    
    if HAS_SEABORN:
        sns.violinplot(
            x=n_monomers,
            y=natoms_in_monomer,
            color='#e64b35',  # Nature-style red
            linewidth=1.2,
            cut=0
        )
    else:
        # Fallback to matplotlib boxplot
        unique_n = sorted(set(n_monomers))
        data_by_n = {n: [] for n in unique_n}
        for n, a in zip(n_monomers, natoms_in_monomer):
            data_by_n[n].append(a)
        
        plt.boxplot(
            [data_by_n[n] for n in unique_n],
            labels=unique_n,
            patch_artist=True,
            boxprops=dict(facecolor='#e64b35', alpha=0.7)
        )
    
    ax = plt.gca()
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1.2)
    ax.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.4)
    
    plt.xlabel("Number of molecules", fontsize=20)
    plt.ylabel("# atoms per molecule", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"  Saved: {output_path}")


def plot_natoms_histogram(natoms_in_monomer, output_path, verbose=True):
    """Plot histogram of atoms per molecule."""
    plt.figure(figsize=(5, 4))
    
    plt.hist(
        natoms_in_monomer,
        bins=30,
        color='#4DBBD5',   # Nature-style blue
        alpha=0.85,
        edgecolor='black',
        linewidth=0.6
    )
    
    ax = plt.gca()
    style_axes(ax)
    
    plt.xlabel('# atoms per molecule', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"  Saved: {output_path}")


def plot_molecules_per_crystal(n_monomers, output_path, verbose=True):
    """Plot bar chart of molecules per crystal (Z distribution)."""
    counter_dict = Counter(n_monomers)
    
    # Sort by Z value
    labels = sorted(counter_dict.keys())
    values = [counter_dict[l] for l in labels]
    
    plt.figure(figsize=(5, 4))
    
    bars = plt.bar(
        labels,
        values,
        color='#f39b7f',  # Nature-style orange
        edgecolor='black',
        linewidth=0.8
    )
    
    # Annotate each bar
    for bar in bars:
        h = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2
        plt.text(
            x, h,
            fmt_km(h),
            ha='center', va='bottom',
            fontsize=16
        )
    
    plt.margins(y=0.15)  # avoid clipping top labels
    
    ax = plt.gca()
    style_axes(ax)
    
    plt.xlabel('# molecules in crystal', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"  Saved: {output_path}")


# =============================================================================
# Main EDA Function
# =============================================================================


def run_eda(
    molcrystals_path: str,
    coarse_grained_path: str,
    output_dir: str,
    dataset_name: str = "dataset",
    verbose: bool = True,
):
    """
    Run full EDA analysis and generate all figures.
    
    Args:
        molcrystals_path: Path to valid_molcrystals.extxyz
        coarse_grained_path: Path to coarse_grained.extxyz
        output_dir: Output directory for figures
        dataset_name: Name of the dataset (for summary)
        verbose: Print progress
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "eda_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    setup_nature_style()
    
    if verbose:
        print("=" * 60)
        print(f"EDA ANALYSIS: {dataset_name}")
        print("=" * 60)
    
    # Load data
    if verbose:
        print(f"\n[1/5] Loading data...")
        print(f"  Molcrystals: {molcrystals_path}")
        print(f"  Coarse-grained: {coarse_grained_path}")
    
    molcrystals = read(molcrystals_path, index=":")
    cg_crystals = read(coarse_grained_path, index=":")
    
    if verbose:
        print(f"  Loaded {len(molcrystals)} molecular crystals")
        print(f"  Loaded {len(cg_crystals)} coarse-grained structures")
    
    if len(molcrystals) != len(cg_crystals):
        print(f"  WARNING: Count mismatch! Using min({len(molcrystals)}, {len(cg_crystals)})")
        n = min(len(molcrystals), len(cg_crystals))
        molcrystals = molcrystals[:n]
        cg_crystals = cg_crystals[:n]
    
    # Analyze elements
    if verbose:
        print(f"\n[2/5] Analyzing elements...")
    
    ele_counter, ele_ratios, unique_elements = analyze_elements(molcrystals)
    
    if verbose:
        print(f"  Unique elements: {sorted(unique_elements)}")
        print(f"  Element counts: {dict(ele_counter)}")
    
    # Analyze molecules
    if verbose:
        print(f"\n[3/5] Analyzing molecules...")
    
    natoms_in_monomer, n_monomers = analyze_molecules(molcrystals, cg_crystals)
    
    values, freqs = np.unique(natoms_in_monomer, return_counts=True)
    mode_atoms = values[np.argmax(freqs)]
    z_distribution = Counter(n_monomers)
    
    if verbose:
        print(f"  Atoms per molecule: min={min(natoms_in_monomer):.0f}, max={max(natoms_in_monomer):.0f}, mode={mode_atoms:.0f}")
        print(f"  Z distribution: {dict(z_distribution)}")
    
    # Generate figures
    if verbose:
        print(f"\n[4/5] Generating figures...")
    
    plot_element_frequency(
        ele_counter,
        figures_dir / "element_freq.png",
        verbose=verbose
    )
    
    plot_atoms_vs_monomers_violin(
        n_monomers, natoms_in_monomer,
        figures_dir / "atoms_vs_monomers_violin.png",
        verbose=verbose
    )
    
    plot_natoms_histogram(
        natoms_in_monomer,
        figures_dir / "natoms_hist.png",
        verbose=verbose
    )
    
    plot_molecules_per_crystal(
        n_monomers,
        figures_dir / "molecules_per_crystal_freq.png",
        verbose=verbose
    )
    
    # Write summary
    if verbose:
        print(f"\n[5/5] Writing summary...")
    
    summary_path = figures_dir / "eda_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"EDA Summary: {dataset_name}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total structures: {len(molcrystals)}\n\n")
        
        f.write("Element Distribution:\n")
        for elem, count in sorted(ele_counter.items(), key=lambda x: -x[1]):
            ratio = ele_ratios[elem]
            f.write(f"  {elem}: {count:,} ({ratio*100:.2f}%)\n")
        f.write("\n")
        
        f.write("Atoms per Molecule:\n")
        f.write(f"  Min: {min(natoms_in_monomer):.0f}\n")
        f.write(f"  Max: {max(natoms_in_monomer):.0f}\n")
        f.write(f"  Mean: {np.mean(natoms_in_monomer):.1f}\n")
        f.write(f"  Mode: {mode_atoms:.0f}\n")
        f.write("\n")
        
        f.write("Z Distribution (molecules per crystal):\n")
        for z, count in sorted(z_distribution.items()):
            f.write(f"  Z={z}: {count:,} ({count/len(molcrystals)*100:.1f}%)\n")
        f.write("\n")
        
        f.write("Generated Figures:\n")
        f.write("  - element_freq.png\n")
        f.write("  - atoms_vs_monomers_violin.png\n")
        f.write("  - natoms_hist.png\n")
        f.write("  - molecules_per_crystal_freq.png\n")
    
    if verbose:
        print(f"  Saved: {summary_path}")
        print(f"\n" + "=" * 60)
        print(f"EDA COMPLETE")
        print(f"Output: {figures_dir}")
        print("=" * 60)
    
    return {
        "n_structures": len(molcrystals),
        "unique_elements": unique_elements,
        "element_counts": dict(ele_counter),
        "z_distribution": dict(z_distribution),
        "natoms_stats": {
            "min": min(natoms_in_monomer),
            "max": max(natoms_in_monomer),
            "mean": np.mean(natoms_in_monomer),
            "mode": mode_atoms,
        },
        "figures_dir": str(figures_dir),
    }


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Run EDA analysis on molecular crystal datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OMC25-MCF dataset
  python run_eda.py --dataset omc25-mcf

  # Thurlemann23 dataset
  python run_eda.py --dataset thurlemann23

  # Custom paths
  python run_eda.py \\
      --molcrystals ./filtered/valid_molcrystals.extxyz \\
      --coarse_grained ./filtered/coarse_grained.extxyz \\
      --output_dir ./my_output

Output:
  <output_dir>/eda_figures/
  ├── element_freq.png
  ├── atoms_vs_monomers_violin.png
  ├── natoms_hist.png
  ├── molecules_per_crystal_freq.png
  └── eda_summary.txt
        """,
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["omc25-mcf", "thurlemann23"],
        help="Pre-configured dataset (sets paths automatically)",
    )
    parser.add_argument(
        "--molcrystals",
        type=str,
        help="Path to valid_molcrystals.extxyz",
    )
    parser.add_argument(
        "--coarse_grained",
        type=str,
        help="Path to coarse_grained.extxyz",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory (default: dataset folder)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output",
    )
    
    args = parser.parse_args()
    
    # Get script directory for relative paths
    script_dir = Path(__file__).parent.resolve()
    
    # Set paths based on dataset
    if args.dataset == "omc25-mcf":
        base_dir = script_dir / "omc25-mcf" / "preprocessed" / "filtered"
        molcrystals_path = base_dir / "valid_molcrystals.extxyz"
        coarse_grained_path = base_dir / "coarse_grained.extxyz"
        output_dir = script_dir / "omc25-mcf" / "preprocessed"
        dataset_name = "OMC25-MCF"
        
    elif args.dataset == "thurlemann23":
        base_dir = script_dir / "thurlemann23" / "preprocessed"
        molcrystals_path = base_dir / "valid_molcrystals.extxyz"
        coarse_grained_path = base_dir / "coarse_grained.extxyz"
        output_dir = script_dir / "thurlemann23" / "preprocessed"
        dataset_name = "Thurlemann23"
        
    else:
        # Custom paths
        if not args.molcrystals or not args.coarse_grained:
            parser.error("Either --dataset or both --molcrystals and --coarse_grained are required")
        
        molcrystals_path = Path(args.molcrystals)
        coarse_grained_path = Path(args.coarse_grained)
        output_dir = Path(args.output_dir) if args.output_dir else molcrystals_path.parent.parent
        dataset_name = "Custom Dataset"
    
    # Override output_dir if specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
    
    # Run EDA
    run_eda(
        molcrystals_path=str(molcrystals_path),
        coarse_grained_path=str(coarse_grained_path),
        output_dir=str(output_dir),
        dataset_name=dataset_name,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
