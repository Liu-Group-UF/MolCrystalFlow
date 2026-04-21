"""
MolCrystalFlow Generation Module.

This module contains the core logic for generating molecular crystal structures
using MolCrystalFlow.

Main components:
- packing_gen: Packing generation functions (feature extraction, inference, filtering)

Usage:
    from molcrystalflow_gen.packing_gen import (
        generate_crystal_structures,
        load_model,
        extract_molecule_features,
        run_inference,
    )
"""

from .packing_gen import (
    # High-level API
    generate_crystal_structures,
    # Model loading
    load_model,
    # Feature extraction
    extract_molecule_features,
    compute_auxiliary_features,
    # Inference
    run_inference,
    create_inference_batch,
    results_to_atoms,
    # Filtering
    filter_structures_by_overlap,
    analyze_structure,
    # Utilities
    set_seed,
)

__all__ = [
    "generate_crystal_structures",
    "load_model",
    "extract_molecule_features",
    "compute_auxiliary_features",
    "run_inference",
    "create_inference_batch",
    "results_to_atoms",
    "filter_structures_by_overlap",
    "analyze_structure",
    "set_seed",
]
