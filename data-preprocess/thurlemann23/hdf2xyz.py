#!/usr/bin/env python
"""
Convert Thurlemann2023 HDF5 crystal dataset to extended XYZ format.

HDF5 dataset structure (per CSD code):
    - 'z': Number of molecules in the unit cell
    - 'elements': Atomic elements symbols of the unit cell (N,)
    - 'elements_monomer': Atomic elements symbols of the molecule (N_mono,)
    - 'coordinates': Coordinates of atoms in the unit cell [Angstrom], (5, N, 3)
    - 'coordinates_monomer': Coordinates of isolated molecule [Angstrom], (5, N_mono, 3)
    - 'cells': Vectors defining the unit cell [Angstrom], (5, 3, 3)
    - 'total_energies': Total energy of the unit cell [kJ/mol], (5,)
    - 'intermolecular_energy': Intermolecular energy [kJ/mol], (5,)
    - 'total_energy_monomer': Total energy of isolated molecule [kJ/mol], scalar

Usage:
    python hdf2xyz.py --input crystal_dataset.hdf5 --output molecule_crystals.extxyz
    python hdf2xyz.py --input crystal_dataset.hdf5 --output molecule_crystals.extxyz --max_structures 100
"""
import argparse
from pathlib import Path

import h5py
from ase import Atoms
from ase.io import write
from tqdm import tqdm


def convert_hdf_to_xyz(
    input_path: str,
    output_path: str,
    max_structures: int = None,
    verbose: bool = True,
) -> int:
    """
    Convert HDF5 crystal dataset to extended XYZ format.

    Args:
        input_path: Path to input HDF5 file.
        output_path: Path to output extended XYZ file.
        max_structures: Maximum number of CSD entries to process (None for all).
        verbose: Whether to print progress information.

    Returns:
        Number of structures written.
    """
    molecules = []

    with h5py.File(input_path, "r") as f:
        csd_codes = list(f.keys())
        if max_structures is not None:
            csd_codes = csd_codes[:max_structures]

        if verbose:
            print(f"Processing {len(csd_codes)} CSD entries from {input_path}")
            iterator = tqdm(csd_codes, desc="Converting")
        else:
            iterator = csd_codes

        for csd_code in iterator:
            mol = f[csd_code]

            # Decode element symbols
            elements = mol["elements"][()]
            if isinstance(elements[0], bytes):
                elements = [e.decode("utf-8") for e in elements]

            elements_monomer = mol["elements_monomer"][()]
            if isinstance(elements_monomer[0], bytes):
                elements_monomer = [e.decode("utf-8") for e in elements_monomer]

            num_molecules = int(mol["z"][()])
            energy_monomer = float(mol["total_energy_monomer"][()])

            # Each CSD entry has 5 configurations (scaled lattices)
            num_configs = mol["cells"].shape[0]
            for i in range(num_configs):
                cell = mol["cells"][i]
                positions = mol["coordinates"][i]
                energy = float(mol["total_energies"][i])
                intermolecular_energy = float(mol["intermolecular_energy"][i])

                atoms = Atoms(symbols=elements, positions=positions, cell=cell, pbc=True)

                # Store metadata in info dict
                atoms.info["csd_ref_code"] = csd_code
                atoms.info["energy"] = energy
                atoms.info["intermolecular_energy"] = intermolecular_energy
                atoms.info["z_value"] = num_molecules
                atoms.info["energy_monomer"] = energy_monomer
                atoms.info["elements_monomer"] = " ".join(elements_monomer)
                atoms.info["config_index"] = i

                molecules.append(atoms)

    # Write all structures to extended XYZ
    write(output_path, molecules, format="extxyz")

    if verbose:
        print(f"Wrote {len(molecules)} structures to {output_path}")

    return len(molecules)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Thurlemann2023 HDF5 crystal dataset to extended XYZ format."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input HDF5 file (e.g., crystal_dataset.hdf5)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="molecule_crystals.extxyz",
        help="Path to output extended XYZ file (default: molecule_crystals.extxyz)",
    )
    parser.add_argument(
        "--max_structures",
        "-n",
        type=int,
        default=None,
        help="Maximum number of CSD entries to process (default: all)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    convert_hdf_to_xyz(
        input_path=str(input_path),
        output_path=args.output,
        max_structures=args.max_structures,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
