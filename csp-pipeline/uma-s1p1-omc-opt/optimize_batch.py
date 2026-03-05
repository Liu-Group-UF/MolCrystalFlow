#!/usr/bin/env python3
"""
Rigid-body optimization batch script using UMA-OMC model.

Usage:
    python optimize_batch.py --start 0 --end 50 --input pred_combined.xyz
"""
import warnings
warnings.filterwarnings("ignore")
from fairchem.core import pretrained_mlip, FAIRChemCalculator
import argparse
from ase.io import read, write
from ase.optimize import BFGS
from utils.rigid_constraint import add_rigid_body_constraints
import os

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="omc")

def main(start, end, input_xyz):
    # Load only the requested slice of structures
    atoms_list = read(input_xyz, format="extxyz", index=f"{start}:{end}")

    optimized_structures = []

    for i, atoms in enumerate(atoms_list, start=start):
        # Clear constraints
        atoms.set_constraint([])

        # Add rigid body constraints (uses bb_indices or bb_idx)
        constraints = add_rigid_body_constraints(atoms)
        atoms.set_constraint(constraints)

        # Attach calculator
        atoms.calc = calc
        os.makedirs("log-files", exist_ok=True)
        
        # Run optimization (100 steps)
        optimizer = BFGS(atoms, logfile=f"log-files/opt_{i}.log")
        optimizer.run(steps=100)

        # Save the final optimized structure
        atoms.set_constraint([])
        optimized_structures.append(atoms.copy())

    # Write all optimized structures for this chunk
    os.makedirs("opt-results", exist_ok=True)
    outfile = f"./opt-results/pred_omc25_100step_opt_{start:05d}_{end:05d}.extxyz"
    write(outfile, optimized_structures, format="extxyz")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True, help="Start index of slice")
    parser.add_argument("--end", type=int, required=True, help="End index of slice")
    parser.add_argument("--input", type=str, default="./pred_sc9_er3.xyz", help="Input XYZ file")
    args = parser.parse_args()

    main(args.start, args.end, args.input)

