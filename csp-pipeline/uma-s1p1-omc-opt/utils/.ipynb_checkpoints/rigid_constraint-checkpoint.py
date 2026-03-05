import numpy as np
from ase.constraints import FixConstraint
from collections import defaultdict

class RigidBodyWrapper(FixConstraint):
    """
    Custom rigid body constraint for ASE (compatible with trajectory I/O).
    Keeps a set of atoms rigid (bond lengths and angles fixed).
    """

    def __init__(self, indices):
        self.indices = np.array(indices, dtype=int)
        self._ref_positions = None
        self._ref_com = None

    def adjust_positions(self, atoms, newpositions):
        """Project new positions back to rigid-body motion."""
        if self._ref_positions is None:
            # Store reference geometry
            self._ref_positions = atoms.positions[self.indices].copy()
            self._ref_com = self._ref_positions.mean(axis=0)

        # Proposed positions
        new = newpositions[self.indices]

        # Compute COM shift
        com_new = new.mean(axis=0)
        com_ref = self._ref_com
        shift = com_new - com_ref

        # Remove COM before alignment
        ref = self._ref_positions - com_ref
        tgt = new - com_new

        # Optimal rotation (Kabsch algorithm)
        H = ref.T @ tgt
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:  # reflection fix
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Apply rigid transform
        new_rigid = (ref @ R.T) + com_new
        newpositions[self.indices] = new_rigid
        return newpositions

    def adjust_forces(self, atoms, forces):
        """Project forces so they don't break rigidity."""
        sub = self.indices
        com = atoms.positions[sub].mean(axis=0)
        pos = atoms.positions[sub] - com

        Ftot = forces[sub].sum(axis=0)
        Ttot = np.cross(pos, forces[sub]).sum(axis=0)

        denom = np.sum(pos**2)
        if denom < 1e-12:  # single-atom case
            forces[sub] = Ftot / len(sub)
        else:
            forces[sub] = Ftot / len(sub) + np.cross(Ttot, pos) / denom
        return forces

    # --- New methods for ASE trajectory writing ---
    def todict(self):
        return {"name": "RigidBodyWrapper", 'kwargs': {"indices": self.indices.tolist()}}

    def __repr__(self):
        return f"RigidBodyWrapper(indices={self.indices.tolist()})"

def add_rigid_body_constraints(atoms):
    bb_idx = atoms.arrays['bb_idx']
    rigid_bodies = defaultdict(list)
    for i, bb in enumerate(bb_idx):
        rigid_bodies[bb].append(i)
    constraints = []
    for body_atoms in rigid_bodies.values():
        constraints.append(RigidBodyWrapper(body_atoms))
    return constraints