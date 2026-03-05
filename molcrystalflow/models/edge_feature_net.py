import math

import torch
from torch import nn
from torch_scatter import scatter


class SinusoidsEmbedding(nn.Module):
    """
    Copied from diffcsp: https://github.com/txie-93/cdvae/blob/main/cdvae/pl_modules/cspnet.py
    """
    def __init__(self, n_frequencies=10, n_space=3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * torch.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RBFEmbedding(nn.Module):
    """
    Embedding layer designed to capture short-range interactions
    based on Gaussian radial basis functions (RBFs).
    """
    def __init__(self, n_basis=32, n_space=3, vmin=0, vmax=1):
        super().__init__()
        self.n_space = n_space
        self.n_basis = n_basis
        self._sigma = 0.05
        self.centers = torch.linspace(vmin, vmax, n_basis)
        self.dim = self.n_basis * self.n_space

    def forward(self, x):
        x = x.unsqueeze(-1)
        centers = self.centers[None, None, :].to(x.device)
        rbf_emb = ((x - centers) / self._sigma) ** 2
        rbf_emb = torch.exp(-rbf_emb)
        return rbf_emb.reshape(-1, self.dim)


class InclinationMLPEmbedding(nn.Module):
    """
    MLP embedding for inclination angle which doesn't have periodicity.
    Maps inclination angle from [0, 1] (normalized [0, π/2]) to a higher dimensional embedding.
    """
    def __init__(self, embedding_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        self.dim = embedding_dim

    def forward(self, x):
        """
        Args:
            x: Inclination angles [num_edges, 1] or [num_edges] (will be unsqueezed)
        Returns:
            Embedded inclination angles [num_edges, embedding_dim]
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # [num_edges, 1]
        return self.mlp(x)  # [num_edges, embedding_dim]


class CSPLayer(nn.Module):
    """
    Message passing layer for cspnet.

    Copied from diffcsp: https://github.com/jiaor17/DiffCSP/blob/main/diffcsp/pl_modules/cspnet.py

    This layer is designed to perform message passing in a crystal structure prediction network (CSPNet). 
    It is adapted from the DiffCSP implementation.
    Attributes:
        hidden_dim (int): The dimension of hidden layers. Default is 128.
        act_fn (nn.Module): Activation function to use. Default is nn.SiLU().
        dis_emb (nn.Module): Optional distance embedding module. Default is None.
        ln (bool): Whether to use layer normalization. Default is False.
        ip (bool): Whether to use inner product for lattice. Default is True.
        dis_dim (int): Dimension of distance embedding. Default is 3.
        edge_mlp (nn.Sequential): MLP for edge features.
        node_mlp (nn.Sequential): MLP for node features.
        layer_norm (nn.LayerNorm): Layer normalization module.
    Methods:
        edge_model(node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff=None):
            Computes edge features based on node features, fractional coordinates, and lattice information.
        node_model(node_features, edge_features, edge_index):
            Computes node features based on aggregated edge features.
        forward(node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff=None):
            Forward pass for the CSPLayer. Computes updated node features.
    """

    def __init__(
            self,
            hidden_dim=128,
            act_fn=nn.SiLU(),
            dis_emb=None,
            rot_dist_emb=None,
            inclination_emb=None,
            represent_angle_edge_to_lattice=False,
            ln=False,
            ip=True
    ):
        super(CSPLayer, self).__init__()

        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.rot_dist_emb = rot_dist_emb
        self.inclination_emb = inclination_emb
        self.ip = True
        self.represent_angle_edge_to_lattice = represent_angle_edge_to_lattice
        # fractional coordinate unit vector does not meet symmetry requirement, hence dropped.
        angle_edge_dims = 6 if represent_angle_edge_to_lattice else 0
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        if rot_dist_emb is not None:
            self.rot_dis_dim = rot_dist_emb.dim
        else:
            self.rot_dis_dim = 2  # Default dimension for two periodic rotation angles (magnitude, azimuthal)
        if inclination_emb is not None:
            self.inclination_dim = inclination_emb.dim
        else:
            self.inclination_dim = 1  # Default dimension for single inclination angle
            
        # Calculate total edge dimension
        total_edge_dim = hidden_dim * 2 + 9 + self.dis_dim + self.rot_dis_dim + self.inclination_dim + angle_edge_dims
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(total_edge_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn)
        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def _create_unit_dots_ltlf(
        self,
        lattice: torch.Tensor,
        edge2graph: torch.LongTensor,
        frac_diff: torch.Tensor,
    ) -> torch.Tensor:
        ltl = lattice @ lattice.transpose(-1, -2)
        dots = torch.einsum("...ij,...j->...i", ltl[edge2graph], frac_diff)
        eps = 1e-8
        unit_dots = dots / (dots.norm(dim=-1, keepdim=True) + eps)
        return unit_dots

    def edge_model(self, node_features, frac_coords, lattices, edge_index, edge2graph,
                   frac_diff=None, rot_diff=None):

        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        if self.represent_angle_edge_to_lattice:
            fracs_unit_dots = self._create_unit_dots_ltlf(
                lattices, edge2graph, frac_diff
            )
            rot_unit_dots = self._create_unit_dots_ltlf(
                lattices, edge2graph, rot_diff
            )
        if frac_diff is None:
            xi, xj = frac_coords[edge_index[0]], frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)
        
        # Process rotation features: separate periodic and non-periodic components
        # rot_diff[:, 0] = rotation angle (periodic)
        # rot_diff[:, 1] = inclination angle (non-periodic, bounded)  
        # rot_diff[:, 2] = azimuthal angle (periodic)
        if rot_diff is not None:
            periodic_rot_diff = torch.stack([rot_diff[:, 0], rot_diff[:, 2]], dim=1)  # [num_edges, 2]
            inclination_diff = rot_diff[:, 1]  # [num_edges]
            
            if self.rot_dist_emb is not None:
                periodic_rot_diff = self.rot_dist_emb(periodic_rot_diff)
            if self.inclination_emb is not None:
                inclination_diff = self.inclination_emb(inclination_diff)
            else:
                inclination_diff = inclination_diff.unsqueeze(-1)  # [num_edges, 1]
        else:
            # Fallback when rot_diff is None (shouldn't happen in normal flow)
            num_edges = hi.shape[0]
            device = hi.device
            periodic_rot_diff = torch.zeros(num_edges, self.rot_dis_dim, device=device)
            inclination_diff = torch.zeros(num_edges, self.inclination_dim, device=device)
        
        if self.ip:
            lattice_ips = lattices @ lattices.transpose(-1, -2)
        else:
            lattice_ips = lattices
        lattice_ips_flatten = lattice_ips.view(-1, 9)
        lattice_ips_flatten_edges = lattice_ips_flatten[edge2graph]
        
        # Concatenate all edge features
        edge_components = [hi, hj, lattice_ips_flatten_edges, frac_diff, periodic_rot_diff, inclination_diff]
        if self.represent_angle_edge_to_lattice:
            edge_components.append(fracs_unit_dots)
            edge_components.append(rot_unit_dots)
        edges_input = torch.cat(edge_components, dim=1)
        # print(edges_input.shape)
        edge_features = self.edge_mlp(edges_input)
        return edge_features

    def node_model(self, node_features, edge_features, edge_index):

        agg = scatter(edge_features, edge_index[0], dim=0, reduce='mean', dim_size=node_features.shape[0])
        agg = torch.cat([node_features, agg], dim=1)
        out = self.node_mlp(agg)
        return out

    def forward(self, node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff=None,
                rot_diff=None):
        node_input = node_features
        edge_features = self.edge_model(node_features, frac_coords, lattices, edge_index, edge2graph,
                                        frac_diff, rot_diff)
        node_output = self.node_model(node_features, edge_features, edge_index)
        out = node_input + node_output
        return out
