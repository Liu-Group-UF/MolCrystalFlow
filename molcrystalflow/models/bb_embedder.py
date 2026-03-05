import torch
import torch.nn as nn
from torch_geometric.nn.pool import (
    radius_graph,
    global_mean_pool
)

from molcrystalflow.data import utils as du
from molcrystalflow.models.egnn import E_GCL

class GaussianSmearing(nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class AttentionPooling(nn.Module):
    """Hierarchical attention pooling for building block embeddings."""
    
    def __init__(self, node_dim, hidden_dim=128, use_coordinates=True):
        super().__init__()
        self.use_coordinates = use_coordinates
        
        # Input dimension includes node features and optionally coordinate norms (1D)
        input_dim = node_dim + (1 if use_coordinates else 0)
        
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, node_features, coordinates, batch):
        """
        Args:
            node_features: [N, D] node feature matrix
            coordinates: [N, 3] coordinate matrix  
            batch: [N] batch assignment vector
            
        Returns:
            bb_embeddings: [M, D] building block embeddings
        """
        # Prepare input for attention network
        if self.use_coordinates:
            # Compute norm of coordinates: [N, 3] -> [N, 1]
            coord_norms = torch.norm(coordinates, p=2, dim=-1, keepdim=True)  # [N, 1]
            attention_input = torch.cat([node_features, coord_norms], dim=-1)
        else:
            attention_input = node_features
            
        # Compute attention scores
        attention_scores = self.attention_net(attention_input)  # [N, 1]
        attention_scores = attention_scores.squeeze(-1)  # [N]
        
        # Apply softmax within each building block
        attention_weights = self._scatter_softmax(attention_scores, batch)
        
        # Weighted aggregation
        weighted_features = node_features * attention_weights.unsqueeze(-1)
        bb_embeddings = self._scatter_sum(weighted_features, batch)
        
        return bb_embeddings
    
    def _scatter_softmax(self, src, index):
        """Apply softmax within each group defined by index."""
        # Get unique indices and their counts
        unique_indices, inverse_indices = torch.unique(index, return_inverse=True)
        
        # Initialize output tensor
        out = torch.zeros_like(src)
        
        # Apply softmax within each group
        for i, idx in enumerate(unique_indices):
            mask = (index == idx)
            group_scores = src[mask]
            group_softmax = torch.softmax(group_scores, dim=0)
            out[mask] = group_softmax
            
        return out
    
    def _scatter_sum(self, src, index):
        """Sum elements within each group defined by index."""
        # Get unique indices
        unique_indices = torch.unique(index)
        
        # Initialize output tensor
        out = torch.zeros(len(unique_indices), src.size(-1), 
                         dtype=src.dtype, device=src.device)
        
        # Sum within each group
        for i, idx in enumerate(unique_indices):
            mask = (index == idx)
            out[i] = src[mask].sum(dim=0)
            
        return out


class BuildingBlockEmbedder(nn.Module):

    def __init__(self, bb_cfg):
        super().__init__()
        self._bb_cfg = bb_cfg
        
        self.num_atom_types = getattr(bb_cfg, 'num_atom_types', 12)
        
        # Attention pooling option
        self.use_attention_pooling = getattr(bb_cfg, 'use_attention_pooling', False)
        self.attention_hidden_dim = getattr(bb_cfg, 'attention_hidden_dim', 128)
        self.attention_use_coords = getattr(bb_cfg, 'attention_use_coordinates', True)
        self.edge_attention = getattr(bb_cfg, 'edge_attention', False)
        
        # Use compact embedding size instead of max_atoms
        self.atom_type_embedder = torch.nn.Embedding(self.num_atom_types, bb_cfg.c_node_dim)
        self.edge_dist_embedder = GaussianSmearing(0.0, bb_cfg.max_radius, bb_cfg.c_edge_dim)    

        self.egnn_layers = nn.ModuleList()
        for i in range(bb_cfg.num_layers):
            self.egnn_layers.append(
                E_GCL(
                    input_nf=bb_cfg.c_node_dim,
                    output_nf=bb_cfg.c_node_dim,
                    hidden_nf=bb_cfg.c_hidden_dim,
                    edges_in_d=bb_cfg.c_edge_dim,
                    act_fn=nn.ReLU(),
                    residual=True,
                    attention=self.edge_attention,
                    normalize=False,
                    coords_agg='mean',
                    tanh=False
                )
            )
        
        # Initialize attention pooling if enabled
        if self.use_attention_pooling:
            self.attention_pooling = AttentionPooling(
                node_dim=bb_cfg.c_node_dim,
                hidden_dim=self.attention_hidden_dim,
                use_coordinates=self.attention_use_coords
            )

    @staticmethod
    def _pairwise_distances(edge_index, pos):
        """        
        Args:
            edge_index (Tensor): [2, E] edge index
            pos (Tensor): [N, 3] node positions
        Returns:
            dists (Tensor): [E] pairwise distances
        """
        row, col = edge_index
        dists = torch.norm(pos[col] - pos[row], p=2, dim=-1)

        return dists

    @staticmethod
    def _repeat_interleave(repeats):
        outs = [torch.full((n, ), i) for i, n in enumerate(repeats)]
        return torch.cat(outs, dim=0).to(repeats.device)
    
    def forward(self, batch):
        local_coords = batch['local_coords']    # [N, 3]

        # Map atomic numbers to compact indices using the dictionary
        atom_types = batch['atom_types']  # These are the atomic numbers (Z)
            
        # Compute node features with compact indices
        node_attr = self.atom_type_embedder(atom_types)     # [N, D]

        # Compute edge index
        batch_bb = self._repeat_interleave(batch['bb_num_vec'])
        bb_edge_index = radius_graph(
            x=local_coords, 
            r=self._bb_cfg.max_radius, 
            batch=batch_bb,
            loop=False,
            max_num_neighbors=self._bb_cfg.max_neighbors)

        # Compute edge distance features
        pair_dist = self._pairwise_distances(bb_edge_index, local_coords)    # [E]
        bb_edge_feats = self.edge_dist_embedder(pair_dist)                   # [E, D]
                     
        # Message passing
        # local_coords = local_coords * du.ANG_TO_NM_SCALE
        for i in range(len(self.egnn_layers)):
            node_update, _ = self.egnn_layers[i](
                h=node_attr,
                coord=local_coords,
                edge_index=bb_edge_index,
                edge_attr=bb_edge_feats
            )

            node_attr = node_attr + node_update
        
        # Pooling: Choose between attention pooling and global mean pooling
        if self.use_attention_pooling:
            bb_attr = self.attention_pooling(node_attr, local_coords, batch_bb)
        else:
            bb_attr = global_mean_pool(node_attr, batch_bb)  # [M, D]
        
        return bb_attr
