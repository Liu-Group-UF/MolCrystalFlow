import math

import torch
from torch import nn
from torch_scatter import scatter_mean, scatter
from torch_geometric.utils import dense_to_sparse

from molcrystalflow.data import utils as du
from molcrystalflow.data.so3_utils import rotmat_to_rotvec
from molcrystalflow.models.bb_embedder import BuildingBlockEmbedder
from molcrystalflow.models.edge_feature_net import (
    CSPLayer,
    InclinationMLPEmbedding,
    RBFEmbedding,
    SinusoidsEmbedding,
)
from molcrystalflow.models.node_feature_net import NodeFeatureNet

EPS = 1e-8


class FlowModel(nn.Module):

    def __init__(self, model_conf):
        super(FlowModel, self).__init__()
        self._model_conf = model_conf
        hidden_dim = model_conf.node_features.c_s
        edge_dim = model_conf.edge_features.c_p
        self.node_feature_net = NodeFeatureNet(model_conf.node_features)
        self.bb_embedder = BuildingBlockEmbedder(model_conf.bb_embedder)
        if model_conf.act_fn == 'silu':
            self.act_fn = nn.SiLU()
        if model_conf.edge_features.dist_emb == 'sin':
            self.dis_emb = SinusoidsEmbedding(n_frequencies=model_conf.edge_features.trans_n_freqs)
        if model_conf.edge_features.rot_dist_emb == 'sin':
            self.rot_dist_emb = SinusoidsEmbedding(n_frequencies=model_conf.edge_features.rot_n_freqs,
                                                   n_space=2)  # Only for periodic angles (rotation, azimuthal)
        if model_conf.edge_features.dist_emb == 'rbf':
            self.dis_emb = RBFEmbedding(n_basis=model_conf.edge_features.trans_n_freqs)
        elif model_conf.edge_features.dist_emb == 'none':
            self.dis_emb = None
        represent_angle_edge_to_lattice = getattr(model_conf.edge_features, 'represent_angle_edge_to_lattice', False)
        # Add inclination angle embedding (non-periodic)
        inclination_emb_dim = getattr(model_conf.edge_features, 'inclination_emb_dim', False)
        if inclination_emb_dim:
            self.inclination_emb = InclinationMLPEmbedding(embedding_dim=inclination_emb_dim)
        else:
            self.inclination_emb = None
        for i in range(0, model_conf.num_layers):
            self.add_module(
                "csp_layer_%d" % i, CSPLayer(model_conf.node_embed_size, 
                                             act_fn=self.act_fn, 
                                             dis_emb=self.dis_emb, 
                                             rot_dist_emb=self.rot_dist_emb,
                                             inclination_emb=self.inclination_emb,
                                             ln=model_conf.ln,
                                             represent_angle_edge_to_lattice=represent_angle_edge_to_lattice)
            )
        self.num_layers = model_conf.num_layers
        
        self.bb_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True), 
            nn.SiLU(),
            nn.Linear(hidden_dim, 3, bias=False)
        )
        
        self.rot_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True), 
            nn.SiLU(),
            nn.Linear(hidden_dim, 3, bias=False)
        )
        
        self.lattice_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, 9, bias=False)
        )
        
        # Axis flip embedding - convert binary pattern to single embedding
        self.axis_flip_embedding = nn.Embedding(2, hidden_dim)
        
        # Create different projection MLPs for each layer to handle concatenated features
        for i in range(self.num_layers):
            self.add_module(
                f"axis_flip_proj_{i}", 
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),  # Concatenated: node_embed + axis_flip
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            )
            # Add normalization layer for each projection
            self.add_module(f"axis_flip_norm_{i}", nn.LayerNorm(hidden_dim))

    def gen_edges(self, num_bbs, trans_t, rotmats_t):
        lis = [torch.ones(n, n, device=num_bbs.device) for n in num_bbs]
        fc_graph = torch.block_diag(*lis)
        fc_edges, _ = dense_to_sparse(fc_graph)
        
        # Compute rotation matrix difference between molecules
        R1 = rotmats_t[fc_edges[0]]  # [num_edges, 3, 3]
        R2 = rotmats_t[fc_edges[1]]  # [num_edges, 3, 3]
        R_diff = torch.bmm(R1.transpose(-2, -1), R2)  # [num_edges, 3, 3]
        
        # Convert R_diff to rotation vector (axis-angle representation)
        rot_vec = rotmat_to_rotvec(R_diff)  # [num_edges, 3]
        
        # Extract rotation angle (magnitude of rotation vector)
        rot_angle = torch.norm(rot_vec, dim=-1, keepdim=True)  # [num_edges, 1]
        
        # Extract rotation axis (normalized rotation vector)
        # Handle zero rotation case to avoid division by zero
        rot_axis = rot_vec / (rot_angle + 1e-8)  # [num_edges, 3]
        
        # Convert rotation axis to spherical coordinates
        # Inclination angle (theta): angle from z-axis, range [0, pi]
        inclination = torch.acos(torch.clamp(rot_axis[..., 2], -1 + 1e-7, 1 - 1e-7))  # [num_edges]
        
        # Azimuthal angle (phi): angle in xy-plane from x-axis, range [-pi, pi]
        azimuthal = torch.atan2(rot_axis[..., 1], rot_axis[..., 0])  # [num_edges]
        
        # Combine all three angles
        rot_diff = torch.stack([
            rot_angle.squeeze(-1) / (2*torch.pi),  # Normalize rotation angle by 2*pi
            inclination / torch.pi,            # Normalize inclination by pi
            azimuthal / (2 * torch.pi)  # Normalize azimuthal to [0, 1]
        ], dim=-1)  # [num_edges, 3]

        return fc_edges, (trans_t[fc_edges[1]] - trans_t[fc_edges[0]]) % 1., rot_diff

    def forward(self, input_feats, remove_mean=True):
        so3_t = input_feats['so3_t']
        r3_t = input_feats['r3_t']
        trans_t = input_feats['trans_t']
        rotmats_t = input_feats['rotmats_t']
        lattice_t = input_feats['lattice_t']
        
        # Initialize building block embeddings
        bb_emb = self.bb_embedder(input_feats)
        bb_num_vec = input_feats['bb_num_vec']         # [num_bbs]
        eigenvalues = input_feats.get('eigenvalues', None)       # [num_bbs, 3]
        geometric_features = input_feats.get('geometric_features', None)       # [num_bbs, 4]
        basic_features = input_feats.get('basic_features', None)       # [num_bbs, 2]
        chemical_features = input_feats.get('chemical_features', None)       # [num_bbs, 8]
        axis_flips = input_feats.get('axis_flips', None)       # [num_bbs, 3]
        
        # Build node_extra features conditionally
        _bb_num_vec = bb_num_vec.unsqueeze(-1).float()  # [num_bbs, 1]
        min_val, max_val = 5, 89
        _bb_num_vec = (_bb_num_vec - min_val) / (max_val - min_val)
        
        node_extra_list = [_bb_num_vec]
        if eigenvalues is not None:
            node_extra_list.append(eigenvalues)
        if geometric_features is not None:
            node_extra_list.append(geometric_features)
        if basic_features is not None:
            node_extra_list.append(basic_features)
        if chemical_features is not None:
            node_extra_list.append(chemical_features)
        
        node_extra = torch.cat(node_extra_list, dim=-1)  # [num_bbs, variable_size]
        # Pass extra features to node_feature_net, including axes_flips
        init_node_embed = self.node_feature_net(
            so3_t,
            r3_t,
            bb_emb,
            node_extra,
        )
        edge_index, frac_diff, rot_diff = self.gen_edges(
            input_feats['num_bbs'],
            trans_t,
            rotmats_t  # add rotation features
        )
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        node_embed = init_node_embed

        node2graph = input_feats['batch']  # [num_bbs]
        edge2graph = node2graph[edge_index[0]]  # [num_edges]
        curr_lattice = lattice_t
        
        # Process axis flips for global embedding
        # Convert [0,0,0] -> 0 and [1,0,0] -> 1 for embedding lookup
        axis_flip_indices = axis_flips[:, 0].long()
        axis_flip_embed = self.axis_flip_embedding(axis_flip_indices)

        for i in range(self.num_layers):
            # Concatenate node embeddings with axis flip embeddings
            node_axis_concat = torch.cat([node_embed, axis_flip_embed], dim=-1)  # [num_bbs, hidden_dim*2]
            # Project back to original dimension using layer-specific MLP
            projected_features = self._modules[f"axis_flip_proj_{i}"](node_axis_concat)  # [num_bbs, hidden_dim]
            # Apply normalization
            projected_features = self._modules[f"axis_flip_norm_{i}"](projected_features)  # [num_bbs, hidden_dim]
            # Add back to original node embeddings (residual connection)
            node_embed_with_flip = node_embed + projected_features  # [num_bbs, hidden_dim]

            node_embed = self._modules["csp_layer_%d" % i](node_embed_with_flip, trans_t, curr_lattice,
                                                        edge_index, edge2graph, frac_diff,
                                                        rot_diff)

        t_vec = self.bb_update(node_embed)
        if remove_mean:
            t_vec_mean = scatter_mean(t_vec, input_feats['batch'], dim=0)
            t_vec_mean = t_vec_mean.repeat_interleave(input_feats['num_bbs'], dim=0)
            t_vec = t_vec - t_vec_mean

        q_vec = self.rot_update(node_embed)
        new_rots = curr_rigids.get_rots().compose_q_update_vec(q_vec)
        pred_rotmats = new_rots.get_rot_mats()

        # Compute final lattice
        graph_embed = scatter(node_embed, node2graph, dim=0, reduce='mean')
        delta_lattice = self.lattice_out(graph_embed)
        delta_lattice = delta_lattice.view(-1, 3, 3)
        pred_lattice = torch.einsum('bij,bjk->bik', delta_lattice, lattice_t) + lattice_t

        return {
            'pred_rotmats': pred_rotmats,
            'pred_lattice': pred_lattice,
            'pred_b_trans': t_vec,
        }
