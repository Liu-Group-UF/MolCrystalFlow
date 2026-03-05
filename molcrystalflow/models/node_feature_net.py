import torch
from torch import nn
from molcrystalflow.models.utils import get_time_embedding


class NodeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(NodeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_bb_input = self._cfg.c_bb_input
        self.c_bb_emb = self._cfg.c_bb_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        self.bb_embedder = nn.Linear(self.c_bb_input, self.c_bb_emb, bias=False)

        embed_size = self._cfg.c_bb_emb + self._cfg.node_extra_dim + self._cfg.c_timestep_emb
        self.linear = nn.Linear(embed_size, self.c_s)
        self.layer_norm = nn.LayerNorm(self.c_s)

    def embed_t(self, timesteps):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=1024 # original max_position: 2056
        )
        return timestep_emb

    def forward(self, so3_t, r3_t, bb_emb, node_extra=None):
        # [M, c_bb_emb]
        bb_emb = self.bb_embedder(bb_emb)

        input_feats = [bb_emb]

        if node_extra is not None:
            input_feats.append(node_extra)
        input_feats.extend([
            self.embed_t(r3_t)
        ])

        linear_output = self.linear(torch.cat(input_feats, dim=-1))
        normalized_output = self.layer_norm(linear_output)
        
        return normalized_output
