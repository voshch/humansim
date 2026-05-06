# MIT License
#
# Adapted from CrowdNav_DSRNN (https://github.com/Shuijing725/CrowdNav_DSRNN)
# upstream commit 91fb53c0f81964bbce8dd47960419fa8ab2fa810.
# Original copyright (c) 2021 Shuijing Liu.
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def _init(
    module: nn.Module,
    weight_init: Callable[..., Any],
    bias_init: Callable[..., Any],
    gain: float = 1.0,
) -> nn.Module:
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def _orth_init(m: nn.Module) -> nn.Module:
    return _init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))


def _orth_init_sqrt2(m: nn.Module) -> nn.Module:
    return _init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), float(np.sqrt(2)))


class _AddBias(nn.Module):
    def __init__(self, bias: torch.Tensor):
        super().__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)
        return x + bias


class _DiagGaussian(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()
        self.fc_mean = _orth_init(nn.Linear(num_inputs, num_outputs))
        self.logstd = _AddBias(torch.zeros(num_outputs))

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_mean(x)


class _RNNBase(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)


class _HumanHumanEdgeRNN(_RNNBase):
    def __init__(self, input_size: int, embedding_size: int, rnn_size: int):
        super().__init__(embedding_size, rnn_size)
        self.encoder_linear = nn.Linear(input_size, embedding_size)
        self.relu = nn.ReLU()
        self.rnn_size = rnn_size

    def forward_infer(self, inp: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # inp: (1, edges, input)  h: (1, edges, rnn)
        encoded = self.relu(self.encoder_linear(inp))
        seq_len, agent_num, _ = encoded.size()
        x = encoded.view(seq_len, agent_num, -1)
        h_in = h.view(1, agent_num, -1)
        out, h_new = self.gru(x, h_in)
        return out.view(seq_len, agent_num, -1), h_new.view(1, agent_num, -1)


class _HumanNodeRNN(_RNNBase):
    def __init__(
        self,
        input_size: int,
        embedding_size: int,
        rnn_size: int,
        output_size: int,
        edge_rnn_size: int,
    ):
        super().__init__(embedding_size * 2, rnn_size)
        self.rnn_size = rnn_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.encoder_linear = nn.Linear(input_size, embedding_size)
        self.relu = nn.ReLU()
        self.edge_embed = nn.Linear(edge_rnn_size, embedding_size)
        self.edge_attention_embed = nn.Linear(edge_rnn_size * 2, embedding_size)
        self.output_linear = nn.Linear(rnn_size, output_size)

    def forward_infer(
        self,
        pos: torch.Tensor,
        h_temporal: torch.Tensor,
        h_spatial_other: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_input = self.relu(self.encoder_linear(pos))
        h_edges = torch.cat((h_temporal, h_spatial_other), -1)
        h_edges_embedded = self.relu(self.edge_attention_embed(h_edges))
        concat_encoded = torch.cat((encoded_input, h_edges_embedded), -1)

        seq_len, agent_num, _ = concat_encoded.size()
        x = concat_encoded.view(seq_len, agent_num, -1)
        h_in = h.view(1, agent_num, -1)
        out, h_new = self.gru(x, h_in)
        out = out.view(seq_len, agent_num, -1)
        outputs = self.output_linear(out)
        return outputs, h_new.view(1, agent_num, -1)


class _EdgeAttention(nn.Module):
    def __init__(self, edge_rnn_size: int, attention_size: int):
        super().__init__()
        self.attention_size = attention_size
        self.temporal_edge_layer = nn.ModuleList([nn.Linear(edge_rnn_size, attention_size)])
        self.spatial_edge_layer = nn.ModuleList([nn.Linear(edge_rnn_size, attention_size)])

    def forward_infer(
        self,
        h_temporal: torch.Tensor,
        h_spatials: torch.Tensor,
    ) -> torch.Tensor:
        # h_temporal: (1, 1, rnn)  h_spatials: (1, num_humans, rnn)
        seq_len = h_temporal.size(0)
        num_humans = h_spatials.size(1)
        h_size = h_spatials.size(2)

        temporal_embed = self.temporal_edge_layer[0](h_temporal)
        spatial_embed = self.spatial_edge_layer[0](h_spatials)

        temporal_embed = temporal_embed.repeat_interleave(num_humans, dim=1)
        attn = (temporal_embed * spatial_embed).sum(dim=2)
        temperature = num_humans / np.sqrt(self.attention_size)
        attn = attn * temperature
        attn = attn.view(seq_len, 1, num_humans)
        attn = torch.nn.functional.softmax(attn, dim=-1)

        h_perm = h_spatials.view(seq_len, num_humans, h_size).permute(0, 2, 1)
        attn_v = attn.view(seq_len, num_humans, 1)
        weighted = torch.bmm(h_perm, attn_v).squeeze(-1).view(seq_len, 1, h_size)
        return weighted


class SRNN(nn.Module):
    """SRNN actor for inference. Submodule names match upstream so that the
    upstream PPO checkpoint loads with strict=True."""

    def __init__(
        self,
        human_num: int,
        human_node_rnn_size: int = 128,
        human_human_edge_rnn_size: int = 256,
        human_node_input_size: int = 3,
        human_human_edge_input_size: int = 2,
        human_node_output_size: int = 256,
        human_node_embedding_size: int = 64,
        human_human_edge_embedding_size: int = 64,
        attention_size: int = 64,
        action_dim: int = 2,
    ):
        super().__init__()
        self.human_num = human_num
        self.human_node_rnn_size = human_node_rnn_size
        self.human_human_edge_rnn_size = human_human_edge_rnn_size
        self.output_size = human_node_output_size

        self.humanNodeRNN = _HumanNodeRNN(
            input_size=human_node_input_size,
            embedding_size=human_node_embedding_size,
            rnn_size=human_node_rnn_size,
            output_size=human_node_output_size,
            edge_rnn_size=human_human_edge_rnn_size,
        )
        self.humanhumanEdgeRNN_spatial = _HumanHumanEdgeRNN(
            input_size=human_human_edge_input_size,
            embedding_size=human_human_edge_embedding_size,
            rnn_size=human_human_edge_rnn_size,
        )
        self.humanhumanEdgeRNN_temporal = _HumanHumanEdgeRNN(
            input_size=human_human_edge_input_size,
            embedding_size=human_human_edge_embedding_size,
            rnn_size=human_human_edge_rnn_size,
        )
        self.attn = _EdgeAttention(human_human_edge_rnn_size, attention_size)

        hidden_size = human_node_output_size

        self.actor = nn.Sequential(
            _orth_init_sqrt2(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            _orth_init_sqrt2(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            _orth_init_sqrt2(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            _orth_init_sqrt2(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        self.critic_linear = _orth_init_sqrt2(nn.Linear(hidden_size, 1))
        self.robot_linear = _orth_init_sqrt2(nn.Linear(7, 3))
        self.human_node_final_linear = _orth_init_sqrt2(nn.Linear(human_node_output_size, 2))

        self.dist = _DiagGaussian(human_node_output_size, action_dim)

        self.temporal_edges = [0]
        self.spatial_edges = list(range(1, human_num + 1))

    def initial_hidden(self, device: torch.device, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
        node_h = torch.zeros(1, 1, self.human_node_rnn_size, dtype=dtype, device=device)
        edge_h = torch.zeros(1, self.human_num + 1, self.human_human_edge_rnn_size, dtype=dtype, device=device)
        return {"human_node_rnn": node_h, "human_human_edge_rnn": edge_h}

    def act(
        self,
        robot_node: torch.Tensor,
        temporal_edges: torch.Tensor,
        spatial_edges: torch.Tensor,
        rnn_hxs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Shapes (inference, single env, seq_len=1):
        #   robot_node:      (1, 1, 7)
        #   temporal_edges:  (1, 1, 2)
        #   spatial_edges:   (1, human_num, 2)
        #   human_node_rnn:  (1, 1, human_node_rnn_size)
        #   human_human_edge_rnn: (1, human_num+1, human_human_edge_rnn_size)
        node_h = rnn_hxs["human_node_rnn"]
        edge_h = rnn_hxs["human_human_edge_rnn"]

        h_temporal_in = edge_h[:, self.temporal_edges, :]
        out_temporal, h_temporal = self.humanhumanEdgeRNN_temporal.forward_infer(temporal_edges, h_temporal_in)

        h_spatial_in = edge_h[:, self.spatial_edges, :]
        out_spatial, h_spatial = self.humanhumanEdgeRNN_spatial.forward_infer(spatial_edges, h_spatial_in)

        new_edge_h = torch.zeros_like(edge_h)
        new_edge_h[:, self.temporal_edges, :] = h_temporal
        new_edge_h[:, self.spatial_edges, :] = h_spatial

        attn_weighted = self.attn.forward_infer(out_temporal, out_spatial)

        nodes_current = self.robot_linear(robot_node)
        outputs, new_node_h = self.humanNodeRNN.forward_infer(nodes_current, out_temporal, attn_weighted, node_h)

        x = outputs[:, 0, :]
        hidden_actor = self.actor(x)
        action_mean = self.dist.mean(hidden_actor)

        new_hxs = {"human_node_rnn": new_node_h, "human_human_edge_rnn": new_edge_h}
        return action_mean, new_hxs
