# SARL ValueNetwork from vita-epfl/CrowdNav (commit master, MIT license).
# Source: crowd_nav/policy/sarl.py + crowd_nav/policy/cadrl.py (mlp helper).
# https://github.com/vita-epfl/CrowdNav
from __future__ import annotations

import torch
import torch.nn as nn


def mlp(input_dim: int, mlp_dims: list[int], last_relu: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    dims = [input_dim] + list(mlp_dims)
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i != len(dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ValueNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        self_state_dim: int,
        mlp1_dims: list[int],
        mlp2_dims: list[int],
        mlp3_dims: list[int],
        attention_dims: list[int],
        with_global_state: bool,
    ) -> None:
        super().__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)
        self.attention_weights: torch.Tensor | None = None

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        size = state.shape
        self_state = state[:, 0, : self.self_state_dim]
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)

        if self.with_global_state:
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)

        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].detach()

        features = mlp2_output.view(size[0], size[1], -1)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        value = self.mlp3(joint_state)
        return value
