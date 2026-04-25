"""Mirror of upstream NSP nn.Modules — layer names match the released SDD checkpoints.

State_dict compatible with realcrane/Human-Trajectory-Prediction-via-Neural-Social-Physics
(ECCV 2022, Yue et al.). The forward path here drops the F2 environment branch, since the
checkpoint we deploy was the wo (without-CVAE) variant and we feed walls through
collision/wall_projection downstream rather than as a semantic raster.
"""

from __future__ import annotations

import torch
from torch import nn


def desired_directions(current_step: torch.Tensor, dest: torch.Tensor) -> torch.Tensor:
    d = dest - current_step
    return d / (torch.norm(d, dim=-1, keepdim=True) + 1e-8)


def _value_p_p(c1: torch.Tensor, peds: torch.Tensor, coefficients: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    diff = peds - c1
    dist = torch.norm(diff, dim=-1)
    potential = sigma * coefficients * torch.exp(-dist / sigma)
    return potential.sum(dim=1)


def f_ab(current_step: torch.Tensor, coefficients: torch.Tensor, current_supplement: torch.Tensor, sigma: torch.Tensor, device: torch.device) -> torch.Tensor:
    c1 = current_supplement[:, :-1, :2]
    peds = current_step.unsqueeze(1)
    v = _value_p_p(c1, peds, coefficients, sigma)
    delta = torch.tensor(1e-3, device=device, dtype=current_step.dtype)
    dx = torch.tensor([[[delta, 0.0]]], device=device, dtype=current_step.dtype)
    dy = torch.tensor([[[0.0, delta]]], device=device, dtype=current_step.dtype)
    dvdx = (_value_p_p(c1, peds + dx, coefficients, sigma) - v) / delta
    dvdy = (_value_p_p(c1, peds + dy, coefficients, sigma) - v) / delta
    grad = torch.stack((dvdx, dvdy), dim=-1)
    return -grad


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: tuple[int, ...] = (1024, 512), activation: str = "relu", discrim: bool = False, dropout: float = -1):
        super().__init__()
        dims = [input_dim, *hidden_size, output_dim]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.activation = nn.ReLU() if activation == "relu" else nn.Sigmoid()
        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout / 3) if i == 1 else self.dropout)(x)
            elif self.sigmoid is not None:
                x = self.sigmoid(x)
        return x


class NSP(nn.Module):
    """Layer naming exactly mirrors model_nsp_wo.NSP so released state_dicts load with strict=True."""

    def __init__(self, input_size: int, embedding_size: int, rnn_size: int, output_size: int, enc_size: tuple[int, ...], dec_size: tuple[int, ...]):
        super().__init__()
        self.max_peds = 25
        self.r_pixel = 100
        self.costheta = float(torch.cos(torch.tensor(torch.pi / 3)).item())

        self.cell1 = nn.LSTMCell(embedding_size, rnn_size)
        self.input_embedding_layer1 = nn.Linear(input_size, embedding_size)
        self.output_layer1 = nn.Linear(rnn_size, output_size)

        self.encoder_dest_state = MLP(input_dim=2, output_dim=output_size, hidden_size=enc_size)
        self.dec_tau = MLP(input_dim=2 * output_size, output_dim=1, hidden_size=dec_size)

        self.cell2 = nn.LSTMCell(embedding_size, rnn_size)
        self.input_embedding_layer2 = nn.Linear(input_size, embedding_size)
        self.output_layer2 = nn.Linear(rnn_size, output_size)

        self.encoder_people_state = MLP(input_dim=4, output_dim=output_size, hidden_size=enc_size)
        self.dec_para_people = MLP(input_dim=2 * output_size, output_dim=1, hidden_size=dec_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward_lstm(
        self,
        input_lstm: torch.Tensor,
        h1: torch.Tensor,
        c1: torch.Tensor,
        h2: torch.Tensor,
        c2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.relu(self.input_embedding_layer1(input_lstm))
        h1n, c1n = self.cell1(e1, (h1, c1))
        out1 = self.output_layer1(h1n)

        e2 = self.relu(self.input_embedding_layer2(input_lstm))
        h2n, c2n = self.cell2(e2, (h2, c2))
        out2 = self.output_layer2(h2n)
        return out1, h1n, c1n, out2, h2n, c2n

    def forward_coefficient_people(
        self,
        outputs_features2: torch.Tensor,
        supplement: torch.Tensor,
        current_step: torch.Tensor,
        current_vel: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # supplement: [peds, max_peds + 1, 5], last row encodes (start_idx, count)
        num_peds = outputs_features2.size(0)
        curr_supp = torch.zeros((num_peds, self.max_peds + 1, 5), dtype=current_step.dtype, device=device)

        # The training distribution had no stationary peds; with current_vel ≈ 0 the cosine cone
        # collapses to "no neighbors visible" and F1 dies, leaving arrived/stalled agents drifting
        # into each other. When the reference direction is too small to define a cone, we admit
        # every in-range neighbor instead.
        vel_norm = torch.norm(current_vel, dim=1)
        for i in range(num_peds):
            n_others = int(supplement[i, -1, 1].item())
            if n_others <= 0:
                curr_supp[i, -1, 1] = 0
                continue
            peds_con = supplement[i, :n_others, :]
            person_dir = peds_con[:, :2] - current_step[i, :]
            dis = torch.norm(person_dir, dim=1).clamp_min(1e-8)
            if vel_norm[i] < 0.5:
                keep = dis < self.r_pixel
            else:
                cosang = (person_dir @ current_vel[i, :]) / (dis * vel_norm[i])
                keep = (dis < self.r_pixel) & (cosang > self.costheta)
            visible = peds_con[keep]
            n_vis = int(visible.shape[0])
            if n_vis > self.max_peds:
                visible = visible[: self.max_peds]
                n_vis = self.max_peds
            curr_supp[i, :n_vis, :] = visible
            curr_supp[i, -1, 1] = n_vis

        enc1 = outputs_features2.unsqueeze(1).expand(-1, self.max_peds, -1)
        feats = self.encoder_people_state(curr_supp[:, :-1, :-1])
        coeff_in = torch.cat((enc1, feats), dim=-1)
        coefficients = (100.0 * self.sigmoid(self.dec_para_people(coeff_in))).squeeze(-1)

        for i in range(num_peds):
            k = int(curr_supp[i, -1, 1].item())
            if k < self.max_peds:
                coefficients[i, k:] = 0.0
        return coefficients, curr_supp

    def forward_next_step(
        self,
        current_step: torch.Tensor,
        current_vel: torch.Tensor,
        initial_speeds: torch.Tensor,
        dest: torch.Tensor,
        features_lstm1: torch.Tensor,
        coefficients: torch.Tensor,
        current_supplement: torch.Tensor,
        sigma: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta_t = torch.tensor(0.4, device=device, dtype=current_step.dtype)
        e = desired_directions(current_step, dest)
        feats_dest = self.encoder_dest_state(dest)
        feats_tau = torch.cat((features_lstm1, feats_dest), dim=-1)
        tau = self.sigmoid(self.dec_tau(feats_tau)) + 0.4

        f0 = (1.0 / tau) * (initial_speeds * e - current_vel)
        f1 = f_ab(current_step, coefficients, current_supplement, sigma, device)

        force = f0 + f1
        w_v = current_vel + delta_t * force
        prediction = current_step + w_v * delta_t
        return prediction, w_v
