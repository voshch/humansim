# GA3C-CADRL (IROS 2018) network. Architecture mirrors mit-acl/cadrl_ros and
# mit-acl/gym-collision-avoidance (MIT license, Copyright (c) 2018 MIT-ACL).
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GA3CCADRLNet(nn.Module):
    HOST_LEN = 4
    OTHER_LEN = 7
    LSTM_HIDDEN = 64
    FC_HIDDEN = 256

    def __init__(self, num_actions: int = 11, max_other_agents: int = 10):
        super().__init__()
        self.num_actions = num_actions
        self.max_other_agents = max_other_agents

        self.lstm = nn.LSTM(input_size=self.OTHER_LEN, hidden_size=self.LSTM_HIDDEN, batch_first=True)
        self.layer1 = nn.Linear(self.HOST_LEN + self.LSTM_HIDDEN, self.FC_HIDDEN)
        self.layer2 = nn.Linear(self.FC_HIDDEN, self.FC_HIDDEN)
        self.fullyconnected1 = nn.Linear(self.FC_HIDDEN, self.FC_HIDDEN)
        self.logits_p = nn.Linear(self.FC_HIDDEN, num_actions)
        self.logits_v = nn.Linear(self.FC_HIDDEN, 1)

    def forward(
        self,
        host_vec: torch.Tensor,
        other_seq: torch.Tensor,
        seq_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # other_seq: (B, max_other_agents, OTHER_LEN); seq_lengths: (B,) int
        B = host_vec.size(0)
        rnn_outputs, _ = self.lstm(other_seq)
        # Match TF dynamic_rnn semantics: take output at index seq_length-1; if 0, use zeros.
        idx = (seq_lengths.clamp(min=1) - 1).view(B, 1, 1).expand(-1, 1, self.LSTM_HIDDEN)
        last_out = rnn_outputs.gather(1, idx).squeeze(1)
        last_out = torch.where((seq_lengths == 0).view(B, 1), torch.zeros_like(last_out), last_out)
        layer1_input = torch.cat([host_vec, last_out], dim=1)
        x = F.relu(self.layer1(layer1_input))
        x = F.relu(self.layer2(x))
        x = F.relu(self.fullyconnected1(x))
        logits_p = self.logits_p(x)
        logits_v = self.logits_v(x).squeeze(1)
        return logits_p, logits_v
