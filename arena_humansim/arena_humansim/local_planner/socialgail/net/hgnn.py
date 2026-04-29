import torch
from torch import nn

from .finalmlp import FinalPredMLP
from .selfatten import SelfAttentionLayer
from .subgraph import SubGraph


class HGNN(nn.Module):
    """Hierarchical GNN: per-polyline SubGraph + global SelfAttention + MLP head.

    Mirrors gail_airl_ppo.network.GNN_modules.vectornet.HGNN exactly so the
    upstream best.pt checkpoint loads byte-for-byte.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        goal_shape=2,
        num_subgraph_layers=2,
        num_global_graph_layer=1,
        subgraph_width=32,
        global_graph_width=32,
        final_mlp_hidden_width=64,
    ):
        super().__init__()
        self.goal_shape = goal_shape
        self.polyline_vec_shape = in_channels * (2**num_subgraph_layers)
        self.subgraph = SubGraph(in_channels, num_subgraph_layers, subgraph_width)
        self.self_atten_layer = SelfAttentionLayer(self.polyline_vec_shape, global_graph_width, need_scale=False)
        self.traj_pred_mlp = FinalPredMLP(global_graph_width + self.goal_shape, out_channels, final_mlp_hidden_width)

    def forward(self, data):
        time_step_len = int(data.time_step_len[0])
        valid_lens = data.valid_len
        sub_graph_out = self.subgraph(data)
        x = sub_graph_out.x.view(-1, time_step_len, self.polyline_vec_shape)
        out = self.self_atten_layer(x, valid_lens)
        out_new = torch.cat((out[:, [0]].squeeze(1), data.goal.view(-1, 2)), dim=1)
        return self.traj_pred_mlp(out_new)


class GraphStateIndependentPolicy(nn.Module):
    """tanh-mean Gaussian policy over HGNN output. Inference uses the mean."""

    def __init__(self, in_channels, action_shape, final_mlp_hidden_width=64):
        super().__init__()
        self.net = HGNN(in_channels, action_shape[0], final_mlp_hidden_width=final_mlp_hidden_width)
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states):
        return torch.tanh(self.net(states))
