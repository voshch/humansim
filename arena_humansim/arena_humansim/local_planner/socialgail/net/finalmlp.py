from torch import nn


class FinalPredMLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_unit):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, out_channels),
        )

    def forward(self, x):
        return self.mlp(x)
