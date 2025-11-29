import torch
from torch import nn

from .utilities import scatter_add, scatter_mean, scatter_softmax


class KernelPointCosmo(nn.Module):

    def __init__(self, in_channels, out_channels, filter):
        super().__init__()
        self.out_channels = out_channels
        mu = filter.unsqueeze(0).float()  # out_channels x k x d
        self.register_buffer("mu", mu)
        self.w = nn.Parameter(
            torch.rand(out_channels, mu.shape[1], in_channels)
        )  # out_channels x k x in_channels
        nn.init.xavier_uniform_(self.w)

    def forward(self, ijk, jkl, triangle_features, hood_coords):
        with torch.no_grad():
            dist = torch.cdist(hood_coords.unsqueeze(0), self.mu)  # n x k
            nn_idx = dist.argmin(dim=2).squeeze(0)
        w = self.w[:, nn_idx]  # use closest kernel point
        f = triangle_features[ijk]
        out_channels = torch.einsum("ni,oni->no", f, w)  # n x out
        triangle_features = scatter_add(
            out_channels, jkl, dim=0, dim_size=triangle_features.shape[0]
        )
        return triangle_features


class NeuralFieldCosmo(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=32,
        num_layers=3,
        dropout=0.0,
        radius=1.0,
        dim=3,
        field_activation=nn.Tanh,
    ):
        super().__init__()
        self.register_buffer("radius", torch.tensor(radius))
        self.register_buffer("in_channels", torch.tensor(in_channels))
        self.register_buffer("out_channels", torch.tensor(out_channels))
        self.neural_field = nn.Sequential(
            nn.Linear(dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            *[
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            * (num_layers - 2),
            nn.Linear(hidden_channels, in_channels * out_channels),
            field_activation(),
        )

    def forward(self, in_edges, out_edges, edge_features, hood_coords):
        w = self.neural_field(hood_coords / self.radius).view(
            -1, self.out_channels, self.in_channels
        )
        f = edge_features[in_edges]
        out_channels = torch.einsum("ni,noi->no", f, w)  # n x out
        edge_features = scatter_mean(
            out_channels, out_edges, dim_size=edge_features.shape[0], dim=0
        )
        return edge_features


class PointTransformerCosmo(nn.Module):
    def __init__(self, in_channels, out_channels, radius, dim=3):
        super().__init__()
        self.register_buffer("radius", torch.tensor(radius))
        self.delta = nn.Sequential(
            nn.Linear(dim, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )
        self.w1 = nn.Linear(in_channels, out_channels, bias=False)
        self.w2 = nn.Linear(in_channels, out_channels, bias=False)
        self.w3 = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, ijk, jkl, tri_features, hood_coords):
        n = tri_features.shape[0]
        d = self.delta(hood_coords / self.radius)
        w1 = self.w1(tri_features)
        w2 = self.w2(tri_features)
        w3 = self.w3(tri_features)
        a = scatter_softmax(w1[jkl] - w2[ijk] + d, jkl, dim=0, dim_size=n)
        tri_features = scatter_add(a * (w3[ijk] + d), jkl, dim=0, dim_size=n)
        return tri_features
