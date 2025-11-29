import torch
from torch import nn
from torch_scatter import scatter_add, scatter_mean, scatter_softmax

"""
Cosmo can be implemented with various filter functions. The underlying principle is always to compute the filter under transformation of a local reference frame (hood_coords) which is derived from neighboring input points. The forward signature of the layer is always the same and inputs can be obtained from a Lift2D or Lift3D module.
"""


class KernelPointCosmo(nn.Module):
    """
    Implements Kernel Point Convolution (Thomas et al. 2019) in the Cosmo framework.
    Note that we implement an optimization trick from KPConvX (Thomas et al. 2024) which uses only the closest kernel point for each input point.
    """

    def __init__(
        self,
        in_channels,  # Number of input channels
        out_channels,  # Number of output channels
        kernel_points,  # Kernel points of shape (k, dim)
    ):
        super().__init__()
        self.out_channels = out_channels
        mu = kernel_points.unsqueeze(0).float()  # out_channels x k x d
        self.register_buffer("mu", mu)
        self.w = nn.Parameter(torch.rand(out_channels, mu.shape[1], in_channels))
        nn.init.xavier_uniform_(self.w)

    def forward(
        self,
        source,  # Source edges ij or triangles ijk (m,)
        target,  # Target edges jk or triangles jkl (m,)
        features,  # Edge or triangle features of shape (m, in_channels)
        hood_coords,  # Locally transformed coordinates of shape (m, dim)
    ):
        m = features.shape[0]
        with torch.no_grad():
            dist = torch.cdist(hood_coords.unsqueeze(0), self.mu)  # m x k
            nn_idx = dist.argmin(dim=2).squeeze(0)
        w = self.w[:, nn_idx]  # use closest kernel point
        f = features[source]
        out_channels = torch.einsum("ni,oni->no", f, w)  # m x out
        features = scatter_add(out_channels, target, dim=0, dim_size=m)
        return features  # Updated features of shape (m, out_channels)


class NeuralFieldCosmo(nn.Module):
    """
    Implements Neural Field Convolution (Proposed with Cosmo in Kucera et al. 2026) in the Cosmo framework. Weight matrices are computed from input coordinates in the local reference frame using a neural field (parameterized by a neural network).
    """

    def __init__(
        self,
        in_channels,  # Number of input channels
        out_channels,  # Number of output channels
        field_channels=32,  # Number of channels in the neural field
        field_layers=3,  # Number of layers in the neural field
        field_dropout=0.0,  # Dropout rate in the neural field
        field_activation=nn.Tanh,  # Activation function in the neural field
        radius=1.0,  # Scale parameter for input coordinates
        dim=3,  # Dimension of the input data (2 or 3)
    ):
        super().__init__()
        self.register_buffer("radius", torch.tensor(radius))
        self.register_buffer("in_channels", torch.tensor(in_channels))
        self.register_buffer("out_channels", torch.tensor(out_channels))
        self.neural_field = nn.Sequential(
            nn.Linear(dim, field_channels),
            nn.LayerNorm(field_channels),
            nn.ReLU(),
            nn.Dropout(field_dropout),
            *[
                nn.Linear(field_channels, field_channels),
                nn.LayerNorm(field_channels),
                nn.ReLU(),
                nn.Dropout(field_dropout),
            ]
            * (field_layers - 2),
            nn.Linear(field_channels, in_channels * out_channels),
            field_activation(),
        )

    def forward(
        self,
        source,  # Source edges ij or triangles ijk (m,)
        target,  # Target edges jk or triangles jkl (m,)
        features,  # Edge or triangle features of shape (m, in_channels)
        hood_coords,  # Locally transformed coordinates of shape (m, dim)
    ):
        m = features.shape[0]
        w = self.neural_field(hood_coords / self.radius).view(
            -1, self.out_channels, self.in_channels
        )
        f = features[source]
        out_channels = torch.einsum("ni,noi->no", f, w)  # m x out
        features = scatter_mean(out_channels, target, dim_size=m, dim=0)
        return features  # Updated features of shape (m, out_channels)


class PointTransformerCosmo(nn.Module):
    """
    Implements Point Transformer Convolution (Zhao et al. 2020) in the Cosmo framework.
    """

    def __init__(
        self,
        in_channels,  # Number of input channels
        out_channels,  # Number of output channels
        radius=1.0,  # Scale parameter for input coordinates
        dim=3,  # Dimension of the input data (2 or 3)
    ):
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

    def forward(
        self,
        source,  # Source edges ij or triangles ijk (m,)
        target,  # Target edges jk or triangles jkl (m,)
        features,  # Edge or triangle features of shape (m, in_channels)
        hood_coords,  # Locally transformed coordinates of shape (m, dim)
    ):
        m = features.shape[0]
        d = self.delta(hood_coords / self.radius)
        w1 = self.w1(features)
        w2 = self.w2(features)
        w3 = self.w3(features)
        a = scatter_softmax(w1[target] - w2[source] + d, target, dim=0, dim_size=m)
        features = scatter_add(a * (w3[source] + d), target, dim=0, dim_size=m)
        return features  # Updated features of shape (m, out_channels)
