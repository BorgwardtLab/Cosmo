<p align="center">
    <img src="https://raw.githubusercontent.com/BorgwardtLab/Cosmo/main/assets/logo.png" width="100"/>
</p>

This repository contains the Cosmo neural network lift and convolution layers. For a usage example and reproduction of the results of the RECOMB 2026 submission "Gaining mechanistic insight from geometric deep learning on molecule structures through equivariant convolution", see https://github.com/BorgwardtLab/RECOMB2026Cosmo.

Installation: `pip install cosmic-torch` or `pip install git+https://github.com/BorgwardtLab/Cosmo`. Make sure to before install [torch](https://pytorch.org/get-started/locally/) and [torch-scatter](https://pypi.org/project/torch-scatter/) according to their instructions.

### Cosmo

Cosmo is a neural network architecture based on message passing on geometric graphs of molecule structures. It applies a convolutional filter by translating it to vertices and rotating it towards neighbors. The resulting feature activation (message) is passed to the neighbor that the filter was pointed at. This way, large geometric patterns can be modeled with a template-matching objective by using multiple Cosmo layers. A Cosmo network is equivariant to translation and rotation, and highly interpretable as its weight matrices can be linearly combined and its filter poses can be reconstructed geometrically. For more details, please see the paper.

### Example Usage

Cosmo layers operate on lifted geometric graphs. These are computed from an adjacency matrix of the data, either given by e.g. atomic bond connectivity, or constructed by e.g. k-NN:

```
adj = torch_geometric.nn.knn_graph(coords, k, batch_index)
```

where `coords` are the input point coordinates of the data, `k` is a hyperparameter, and `batch_index` assigns each node to an instance in the batch (compare the computing principles of [PyG](https://pytorch-geometric.readthedocs.io/en/2.4.0/index.html), which we highly recommend to use).

Given coordinates, node features (e.g. one-hot encoded atom type), and the adjacency we can lift the input graph:

```
L = Lift2D()(features, coords, adj, batch_index) # or Lift3D()
```

The `L` namespace contains everything that we need to compute in subsequent Cosmo layers:

```
features = layer(L.source, L.target, L.features, L.hood_coords)
```

After the Cosmo layers we need to undo the lift operation (lowering) to obtain features on the input graph. This is done by aggregating the edge/triangle features to the nodes, which yields a standard graph object that can be further computed on with PyG layers, for example.

```
node_features = Lower(agg="max")(features, L.lifted2node, num_nodes)
```

Or, if features should be aggregated directly to the instance (graph) level:

```
graph_features = Lower(agg="max")(features, L.lifted2inst, num_instances)
```

An entire Cosmo network for a node classification task could look like this:

```
from cosmic import *
import torch.nn as nn

class CosmoModel(nn.Module):

    def __init__(self):
        self.lift = Lift3D()
        self.lower = Lower()
        self.cosmo_layers = nn.ModuleList([
            NeuralFieldCosmo(in_channels=5, out_channels=128, dim=3),
            NeuralFieldCosmo(in_channels=128, out_channels=128, dim=3),
            NeuralFieldCosmo(in_channels=128, out_channels=10, dim=3)
        ])

    def forward(self, node_features, coords, adj, batch_index, num_nodes):
        L = self.lift(node_features, coords, adj, batch_index)
        features = L.features
        for layer in self.cosmo_layers:
            features = layer(L.source, L.target, features, L.hood_coords)
        node_features = self.lower(features, L.lifted2node, num_nodes)
        # there could be some classic GNN-layers here, or an MLP head
        return node_features
```


### Citation

TBD

### License

TBD