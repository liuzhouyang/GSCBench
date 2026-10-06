import torch
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Module, Dropout, Identity, BatchNorm1d, InstanceNorm1d, \
                     LayerNorm, Linear, Sequential, ReLU, ModuleList, GRUCell
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, \
                               SAGEConv, GINConv
                               
class ImpactProp(MessagePassing):
    def __init__(self, aggr = 'sum'):
        super(ImpactProp, self).__init__(aggr = aggr)

    def forward(self,
                e,
                edge_index):
        """"""
        out = e + self.propagate(edge_index, x=e)
        return out
        
class norm_act_drop(Module):
    def __init__(self, size: int, norm_module: str, activation: str, \
                       dropout_prob: float, final_layer: bool = False):
        super().__init__()
        self.norm = self.get_norm_layer(size, norm_module) \
                                if norm_module != 'none' else None
        self.activation, self.dropout = None, None
        if not final_layer:
            self.activation = getattr(torch.nn, activation)()
            self.dropout = Dropout(dropout_prob) \
                                        if dropout_prob else None

    @staticmethod
    def get_norm_layer(size, norm_module='none'):
        if norm_module == 'none':
            return Identity()
        elif norm_module == 'batch':
            return BatchNorm1d(size)
        elif norm_module == 'instance':
            return InstanceNorm1d(size)
        elif norm_module == 'layer':
            return LayerNorm(size)
        else:
            return NotImplementedError(f"Not Implemented norm layer {norm_module}")

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class GNN(Module):
    def __init__(self, args):
        super(GNN, self).__init__()
        self.num_layers = args.num_layers

        self.preprocessor = Linear(args.input_dim, args.hidden_dim)
        
        self.convs = ModuleList()
        block_layer = self.build_layer(args.layer_type)
        for i in range(self.num_layers):
            self.convs.append(block_layer(args.hidden_dim, args.hidden_dim))
        self.skip_connection = args.skip_connection
        if self.skip_connection == 'linear':
            self.skip = Linear(args.hidden_dim * 2, args.hidden_dim)
               
    def build_layer(self, layer_type):
        if layer_type == 'GIN':
            return self.get_gin_layer
        elif layer_type == 'SAGE':
            return SAGEConv
        elif layer_type == 'GCN':
            return GCNConv
        elif layer_type == 'GAT':
            return GATConv
        elif layer_type == 'GATv2':
            return GATv2Conv
        else:
            raise NotImplementedError('GNN model not implemented')

    def get_gin_layer(self, i, h):
        return GINConv(Sequential(
                Linear(i, h), ReLU(), Linear(h, h)))

    def forward(self, x, edge_index):
        # [num_nodes_per_batch, one_hot] --> [num_nodes_per_batch, input_dim]
        x = self.preprocessor(x)
        h0 = None
        #all_emb = None
        for i in range(self.num_layers):
            # [num_nodes_per_batch, input_dim] --> [num_nodes_per_batch, hidden_dim]
            z = self.convs[i](x, edge_index)
            if h0 is None:
                h0 = torch.zeros_like(z).float()
            if self.skip_connection == 'linear':
                x = self.skip(torch.cat([h0,z]))
            elif self.skip_connection == 'identity':
                x = h0 + z
                h0 = x
            else:
                x = z
            x = F.relu(x)
        return x
