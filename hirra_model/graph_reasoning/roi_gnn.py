"""
ROI Graph Neural Network

Graph Attention Network (GATv2) for reasoning over ROI relationships.
Performs message passing to enhance ROI features with contextual information
from spatially and semantically related lesions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Try to import PyTorch Geometric components
try:
    from torch_geometric.nn import GATv2Conv
    from torch_geometric.utils import add_self_loops
    TORCH_GEOMETRIC_AVAILABLE = True
except (ImportError, OSError) as e:
    TORCH_GEOMETRIC_AVAILABLE = False
    print(f"Warning: torch_geometric not available ({type(e).__name__}). Using manual GATv2 implementation.")


class ManualGATv2Layer(nn.Module):
    """
    Manual implementation of GATv2 for cases where torch_geometric is not available.
    This is a simplified version without all optimizations.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        edge_dim: Optional[int] = None,
        concat: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        # Linear transformations
        self.lin_src = nn.Linear(in_channels, heads * out_channels)
        self.lin_dst = nn.Linear(in_channels, heads * out_channels)

        # Edge feature transformation
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels)
        else:
            self.lin_edge = None

        # Attention weights
        self.att = nn.Parameter(torch.Tensor(1, heads, out_channels))

        # Output dimension
        if concat:
            self.out_dim = heads * out_channels
        else:
            self.out_dim = out_channels

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [N, in_channels] - Node features
            edge_index: [2, E] - Edge connectivity
            edge_attr: [E, edge_dim] - Edge features (optional)

        Returns:
            out: [N, out_dim] - Updated node features
        """
        N = x.shape[0]
        H = self.heads
        C = self.out_channels

        # Transform node features
        x_src = self.lin_src(x).view(N, H, C)  # [N, H, C]
        x_dst = self.lin_dst(x).view(N, H, C)  # [N, H, C]

        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=N)
        E = edge_index.shape[1]

        # Get source and target features
        src = edge_index[0]  # [E]
        dst = edge_index[1]  # [E]
        x_i = x_src[dst]  # [E, H, C]
        x_j = x_src[src]  # [E, H, C]

        # Add edge features if available
        if edge_attr is not None and self.lin_edge is not None:
            # Add self-loop edge features (zeros)
            self_loop_attr = torch.zeros(
                N, edge_attr.shape[1],
                device=edge_attr.device, dtype=edge_attr.dtype
            )
            edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)  # [E_new, edge_dim]

            edge_feat = self.lin_edge(edge_attr).view(E, H, C)  # [E, H, C]
            alpha_src = x_i + x_j + edge_feat  # [E, H, C]
        else:
            alpha_src = x_i + x_j  # [E, H, C]

        # Compute attention scores (GATv2 style: activation before dot product)
        alpha = (F.leaky_relu(alpha_src, 0.2) * self.att).sum(dim=-1)  # [E, H]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Softmax over neighbors for each node
        alpha = torch.zeros(N, E, H, device=x.device).scatter_(
            1, dst.unsqueeze(0).unsqueeze(-1).expand(N, -1, H),
            alpha.unsqueeze(0).expand(N, -1, -1)
        )  # [N, E, H]
        alpha = F.softmax(alpha, dim=1)

        # Message passing
        messages = x_j.unsqueeze(0).expand(N, -1, -1, -1)  # [N, E, H, C]
        alpha_expanded = alpha.unsqueeze(-1)  # [N, E, H, 1]
        out = (messages * alpha_expanded).sum(dim=1)  # [N, H, C]

        # Concatenate or average heads
        if self.concat:
            out = out.view(N, H * C)
        else:
            out = out.mean(dim=1)

        return out


class ROIGraphAttentionNetwork(nn.Module):
    """
    Graph Attention Network for ROI reasoning.

    Architecture:
    - Multiple layers of GAT with edge features
    - Multi-head attention (8 heads default)
    - Residual connections
    - Layer normalization
    - Feed-forward networks
    - Global graph pooling
    """
    def __init__(
        self,
        node_dim: int = 512,
        edge_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        """
        Args:
            node_dim: Dimension of input node features
            edge_dim: Dimension of edge features
            hidden_dim: Hidden dimension for GAT layers
            num_heads: Number of attention heads
            num_layers: Number of GAT layers
            dropout: Dropout probability
        """
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Input normalization
        self.input_norm = nn.LayerNorm(node_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.ffns = nn.ModuleList()

        for i in range(num_layers):
            in_dim = node_dim if i == 0 else hidden_dim

            # GAT layer (use torch_geometric if available, otherwise manual)
            if TORCH_GEOMETRIC_AVAILABLE:
                gat = GATv2Conv(
                    in_channels=in_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    edge_dim=edge_dim,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=True
                )
            else:
                gat = ManualGATv2Layer(
                    in_channels=in_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    edge_dim=edge_dim,
                    concat=True,
                    dropout=dropout
                )

            self.gat_layers.append(gat)
            self.norms.append(nn.LayerNorm(hidden_dim))

            # Feed-forward network
            self.ffns.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            ))

        # Projection layer if input and hidden dims differ
        if node_dim != hidden_dim:
            self.input_proj = nn.Linear(node_dim, hidden_dim)
        else:
            self.input_proj = None

        # Graph-level pooling
        self.graph_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def global_mean_pool(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Global mean pooling over graphs.

        Args:
            x: [N, hidden_dim] - Node features
            batch: [N] - Batch assignment (optional)

        Returns:
            out: [num_graphs, hidden_dim] - Graph-level features
        """
        if batch is None:
            # Single graph
            return x.mean(dim=0, keepdim=True)
        else:
            # Multiple graphs
            num_graphs = batch.max().item() + 1
            out = torch.zeros(
                num_graphs, x.shape[1],
                device=x.device, dtype=x.dtype
            )
            for i in range(num_graphs):
                mask = (batch == i)
                if mask.sum() > 0:
                    out[i] = x[mask].mean(dim=0)
            return out

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the graph network.

        Args:
            node_features: [N, node_dim] - ROI features
            edge_index: [2, E] - Graph connectivity
            edge_attr: [E, edge_dim] - Edge features
            batch: [N] - Batch assignment for multiple graphs (optional)

        Returns:
            enhanced_features: [N, hidden_dim] - Enhanced ROI features
            graph_representation: [num_graphs, hidden_dim] - Graph-level features
        """
        # Handle empty graph
        if node_features.shape[0] == 0:
            return node_features, torch.zeros(1, self.hidden_dim, device=node_features.device)

        x = self.input_norm(node_features)

        # Project input if dimensions differ
        if self.input_proj is not None:
            residual = self.input_proj(x)
        else:
            residual = x

        # Apply GAT layers with residual connections
        for i in range(self.num_layers):
            # Graph attention
            x_new = self.gat_layers[i](x, edge_index, edge_attr)
            x_new = self.norms[i](x_new)

            # Residual connection (project x if first layer and dims differ)
            if i == 0 and self.input_proj is not None:
                x = residual + x_new
            else:
                x = x + x_new

            # Feed-forward network
            x = x + self.ffns[i](x)

        enhanced_features = x

        # Global graph representation
        graph_repr = self.global_mean_pool(x, batch)
        graph_repr = self.graph_pool(graph_repr)

        return enhanced_features, graph_repr
