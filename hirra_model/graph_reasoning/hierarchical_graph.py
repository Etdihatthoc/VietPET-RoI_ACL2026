"""
Hierarchical ROI Graph Reasoning

Two-level graph structure:
1. ROI-level: Fine-grained lesion relationships
2. Organ-level: Coarse anatomical structure

Message passing flow:
- Bottom-up: ROIs aggregate to organs
- Organ-level: Inter-organ relationships
- Top-down: Organs broadcast context back to ROIs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .roi_gnn import ROIGraphAttentionNetwork


class HierarchicalROIGraph(nn.Module):
    """
    Hierarchical graph with two levels:

    Organ-Level Graph:
        Brain ← → Neck ← → Chest ← → Abdomen ← → Pelvis
          ↓        ↓        ↓          ↓          ↓
    ROI-Level Graph:
        [ROI 1] [ROI 2] [ROI 3,4] [ROI 5,6,7] [ROI 8]

    This structure allows modeling both:
    - Fine-grained lesion interactions (ROI-level)
    - Coarse anatomical context (Organ-level)
    """

    # Anatomical regions based on z-coordinate ranges (assuming whole-body CT/PET)
    ORGAN_REGIONS = {
        0: {'name': 'Brain', 'z_range': (0, 20)},
        1: {'name': 'Neck', 'z_range': (21, 40)},
        2: {'name': 'Chest/Lung', 'z_range': (41, 80)},
        3: {'name': 'Upper Abdomen', 'z_range': (81, 120)},
        4: {'name': 'Lower Abdomen', 'z_range': (121, 150)},
        5: {'name': 'Pelvis', 'z_range': (151, 200)},
    }

    def __init__(
        self,
        node_dim: int = 512,
        num_organs: int = 6,
        num_heads: int = 8,
        dropout: float = 0.2
    ):
        """
        Args:
            node_dim: Dimension of node features
            num_organs: Number of anatomical organs/regions
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.node_dim = node_dim
        self.num_organs = num_organs

        # ROI-level GNN (3 layers)
        self.roi_gnn = ROIGraphAttentionNetwork(
            node_dim=node_dim,
            edge_dim=node_dim,
            hidden_dim=node_dim,
            num_heads=num_heads,
            num_layers=3,
            dropout=dropout
        )

        # Organ-level GNN (2 layers, simpler)
        self.organ_gnn = ROIGraphAttentionNetwork(
            node_dim=node_dim,
            edge_dim=node_dim,
            hidden_dim=node_dim,
            num_heads=num_heads,
            num_layers=2,
            dropout=dropout
        )

        # Learnable organ embeddings (initialize as zero, will be populated from ROIs)
        self.organ_embeddings = nn.Parameter(torch.randn(num_organs, node_dim) * 0.02)

        # ROI-to-Organ aggregation (bottom-up)
        self.roi2organ_attn = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.roi2organ_norm = nn.LayerNorm(node_dim)

        # Organ-to-ROI broadcast (top-down)
        self.organ2roi_attn = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.organ2roi_norm = nn.LayerNorm(node_dim)

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def assign_rois_to_organs(self, roi_boxes: torch.Tensor) -> torch.Tensor:
        """
        Assign ROIs to anatomical organs based on bounding box z-coordinate.

        Args:
            roi_boxes: [N, 6] - [z_min, y_min, x_min, z_max, y_max, x_max]

        Returns:
            organ_assignments: [N] - Organ ID for each ROI
        """
        N = roi_boxes.shape[0]
        device = roi_boxes.device

        # Compute centers (use z-coordinate for anatomical region)
        centers = (roi_boxes[:, :3] + roi_boxes[:, 3:]) / 2.0  # [N, 3]
        z_centers = centers[:, 0]  # [N]

        # Assign to organs based on z-coordinate
        organ_assignments = torch.zeros(N, dtype=torch.long, device=device)

        for i, z in enumerate(z_centers):
            assigned = False
            for organ_id, region in self.ORGAN_REGIONS.items():
                if region['z_range'][0] <= z <= region['z_range'][1]:
                    organ_assignments[i] = organ_id
                    assigned = True
                    break

            # If outside defined ranges, assign to nearest region
            if not assigned:
                if z < self.ORGAN_REGIONS[0]['z_range'][0]:
                    organ_assignments[i] = 0  # Brain
                else:
                    organ_assignments[i] = self.num_organs - 1  # Pelvis

        return organ_assignments

    def build_organ_graph(
        self,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build fully-connected organ-level graph.

        Args:
            device: Device for tensors

        Returns:
            edge_index: [2, E] - Organ graph edges
            edge_attr: [E, node_dim] - Edge features (learnable)
        """
        num_organs = self.num_organs

        # Create fully-connected graph (all organs connected)
        edges = []
        for i in range(num_organs):
            for j in range(num_organs):
                if i != j:  # No self-loops
                    edges.append([i, j])

        edge_index = torch.tensor(edges, dtype=torch.long, device=device).T  # [2, E]

        # Edge features (initialized as zeros, will be learned during training)
        E = edge_index.shape[1]
        edge_attr = torch.zeros(E, self.node_dim, device=device)

        return edge_index, edge_attr

    def aggregate_rois_to_organs(
        self,
        roi_features: torch.Tensor,
        organ_assignments: torch.Tensor
    ) -> torch.Tensor:
        """
        Bottom-up: Aggregate ROI features to organ features using attention.

        Args:
            roi_features: [N, node_dim] - Enhanced ROI features
            organ_assignments: [N] - Organ ID for each ROI

        Returns:
            organ_features: [num_organs, node_dim] - Organ-level features
        """
        device = roi_features.device
        dtype = roi_features.dtype
        organ_features = self.organ_embeddings.clone().to(device=device, dtype=dtype)  # [num_organs, node_dim]

        # For each organ, aggregate ROIs belonging to it
        for organ_id in range(self.num_organs):
            mask = (organ_assignments == organ_id)

            if mask.sum() > 0:
                # ROIs in this organ
                rois_in_organ = roi_features[mask]  # [N_organ, node_dim]

                # Attention: organ embedding queries ROIs
                organ_query = organ_features[organ_id:organ_id+1].unsqueeze(0)  # [1, 1, node_dim]
                roi_kv = rois_in_organ.unsqueeze(0)  # [1, N_organ, node_dim]

                aggregated, _ = self.roi2organ_attn(
                    query=organ_query,
                    key=roi_kv,
                    value=roi_kv
                )  # [1, 1, node_dim]

                # Update organ feature with residual
                organ_features[organ_id] = self.roi2organ_norm(
                    organ_features[organ_id] + aggregated.squeeze()
                )

        return organ_features

    def broadcast_organs_to_rois(
        self,
        roi_features: torch.Tensor,
        organ_features: torch.Tensor,
        organ_assignments: torch.Tensor
    ) -> torch.Tensor:
        """
        Top-down: Broadcast organ context back to ROIs.

        Args:
            roi_features: [N, node_dim] - ROI features
            organ_features: [num_organs, node_dim] - Enhanced organ features
            organ_assignments: [N] - Organ ID for each ROI

        Returns:
            roi_features_enhanced: [N, node_dim] - ROIs with organ context
        """
        N = roi_features.shape[0]
        roi_features_enhanced = roi_features.clone()

        for i in range(N):
            organ_id = organ_assignments[i]

            # Attention: ROI queries its parent organ
            roi_query = roi_features[i:i+1].unsqueeze(0)  # [1, 1, node_dim]
            organ_kv = organ_features[organ_id:organ_id+1].unsqueeze(0)  # [1, 1, node_dim]

            context, _ = self.organ2roi_attn(
                query=roi_query,
                key=organ_kv,
                value=organ_kv
            )  # [1, 1, node_dim]

            # Update ROI feature with residual
            roi_features_enhanced[i] = self.organ2roi_norm(
                roi_features[i] + context.squeeze()
            )

        return roi_features_enhanced

    def forward(
        self,
        roi_features: torch.Tensor,
        roi_boxes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hierarchical graph reasoning with two-level message passing.

        Args:
            roi_features: [N, node_dim] - Initial ROI features
            roi_boxes: [N, 6] - ROI bounding boxes
            edge_index: [2, E] - ROI-level graph edges
            edge_attr: [E, node_dim] - ROI-level edge features

        Returns:
            roi_features_final: [N, node_dim] - Enhanced ROI features with hierarchical context
            graph_repr: [1, node_dim] - Global graph representation
        """
        N = roi_features.shape[0]
        device = roi_features.device

        # Handle single ROI case
        if N == 1:
            graph_repr = roi_features.mean(dim=0, keepdim=True)
            return roi_features, graph_repr

        # Step 1: Assign ROIs to organs
        organ_assignments = self.assign_rois_to_organs(roi_boxes)  # [N]

        # Step 2: ROI-level message passing
        roi_enhanced, _ = self.roi_gnn(roi_features, edge_index, edge_attr)  # [N, node_dim]

        # Step 3: Bottom-up - Aggregate ROIs to organs
        organ_features = self.aggregate_rois_to_organs(roi_enhanced, organ_assignments)  # [num_organs, node_dim]

        # Step 4: Organ-level message passing
        organ_edge_index, organ_edge_attr = self.build_organ_graph(device)
        organ_enhanced, graph_repr = self.organ_gnn(
            organ_features, organ_edge_index, organ_edge_attr
        )  # [num_organs, node_dim], [1, node_dim]

        # Step 5: Top-down - Broadcast organs to ROIs
        roi_final = self.broadcast_organs_to_rois(
            roi_enhanced, organ_enhanced, organ_assignments
        )  # [N, node_dim]

        # Step 6: Final projection
        roi_final = self.output_proj(roi_final)

        return roi_final, graph_repr
