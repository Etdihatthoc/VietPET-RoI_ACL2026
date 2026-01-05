"""
ROI-Pair Graph Builder

Constructs graphs from ROI pairs by computing spatial relationships,
intensity patterns, and feature similarities. The graph explicitly models
relationships between lesions for downstream reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


def scale_boxes_to_image_space(
    boxes: torch.Tensor,
    image_shape: Tuple[int, int, int],
    original_spatial_dims: Tuple[int, int, int]
) -> torch.Tensor:
    """
    Scale boxes from original coordinate space to current image shape coordinate space.

    Args:
        boxes: [N, 6] boxes in format (z_min, y_min, x_min, z_max, y_max, x_max)
               in original image coordinate space
        image_shape: (T, H, W) of current image tensor
        original_spatial_dims: (T_orig, H_orig, W_orig) of original image

    Returns:
        boxes_scaled: [N, 6] boxes scaled to current image shape
    """
    T_img, H_img, W_img = image_shape
    T_orig, H_orig, W_orig = original_spatial_dims

    # Calculate scale factors for each dimension
    scale = torch.tensor(
        [T_img / T_orig, H_img / H_orig, W_img / W_orig],
        device=boxes.device,
        dtype=boxes.dtype
    )

    # Clone boxes to avoid in-place modification
    boxes_scaled = boxes.clone()

    # Scale z coordinates
    boxes_scaled[:, [0, 3]] = boxes_scaled[:, [0, 3]] * scale[0]
    # Scale y coordinates
    boxes_scaled[:, [1, 4]] = boxes_scaled[:, [1, 4]] * scale[1]
    # Scale x coordinates
    boxes_scaled[:, [2, 5]] = boxes_scaled[:, [2, 5]] * scale[2]

    return boxes_scaled


class SpatialRelationEncoder(nn.Module):
    """
    Encodes spatial relationship features into a dense vector representation.

    Input features:
    - Normalized distance (1)
    - IoU overlap (1)
    - Relative position x, y, z (3)
    - Relative size (1)
    - Same region indicator (1)
    Total: 7 features
    """
    def __init__(self, input_dim=7, output_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )

    def forward(self, spatial_features):
        """
        Args:
            spatial_features: [*, 7] - Spatial relation features
        Returns:
            encoded: [*, 64] - Encoded spatial features
        """
        return self.encoder(spatial_features)


class ROIPairGraphBuilder(nn.Module):
    """
    Builds graphs connecting ROI pairs based on:
    1. Spatial proximity and overlap
    2. Feature similarity (cosine similarity)
    3. Intensity patterns from CT/PET
    4. Anatomical region co-occurrence
    5. Size correlation

    The graph uses k-nearest neighbor connectivity based on spatial distance.
    """
    def __init__(
        self,
        d_model: int = 512,
        spatial_dim: int = 64,
        num_edge_types: int = 5,
        k_neighbors: int = 5,
        distance_threshold: float = 100.0
    ):
        """
        Args:
            d_model: Dimension of ROI node features
            spatial_dim: Dimension of encoded spatial features
            num_edge_types: Number of edge types to classify (not used for now, for future)
            k_neighbors: Number of nearest neighbors to connect
            distance_threshold: Maximum distance (in mm) to consider connection
        """
        super().__init__()

        self.d_model = d_model
        self.spatial_dim = spatial_dim
        self.k_neighbors = k_neighbors
        self.distance_threshold = distance_threshold

        # Spatial relation encoder
        self.spatial_encoder = SpatialRelationEncoder(
            input_dim=7, output_dim=spatial_dim
        )

        # Intensity relation encoder
        self.intensity_encoder = nn.Sequential(
            nn.Linear(10, 64),  # 5 CT stats + 5 PET stats
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, spatial_dim),
            nn.LayerNorm(spatial_dim),
            nn.ReLU()
        )

        # Edge feature encoder (combines node features + spatial + intensity)
        edge_input_dim = d_model * 2 + spatial_dim * 2  # Node pair + spatial + intensity
        self.edge_feature_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )

        # Feature similarity (cosine)
        self.cosine_sim = nn.CosineSimilarity(dim=-1)

    def compute_3d_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute 3D IoU between two sets of boxes.

        Args:
            boxes1: [N, 6] - [z_min, y_min, x_min, z_max, y_max, x_max]
            boxes2: [M, 6]
        Returns:
            iou: [N, M] - IoU matrix
        """
        # Expand dimensions for broadcasting
        boxes1 = boxes1.unsqueeze(1)  # [N, 1, 6]
        boxes2 = boxes2.unsqueeze(0)  # [1, M, 6]

        # Intersection coordinates
        inter_min = torch.max(boxes1[..., :3], boxes2[..., :3])  # [N, M, 3]
        inter_max = torch.min(boxes1[..., 3:], boxes2[..., 3:])  # [N, M, 3]

        # Intersection volume
        inter_dims = torch.clamp(inter_max - inter_min, min=0)  # [N, M, 3]
        inter_volume = inter_dims[..., 0] * inter_dims[..., 1] * inter_dims[..., 2]  # [N, M]

        # Individual volumes
        dims1 = boxes1[..., 3:] - boxes1[..., :3]  # [N, 1, 3]
        dims2 = boxes2[..., 3:] - boxes2[..., :3]  # [1, M, 3]
        volume1 = dims1[..., 0] * dims1[..., 1] * dims1[..., 2]  # [N, 1]
        volume2 = dims2[..., 0] * dims2[..., 1] * dims2[..., 2]  # [1, M]

        # Union volume
        union_volume = volume1 + volume2 - inter_volume  # [N, M]

        # IoU
        iou = inter_volume / (union_volume + 1e-6)

        return iou.squeeze()  # [N, M] or [N] if M=1

    def compute_spatial_relations(
        self,
        boxes: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute pairwise spatial relationships between ROIs.

        Args:
            boxes: [N, 6] - Bounding boxes [z_min, y_min, x_min, z_max, y_max, x_max]

        Returns:
            Dictionary containing:
            - distance: [N, N] - Euclidean distance between centers
            - normalized_distance: [N, N] - Distance normalized by average size
            - overlap: [N, N] - 3D IoU overlap
            - relative_pos: [N, N, 3] - Relative position (dz, dy, dx)
            - relative_size: [N, N] - Volume ratio
            - same_region: [N, N] - Binary indicator of same anatomical region
            - encoded: [N, N, 64] - Encoded spatial features
        """
        N = boxes.shape[0]
        device = boxes.device

        # Compute centers
        centers = (boxes[:, :3] + boxes[:, 3:]) / 2.0  # [N, 3]

        # Compute sizes and volumes
        sizes = boxes[:, 3:] - boxes[:, :3]  # [N, 3]
        volumes = sizes.prod(dim=-1, keepdim=True)  # [N, 1]

        # Pairwise Euclidean distance
        distances = torch.cdist(centers, centers, p=2)  # [N, N]

        # Normalized distance (by average volume^(1/3) which approximates radius)
        avg_sizes = (volumes.pow(1/3) + volumes.pow(1/3).T) / 2
        normalized_distances = distances / (avg_sizes + 1e-6)

        # 3D IoU (overlap)
        ious = self.compute_3d_iou(boxes, boxes)  # [N, N]

        # Relative positions (normalized by source box size)
        relative_pos = (centers.unsqueeze(1) - centers.unsqueeze(0)) / (sizes.unsqueeze(1) + 1e-6)  # [N, N, 3]

        # Relative size (volume ratio)
        relative_sizes = volumes / (volumes.T + 1e-6)  # [N, N]

        # FIX: Anatomical region based on adaptive z-coordinate quartiles
        # Divide the z-range into 4 equal regions (head/neck, chest, abdomen, pelvis)
        # This adapts to the actual coordinate space instead of using hardcoded values
        z_centers = centers[:, 0]  # [N]
        z_min_all = z_centers.min()
        z_max_all = z_centers.max()
        z_range = z_max_all - z_min_all + 1e-6  # Avoid division by zero
        # Normalize to [0, 4) and take floor to get region index
        regions = ((z_centers - z_min_all) / z_range * 4).long().clamp(0, 3)  # [N]
        same_region = (regions.unsqueeze(1) == regions.unsqueeze(0)).float()  # [N, N]

        # Concatenate all spatial features
        spatial_feats = torch.cat([
            normalized_distances.unsqueeze(-1),   # [N, N, 1]
            ious.unsqueeze(-1),                   # [N, N, 1]
            relative_pos,                         # [N, N, 3]
            relative_sizes.unsqueeze(-1),         # [N, N, 1]
            same_region.unsqueeze(-1)             # [N, N, 1]
        ], dim=-1)  # [N, N, 7]

        # Encode spatial features
        encoded_spatial = self.spatial_encoder(spatial_feats.view(-1, 7)).view(N, N, -1)  # [N, N, 64]

        return {
            'distance': distances,
            'normalized_distance': normalized_distances,
            'overlap': ious,
            'relative_pos': relative_pos,
            'relative_size': relative_sizes,
            'same_region': same_region,
            'encoded': encoded_spatial
        }

    def compute_intensity_relations(
        self,
        boxes: torch.Tensor,
        ct_image: Optional[torch.Tensor],
        pet_image: Optional[torch.Tensor],
        original_spatial_dims: Tuple[int, int, int]
    ) -> torch.Tensor:
        """
        Extract intensity statistics from CT/PET within each ROI and compute pairwise differences.

        Args:
            boxes: [N, 6] - Bounding boxes in original coordinate space
            ct_image: [1, Z, H, W] - CT scan (optional)
            pet_image: [1, Z, H, W] - PET scan (optional)
            original_spatial_dims: (T, H, W) - Original image dimensions

        Returns:
            intensity_features: [N, N, 64] - Encoded intensity relation features
        """
        N = boxes.shape[0]
        device = boxes.device

        # If no images provided, return zero features
        if ct_image is None or pet_image is None:
            return torch.zeros(N, N, self.spatial_dim, device=device)

        # FIX: Scale boxes from original coordinate space to current image space
        image_shape = (ct_image.shape[1], ct_image.shape[2], ct_image.shape[3])  # (Z, H, W)
        boxes_scaled = scale_boxes_to_image_space(boxes, image_shape, original_spatial_dims)

        # Extract ROI intensity statistics
        ct_stats_list = []
        pet_stats_list = []

        for i in range(N):
            box = boxes_scaled[i].long()  # Use scaled boxes
            z_min, y_min, x_min, z_max, y_max, x_max = box

            # Clamp to image boundaries
            z_min = max(0, z_min)
            y_min = max(0, y_min)
            x_min = max(0, x_min)
            z_max = min(ct_image.shape[1], z_max)
            y_max = min(ct_image.shape[2], y_max)
            x_max = min(ct_image.shape[3], x_max)

            # Extract ROI regions
            ct_roi = ct_image[0, z_min:z_max, y_min:y_max, x_min:x_max]
            pet_roi = pet_image[0, z_min:z_max, y_min:y_max, x_min:x_max]

            # Compute statistics (mean, std, min, max, median)
            if ct_roi.numel() > 0:
                ct_stats = torch.tensor([
                    ct_roi.mean().item(),
                    ct_roi.std().item(),
                    ct_roi.min().item(),
                    ct_roi.max().item(),
                    ct_roi.median().item()
                ], device=device)
                pet_stats = torch.tensor([
                    pet_roi.mean().item(),
                    pet_roi.std().item(),
                    pet_roi.min().item(),
                    pet_roi.max().item(),
                    pet_roi.median().item()
                ], device=device)
            else:
                # Empty ROI
                ct_stats = torch.zeros(5, device=device)
                pet_stats = torch.zeros(5, device=device)

            ct_stats_list.append(ct_stats)
            pet_stats_list.append(pet_stats)

        ct_stats = torch.stack(ct_stats_list)  # [N, 5]
        pet_stats = torch.stack(pet_stats_list)  # [N, 5]

        # Pairwise absolute differences (L1 distance)
        ct_diff = torch.abs(ct_stats.unsqueeze(1) - ct_stats.unsqueeze(0))  # [N, N, 5]
        pet_diff = torch.abs(pet_stats.unsqueeze(1) - pet_stats.unsqueeze(0))  # [N, N, 5]

        # Concatenate
        intensity_features = torch.cat([ct_diff, pet_diff], dim=-1)  # [N, N, 10]

        # Encode intensity features
        encoded_intensity = self.intensity_encoder(intensity_features.view(-1, 10)).view(N, N, -1)  # [N, N, 64]

        return encoded_intensity

    def build_adjacency(
        self,
        spatial_relations: Dict[str, torch.Tensor],
        feature_similarity: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build graph adjacency matrix using k-nearest neighbors based on spatial distance.

        Args:
            spatial_relations: Dictionary with 'distance' key
            feature_similarity: [N, N] - Feature cosine similarity

        Returns:
            edge_index: [2, E] - Graph edges (source, target)
            edge_mask: [N, N] - Binary mask of edges
        """
        N = spatial_relations['distance'].shape[0]
        device = spatial_relations['distance'].device

        distances = spatial_relations['distance']

        # For each node, connect to k nearest neighbors (excluding self)
        # Set diagonal to inf to exclude self-connections
        distances_masked = distances.clone()
        distances_masked.fill_diagonal_(float('inf'))

        # Apply distance threshold
        distances_masked = torch.where(
            distances_masked > self.distance_threshold,
            torch.tensor(float('inf'), device=device),
            distances_masked
        )

        # Get k nearest neighbors for each node
        k = min(self.k_neighbors, N - 1)  # Can't have more than N-1 neighbors
        _, indices = torch.topk(distances_masked, k, dim=1, largest=False)  # [N, k]

        # Build edge list
        sources = torch.arange(N, device=device).unsqueeze(1).expand(-1, k)  # [N, k]
        targets = indices  # [N, k]

        edge_index = torch.stack([sources.flatten(), targets.flatten()], dim=0)  # [2, N*k]

        # Remove invalid edges (where distance was inf)
        valid_mask = distances_masked[edge_index[0], edge_index[1]] < float('inf')
        edge_index = edge_index[:, valid_mask]

        # Create binary adjacency mask
        edge_mask = torch.zeros(N, N, device=device)
        edge_mask[edge_index[0], edge_index[1]] = 1.0

        return edge_index, edge_mask

    def encode_edge_features(
        self,
        roi_features: torch.Tensor,
        edge_index: torch.Tensor,
        spatial_encoded: torch.Tensor,
        intensity_encoded: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode features for each edge in the graph.

        Args:
            roi_features: [N, d_model] - ROI node features
            edge_index: [2, E] - Graph edges
            spatial_encoded: [N, N, spatial_dim] - Encoded spatial features
            intensity_encoded: [N, N, spatial_dim] - Encoded intensity features

        Returns:
            edge_features: [E, d_model] - Edge feature vectors
        """
        E = edge_index.shape[1]

        # Get source and target node features
        src_features = roi_features[edge_index[0]]  # [E, d_model]
        tgt_features = roi_features[edge_index[1]]  # [E, d_model]

        # Get spatial and intensity features for each edge
        spatial_feats = spatial_encoded[edge_index[0], edge_index[1]]  # [E, spatial_dim]
        intensity_feats = intensity_encoded[edge_index[0], edge_index[1]]  # [E, spatial_dim]

        # Concatenate all edge features
        edge_feats = torch.cat([
            src_features,      # [E, d_model]
            tgt_features,      # [E, d_model]
            spatial_feats,     # [E, spatial_dim]
            intensity_feats    # [E, spatial_dim]
        ], dim=-1)  # [E, d_model*2 + spatial_dim*2]

        # Encode to final edge feature dimension
        edge_features = self.edge_feature_encoder(edge_feats)  # [E, d_model]

        return edge_features

    def forward(
        self,
        roi_features: torch.Tensor,
        roi_boxes: torch.Tensor,
        original_spatial_dims: Tuple[int, int, int],
        ct_image: Optional[torch.Tensor] = None,
        pet_image: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build ROI-pair graph with encoded edge features.

        Args:
            roi_features: [N, d_model] - ROI features from extractor
            roi_boxes: [N, 6] - Bounding boxes [z_min, y_min, x_min, z_max, y_max, x_max]
            original_spatial_dims: (T, H, W) - Original image dimensions
            ct_image: [1, Z, H, W] - CT scan (optional, for intensity features)
            pet_image: [1, Z, H, W] - PET scan (optional, for intensity features)

        Returns:
            edge_index: [2, E] - Graph edges
            edge_attr: [E, d_model] - Edge features
            spatial_relations: Dict - Dictionary of spatial relation tensors (for visualization)
        """
        N = roi_features.shape[0]
        device = roi_features.device

        # Handle single ROI case (no graph needed)
        if N <= 1:
            # Return empty graph
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
            edge_attr = torch.zeros(0, self.d_model, device=device)
            return edge_index, edge_attr, {}

        # 1. Compute spatial relationships
        spatial_relations = self.compute_spatial_relations(roi_boxes)

        # 2. Compute feature similarity
        feat_sim = self.cosine_sim(
            roi_features.unsqueeze(1),  # [N, 1, d_model]
            roi_features.unsqueeze(0)   # [1, N, d_model]
        )  # [N, N]

        # 3. Compute intensity relationships (with coordinate scaling)
        intensity_encoded = self.compute_intensity_relations(
            roi_boxes, ct_image, pet_image, original_spatial_dims
        )

        # 4. Build adjacency matrix
        edge_index, edge_mask = self.build_adjacency(spatial_relations, feat_sim)

        # 5. Encode edge features
        edge_attr = self.encode_edge_features(
            roi_features,
            edge_index,
            spatial_relations['encoded'],
            intensity_encoded
        )

        return edge_index, edge_attr, spatial_relations
