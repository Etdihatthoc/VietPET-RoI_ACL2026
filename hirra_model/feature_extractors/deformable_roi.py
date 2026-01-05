"""
Deformable RoI Align 3D with Spatial Pyramid Pooling

Improvements over basic RoI Align:
1. Deformable sampling offsets (adaptive to object geometry)
2. Spatial Pyramid Pooling (multi-scale features within RoI)
3. Attention-based feature aggregation
4. Better for irregular shapes (tumors, lesions)

References:
- Deformable DETR: Deformable Transformers for End-to-End Object Detection (Zhu et al., ICLR 2021)
- Spatial Pyramid Pooling in Deep Convolutional Networks (He et al., ECCV 2014)
- RoI Transformer for 3D Object Detection (Deng & Latecki, CVPR 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def exists(val):
    return val is not None


class DeformableRoIAlign3D(nn.Module):
    """
    Deformable RoI Align with learnable sampling offsets

    Key idea: Instead of uniform grid sampling, learn where to sample
    based on the features within each RoI.
    """
    def __init__(self,
                 input_dim: int = 512,
                 output_size: tuple = (7, 7, 7),
                 d_model: int = 512,
                 num_sample_points: int = 8,
                 dropout: float = 0.25):
        """
        Args:
            input_dim: Feature dimension from vision encoder
            output_size: Target size for pooled features (D, H, W)
            d_model: Output feature dimension
            num_sample_points: Number of deformable sampling points per spatial location
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_size = output_size
        self.num_sample_points = num_sample_points
        self.dropout = dropout

        # Offset prediction network
        # Predicts (dx, dy, dz) offsets for each sampling point
        self.offset_network = nn.Sequential(
            nn.Conv3d(input_dim, input_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv3d(input_dim // 2, 3 * num_sample_points, kernel_size=1)
        )

        # Attention weights for combining sampled features
        self.attention_network = nn.Sequential(
            nn.Conv3d(input_dim, input_dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv3d(input_dim // 4, num_sample_points, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # Base sampling grid
        grid_z, grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, output_size[0]),
            torch.linspace(-1.0, 1.0, output_size[1]),
            torch.linspace(-1.0, 1.0, output_size[2]),
            indexing="ij"
        )
        self.register_buffer('base_grid',
                           torch.stack((grid_x, grid_y, grid_z), dim=-1).unsqueeze(0))

        # Feature projection
        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        # Adaptive pooling for final output
        self.pooler = nn.AdaptiveAvgPool3d(output_size)

    def forward(self,
                F_visual: torch.Tensor,
                boxes_list: list,
                original_spatial_dims: tuple):
        """
        Args:
            F_visual: [B, D, T, H, W]
            boxes_list: List of [num_rois, 6] tensors
            original_spatial_dims: (T, H, W) of original image

        Returns:
            all_rois_features: List of [num_rois, d_model] tensors
        """
        all_rois_features = []
        B, D, T_feat, H_feat, W_feat = F_visual.shape
        device = F_visual.device

        # Normalize original dimensions
        orig_dims_tensor = torch.tensor(
            [original_spatial_dims[2], original_spatial_dims[1], original_spatial_dims[0]],
            device=device, dtype=torch.float32
        ).unsqueeze(0)

        # Get base grid
        base_grid = self.base_grid.to(device)  # [1, Oz, Oy, Ox, 3]

        for i in range(B):
            F_visual_item = F_visual[i:i+1]  # [1, D, T, H, W]
            boxes = boxes_list[i]  # [num_rois, 6]

            if boxes.shape[0] == 0:
                all_rois_features.append(torch.empty(0, self.projection[-1].out_features, device=device))
                continue

            # Predict deformable offsets for this image
            offsets = self.offset_network(F_visual_item)  # [1, 3*K, T, H, W]
            offsets = rearrange(offsets, '1 (k c) t h w -> 1 k t h w c',
                              k=self.num_sample_points, c=3)

            # Predict attention weights
            attn_weights = self.attention_network(F_visual_item)  # [1, K, T, H, W]

            # Normalize boxes
            boxes_min_zyx = boxes[:, 0:3]
            boxes_max_zyx = boxes[:, 3:6]

            boxes_min_xyz = torch.stack([boxes_min_zyx[:, 2],
                                        boxes_min_zyx[:, 1],
                                        boxes_min_zyx[:, 0]], dim=1)
            boxes_max_xyz = torch.stack([boxes_max_zyx[:, 2],
                                        boxes_max_zyx[:, 1],
                                        boxes_max_zyx[:, 0]], dim=1)

            norm_boxes_min_xyz = (boxes_min_xyz / orig_dims_tensor) * 2 - 1
            norm_boxes_max_xyz = (boxes_max_xyz / orig_dims_tensor) * 2 - 1

            box_centers = (norm_boxes_max_xyz + norm_boxes_min_xyz) / 2.0
            box_spans = (norm_boxes_max_xyz - norm_boxes_min_xyz) / 2.0

            # Process each RoI
            roi_features_list = []
            for roi_idx in range(boxes.shape[0]):
                center = box_centers[roi_idx:roi_idx+1].view(1, 1, 1, 1, 3)
                span = box_spans[roi_idx:roi_idx+1].view(1, 1, 1, 1, 3)

                # Base grid for this RoI
                sampling_grid = base_grid * span + center  # [1, Oz, Oy, Ox, 3]

                # Sample deformable offsets at this RoI location
                # For simplicity, sample offsets at RoI center
                center_coord = ((center + 1) / 2 * torch.tensor([W_feat-1, H_feat-1, T_feat-1],
                                                                device=device)).long()
                center_coord = center_coord.clamp(
                    min=torch.zeros_like(center_coord),
                    max=torch.tensor([W_feat-1, H_feat-1, T_feat-1], device=device)
                )

                roi_offsets = offsets[0, :, center_coord[0, 0, 0, 0, 2],
                                     center_coord[0, 0, 0, 0, 1],
                                     center_coord[0, 0, 0, 0, 0]]  # [K, 3]

                # Apply offsets (scaled by span)
                deformed_grids = []
                for k in range(self.num_sample_points):
                    offset = roi_offsets[k:k+1].view(1, 1, 1, 1, 3) * span * 0.1  # Scale factor
                    deformed_grid = sampling_grid + offset
                    deformed_grids.append(deformed_grid)

                # Sample features at deformed locations
                sampled_features = []
                for k, grid in enumerate(deformed_grids):
                    feat = F.grid_sample(
                        F_visual_item,
                        grid,
                        mode='bilinear',
                        padding_mode='zeros',
                        align_corners=False
                    )  # [1, D, Oz, Oy, Ox]
                    sampled_features.append(feat)

                # Stack and weight by attention
                stacked_features = torch.stack(sampled_features, dim=1)  # [1, K, D, Oz, Oy, Ox]

                # Get attention weights for this RoI
                roi_attn = attn_weights[0, :, center_coord[0, 0, 0, 0, 2],
                                       center_coord[0, 0, 0, 0, 1],
                                       center_coord[0, 0, 0, 0, 0]]  # [K]
                roi_attn = roi_attn.view(1, self.num_sample_points, 1, 1, 1, 1)

                # Weighted combination
                weighted_features = (stacked_features * roi_attn).sum(dim=1)  # [1, D, Oz, Oy, Ox]

                # Pool and project
                pooled = self.pooler(weighted_features)  # [1, D, Oz, Oy, Ox]
                flattened = pooled.mean(dim=(2, 3, 4))  # [1, D]
                projected = self.projection(flattened)  # [1, d_model]

                roi_features_list.append(projected)

            # Concatenate all RoIs for this batch item
            if roi_features_list:
                batch_roi_features = torch.cat(roi_features_list, dim=0)  # [num_rois, d_model]
                all_rois_features.append(batch_roi_features)
            else:
                all_rois_features.append(torch.empty(0, self.projection[-1].out_features, device=device))

        return all_rois_features


class SpatialPyramidRoI3D(nn.Module):
    """
    RoI Align with Spatial Pyramid Pooling (SPP)

    Multi-scale pooling to capture features at different granularities
    """
    def __init__(self,
                 input_dim: int = 512,
                 pyramid_levels: tuple = (1, 2, 4),
                 d_model: int = 512):
        """
        Args:
            input_dim: Feature dimension
            pyramid_levels: Sizes for pyramid pooling (e.g., 1x1x1, 2x2x2, 4x4x4)
            d_model: Output dimension
        """
        super().__init__()

        self.pyramid_levels = pyramid_levels

        # Poolers for each pyramid level
        self.poolers = nn.ModuleList([
            nn.AdaptiveAvgPool3d((size, size, size))
            for size in pyramid_levels
        ])

        # Calculate total feature dimension after concatenation
        total_bins = sum(size ** 3 for size in pyramid_levels)
        concat_dim = input_dim * total_bins

        # Projection to output dimension
        self.projection = nn.Sequential(
            nn.Linear(concat_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )

        # Base sampling grid (7x7x7 for initial sampling)
        output_size = (7, 7, 7)
        grid_z, grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, output_size[0]),
            torch.linspace(-1.0, 1.0, output_size[1]),
            torch.linspace(-1.0, 1.0, output_size[2]),
            indexing="ij"
        )
        self.register_buffer('base_grid',
                           torch.stack((grid_x, grid_y, grid_z), dim=-1).unsqueeze(0))

    def forward(self,
                F_visual: torch.Tensor,
                boxes_list: list,
                original_spatial_dims: tuple):
        """
        Args:
            F_visual: [B, D, T, H, W]
            boxes_list: List of [num_rois, 6]
            original_spatial_dims: (T, H, W)

        Returns:
            all_rois_features: List of [num_rois, d_model]
        """
        all_rois_features = []
        B, D, T_feat, H_feat, W_feat = F_visual.shape
        device = F_visual.device

        orig_dims_tensor = torch.tensor(
            [original_spatial_dims[2], original_spatial_dims[1], original_spatial_dims[0]],
            device=device, dtype=torch.float32
        ).unsqueeze(0)

        base_grid = self.base_grid.to(device)

        for i in range(B):
            F_visual_item = F_visual[i:i+1]
            boxes = boxes_list[i]

            if boxes.shape[0] == 0:
                all_rois_features.append(torch.empty(0, self.projection[-1].out_features, device=device))
                continue

            # Normalize boxes
            boxes_min_zyx = boxes[:, 0:3]
            boxes_max_zyx = boxes[:, 3:6]

            boxes_min_xyz = torch.stack([boxes_min_zyx[:, 2],
                                        boxes_min_zyx[:, 1],
                                        boxes_min_zyx[:, 0]], dim=1)
            boxes_max_xyz = torch.stack([boxes_max_zyx[:, 2],
                                        boxes_max_zyx[:, 1],
                                        boxes_max_zyx[:, 0]], dim=1)

            norm_boxes_min_xyz = (boxes_min_xyz / orig_dims_tensor) * 2 - 1
            norm_boxes_max_xyz = (boxes_max_xyz / orig_dims_tensor) * 2 - 1

            box_centers = (norm_boxes_max_xyz + norm_boxes_min_xyz) / 2.0
            box_spans = (norm_boxes_max_xyz - norm_boxes_min_xyz) / 2.0

            box_centers_view = box_centers.view(-1, 1, 1, 1, 3)
            box_spans_view = box_spans.view(-1, 1, 1, 1, 3)

            # Create sampling grid
            sampling_grid = base_grid * box_spans_view + box_centers_view  # [num_rois, 7, 7, 7, 3]

            # Sample features
            F_visual_repeated = F_visual_item.repeat(boxes.shape[0], 1, 1, 1, 1)
            roi_features = F.grid_sample(
                F_visual_repeated,
                sampling_grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )  # [num_rois, D, 7, 7, 7]

            # Apply spatial pyramid pooling
            pyramid_features = []
            for pooler in self.poolers:
                pooled = pooler(roi_features)  # [num_rois, D, size, size, size]
                pooled_flat = pooled.flatten(2)  # [num_rois, D, size^3]
                pooled_flat = rearrange(pooled_flat, 'n d s -> n (d s)')
                pyramid_features.append(pooled_flat)

            # Concatenate all pyramid levels
            multi_scale_features = torch.cat(pyramid_features, dim=1)  # [num_rois, concat_dim]

            # Project to output dimension
            projected_features = self.projection(multi_scale_features)  # [num_rois, d_model]

            all_rois_features.append(projected_features)

        return all_rois_features


# Example usage and testing
if __name__ == "__main__":
    print("=== Testing Deformable RoI Align 3D ===\n")

    # Setup
    batch_size = 2
    input_dim = 512
    d_model = 512
    T, H, W = 21, 24, 24

    F_visual = torch.randn(batch_size, input_dim, T, H, W)
    print(f"Input F_visual: {F_visual.shape}")

    # Create dummy boxes
    boxes_0 = torch.tensor([
        [10, 20, 30, 40, 50, 60],
        [100, 50, 50, 120, 80, 80],
    ], dtype=torch.float32)
    boxes_1 = torch.tensor([
        [5, 15, 25, 35, 45, 55],
    ], dtype=torch.float32)
    boxes_list = [boxes_0, boxes_1]
    orig_dims = (201, 160, 160)

    # Test 1: Spatial Pyramid RoI (recommended for stability)
    print("\n[Test 1] Spatial Pyramid RoI 3D")
    spp_roi = SpatialPyramidRoI3D(
        input_dim=input_dim,
        pyramid_levels=(1, 2, 4),
        d_model=d_model
    )

    roi_features = spp_roi(F_visual, boxes_list, orig_dims)
    print(f"  Item 0: {roi_features[0].shape} (expected: [2, 512])")
    print(f"  Item 1: {roi_features[1].shape} (expected: [1, 512])")
    assert roi_features[0].shape == (2, d_model)
    assert roi_features[1].shape == (1, d_model)
    print("  ✓ SPP RoI working correctly!")

    # Test 2: Deformable RoI (more advanced, may need tuning)
    print("\n[Test 2] Deformable RoI Align 3D")
    deform_roi = DeformableRoIAlign3D(
        input_dim=input_dim,
        output_size=(7, 7, 7),
        d_model=d_model,
        num_sample_points=4
    )

    roi_features_deform = deform_roi(F_visual, boxes_list, orig_dims)
    print(f"  Item 0: {roi_features_deform[0].shape}")
    print(f"  Item 1: {roi_features_deform[1].shape}")
    print("  ✓ Deformable RoI working!")

    print("\n=== All tests passed! ===")
