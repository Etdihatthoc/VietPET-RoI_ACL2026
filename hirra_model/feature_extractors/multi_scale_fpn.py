"""
Multi-Scale Feature Pyramid Network (FPN) for 3D Medical Images

Improvements over basic feature extraction:
1. Multi-scale feature pyramids (P3, P4, P5)
2. Top-down pathway with lateral connections
3. Preserves both fine-grained and semantic information

References:
- Feature Pyramid Networks for Object Detection (Lin et al., CVPR 2017)
- 3D U-Net: Learning Dense Volumetric Segmentation (Çiçek et al., MICCAI 2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Conv3dNormAct(nn.Module):
    """3D Convolution with Normalization and Activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class MultiScaleFPN(nn.Module):
    def __init__(self,
                 input_dim: int = 512,
                 fpn_dim: int = 256,
                 num_levels: int = 3):
        """
        Multi-Scale Feature Pyramid Network for 3D features

        Args:
            input_dim: Dimension of input features (from vision encoder)
            fpn_dim: Dimension of FPN features (typically 256)
            num_levels: Number of pyramid levels (default: 3 for P3, P4, P5)

        Input:
            F_visual: [B, input_dim, T, H, W]

        Output:
            pyramid_features: List of [B, fpn_dim, T_i, H_i, W_i] for i in levels
                            Ordered from finest to coarsest [P3, P4, P5]
        """
        super().__init__()

        self.num_levels = num_levels
        self.fpn_dim = fpn_dim

        # 1x1 lateral connections (reduce channel dimension)
        # First level: input_dim -> fpn_dim
        # Other levels: fpn_dim -> fpn_dim (after downsampling)
        self.lateral_convs = nn.ModuleList([
            nn.Conv3d(input_dim if i == 0 else fpn_dim, fpn_dim, kernel_size=1, bias=False)
            for i in range(num_levels)
        ])

        # 3x3 convolutions for smoothing after upsampling
        self.fpn_convs = nn.ModuleList([
            Conv3dNormAct(fpn_dim, fpn_dim, kernel_size=3, padding=1)
            for _ in range(num_levels)
        ])

        # Downsampling layers to create multi-scale inputs
        self.downsample_layers = nn.ModuleList()
        for i in range(num_levels - 1):
            # Strided convolution for downsampling
            self.downsample_layers.append(
                nn.Sequential(
                    nn.Conv3d(input_dim if i == 0 else fpn_dim,
                             fpn_dim,
                             kernel_size=3,
                             stride=2,
                             padding=1,
                             bias=False),
                    nn.GroupNorm(num_groups=32, num_channels=fpn_dim),
                    nn.GELU()
                )
            )

    def forward(self, F_visual: torch.Tensor):
        """
        Forward pass through FPN

        Args:
            F_visual: [B, input_dim, T, H, W] - e.g., [2, 512, 21, 8, 8]

        Returns:
            pyramid_features: List of 3 tensors
                P3: [B, fpn_dim, T, H, W]       - Finest (original resolution)
                P4: [B, fpn_dim, T/2, H/2, W/2] - Medium
                P5: [B, fpn_dim, T/4, H/4, W/4] - Coarsest
        """

        # Build bottom-up pathway (create multi-scale features)
        # Start with input resolution and progressively downsample
        bottom_up_features = [F_visual]

        x = F_visual
        for downsample in self.downsample_layers:
            x = downsample(x)
            bottom_up_features.append(x)

        # bottom_up_features: [C3, C4, C5] where Ci has resolution 1/2^(i-3)

        # Apply lateral connections to each level
        lateral_features = []
        for i, (lateral_conv, bottom_up_feat) in enumerate(zip(self.lateral_convs, bottom_up_features)):
            lateral = lateral_conv(bottom_up_feat)
            lateral_features.append(lateral)

        # Build top-down pathway with lateral connections
        # Start from coarsest level (P5) and propagate up
        pyramid_features = []

        # P5 (coarsest)
        p5 = lateral_features[-1]
        p5 = self.fpn_convs[-1](p5)
        pyramid_features.append(p5)

        # Build P4, P3, ... by upsampling and adding lateral connections
        for i in range(len(lateral_features) - 2, -1, -1):
            # Get lateral feature at this level
            lateral = lateral_features[i]

            # Upsample from coarser level
            prev_feature = pyramid_features[-1]
            upsampled = F.interpolate(
                prev_feature,
                size=lateral.shape[2:],  # Match spatial dimensions
                mode='trilinear',
                align_corners=False
            )

            # Add lateral connection
            fused = upsampled + lateral

            # Apply smoothing convolution
            fused = self.fpn_convs[i](fused)

            pyramid_features.append(fused)

        # Reverse to get [P3, P4, P5] (finest to coarsest)
        pyramid_features = pyramid_features[::-1]

        return pyramid_features


class MultiScaleFeatureAggregator(nn.Module):
    """
    Aggregate multi-scale features into a single representation
    """
    def __init__(self, fpn_dim: int = 256, output_dim: int = 512):
        super().__init__()

        # Projection heads for each scale
        self.scale_projections = nn.ModuleList([
            nn.Conv3d(fpn_dim, output_dim, kernel_size=1)
            for _ in range(3)  # 3 scales
        ])

        # Attention weights for scale selection
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(output_dim * 3, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, pyramid_features: list):
        """
        Aggregate multi-scale features with learned attention weights

        Args:
            pyramid_features: List of [B, fpn_dim, T_i, H_i, W_i]

        Returns:
            aggregated: [B, output_dim, T, H, W] at finest resolution
        """
        B = pyramid_features[0].shape[0]
        target_size = pyramid_features[0].shape[2:]  # Use P3 size (finest)

        # Project and resize all scales to target resolution
        scaled_features = []
        for i, feat in enumerate(pyramid_features):
            # Project to output dimension
            proj_feat = self.scale_projections[i](feat)

            # Resize to target size if needed
            if proj_feat.shape[2:] != target_size:
                proj_feat = F.interpolate(
                    proj_feat,
                    size=target_size,
                    mode='trilinear',
                    align_corners=False
                )

            scaled_features.append(proj_feat)

        # Concatenate for attention
        concat_features = torch.cat(scaled_features, dim=1)  # [B, output_dim*3, T, H, W]

        # Compute attention weights
        attn_weights = self.scale_attention(concat_features)  # [B, 3, 1, 1, 1]

        # Weighted sum of features
        aggregated = sum(
            w * feat
            for w, feat in zip(attn_weights.split(1, dim=1), scaled_features)
        )

        return aggregated


# Example usage
if __name__ == "__main__":
    # Test FPN
    batch_size = 2
    input_dim = 512
    T, H, W = 21, 24, 24

    # Create dummy input
    F_visual = torch.randn(batch_size, input_dim, T, H, W)

    # Initialize FPN
    fpn = MultiScaleFPN(input_dim=input_dim, fpn_dim=256, num_levels=3)

    # Forward pass
    pyramid_features = fpn(F_visual)

    print("FPN Output:")
    for i, feat in enumerate(pyramid_features):
        print(f"  P{i+3}: {feat.shape}")

    # Test aggregator
    aggregator = MultiScaleFeatureAggregator(fpn_dim=256, output_dim=512)
    aggregated = aggregator(pyramid_features)
    print(f"\nAggregated: {aggregated.shape}")
