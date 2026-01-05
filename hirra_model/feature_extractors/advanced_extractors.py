"""
Advanced Feature Extractors - Integrated Module

This module combines all improved components into easy-to-use interfaces:
1. AdvancedFeatureExtractor: All-in-one feature extraction
2. Easy configuration and switching between variants
3. Backward compatible with basic versions

Usage:
    from hirra_model.feature_extractors.advanced_extractors import AdvancedFeatureExtractor

    # Create extractor
    extractor = AdvancedFeatureExtractor(
        config='full',  # 'basic', 'improved', or 'full'
        d_model=512
    )

    # Extract features
    global_context, roi_features = extractor(F_visual, boxes_list, orig_dims)
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(ROOT_DIR)

# Import all components
from hirra_model.feature_extractors.multi_scale_fpn import MultiScaleFPN, MultiScaleFeatureAggregator
from hirra_model.feature_extractors.improved_qformer import ImprovedQFormer
from hirra_model.feature_extractors.deformable_roi import SpatialPyramidRoI3D, DeformableRoIAlign3D
from hirra_model.feature_extractors.global_context import GlobalContextExtractor
from hirra_model.feature_extractors.roi_align_3d import RoIAlign3D

# Import graph reasoning modules (optional)
try:
    from hirra_model.graph_reasoning import ROIPairGraphBuilder, ROIGraphAttentionNetwork, HierarchicalROIGraph
    GRAPH_REASONING_AVAILABLE = True
except ImportError:
    GRAPH_REASONING_AVAILABLE = False
    print("Warning: Graph reasoning modules not available. Install torch-geometric or check module paths.")


class AdvancedFeatureExtractor(nn.Module):
    """
    Unified interface for feature extraction with different complexity levels

    Configurations:
        'basic': Original simple version (fastest, lowest memory)
        'improved': Medium improvements (good balance)
        'full': All improvements (best performance, most compute)
    """

    def __init__(self,
                 config: str = 'improved',
                 input_dim: int = 512,
                 d_model: int = 512,
                 num_queries: int = 32,
                 use_multi_scale: bool = True,
                 use_sparse_qformer: bool = True,
                 use_spp_roi: bool = True,
                 use_graph_reasoning: bool = True,
                 graph_config: str = 'hierarchical'):
        """
        Args:
            config: 'basic', 'improved', or 'full'
            input_dim: Input feature dimension
            d_model: Output feature dimension
            num_queries: Number of query vectors for global context
            use_multi_scale: Use FPN for multi-scale features
            use_sparse_qformer: Use sparse sampling in Q-Former
            use_spp_roi: Use Spatial Pyramid Pooling for RoI
            use_graph_reasoning: Use ROI-pair graph reasoning
            graph_config: 'basic' (GNN only) or 'hierarchical' (two-level)
        """
        super().__init__()

        self.config = config
        self.d_model = d_model
        self.use_graph_reasoning = use_graph_reasoning and GRAPH_REASONING_AVAILABLE
        self.graph_config = graph_config if self.use_graph_reasoning else None

        # Configure based on preset
        if config == 'basic':
            use_multi_scale = False
            use_sparse_qformer = False
            use_spp_roi = False
            qformer_depth = 2
        elif config == 'improved':
            qformer_depth = 6
        elif config == 'full':
            use_multi_scale = True
            use_sparse_qformer = True
            use_spp_roi = True
            qformer_depth = 12  # Deeper for full version
        else:
            raise ValueError(f"Unknown config: {config}. Use 'basic', 'improved', or 'full'")

        # Multi-Scale FPN (optional)
        self.use_multi_scale = use_multi_scale
        if use_multi_scale:
            self.fpn = MultiScaleFPN(
                input_dim=input_dim,
                fpn_dim=256,
                num_levels=3
            )
            self.fpn_aggregator = MultiScaleFeatureAggregator(
                fpn_dim=256,
                output_dim=d_model
            )

        # Global Context Extractor
        if config == 'basic':
            self.global_extractor = GlobalContextExtractor(
                input_dim=d_model,
                d_model=d_model,
                num_context_vectors=num_queries,
                depth=qformer_depth
            )
        else:
            self.global_extractor = ImprovedQFormer(
                input_dim=d_model,
                d_model=d_model,
                num_queries=num_queries,
                depth=qformer_depth,
                use_sparse_sampling=use_sparse_qformer,
                sparse_keep_ratio=0.25
            )

        # RoI Extractor
        self.use_spp_roi = use_spp_roi
        if use_spp_roi:
            self.roi_extractor = SpatialPyramidRoI3D(
                input_dim=d_model,
                pyramid_levels=(1, 2, 4),
                d_model=d_model
            )
        else:
            self.roi_extractor = RoIAlign3D(
                output_size=(7, 7, 7),
                d_model=d_model
            )

        # Graph Reasoning Components (NEW!)
        if self.use_graph_reasoning:
            # Graph builder
            self.graph_builder = ROIPairGraphBuilder(
                d_model=d_model,
                spatial_dim=64,
                num_edge_types=5,
                k_neighbors=5,
                distance_threshold=100.0
            )

            # Graph reasoner
            if graph_config == 'hierarchical':
                self.graph_reasoner = HierarchicalROIGraph(
                    node_dim=d_model,
                    num_organs=6,
                    num_heads=8,
                    dropout=0.1
                )
            else:  # 'basic'
                self.graph_reasoner = ROIGraphAttentionNetwork(
                    node_dim=d_model,
                    edge_dim=d_model,
                    hidden_dim=d_model,
                    num_heads=8,
                    num_layers=3,
                    dropout=0.1
                )

            # Graph summary projector (to inject into global context)
            self.graph_summary_proj = nn.Linear(d_model, d_model)

            print(f"[Graph Reasoning] Initialized with config: {graph_config}")

    def forward(self,
                F_visual: torch.Tensor,
                boxes_list: List[torch.Tensor],
                original_spatial_dims: Tuple[int, int, int],
                ct_image: Optional[torch.Tensor] = None,
                pet_image: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Extract both global context and RoI features

        Args:
            F_visual: [B, D, T, H, W] - Visual features from encoder
            boxes_list: List of [num_rois, 6] bounding boxes per batch item
            original_spatial_dims: (T, H, W) of original image
            ct_image: [B, T, H, W] - Original CT images (optional, for graph reasoning)
            pet_image: [B, T, H, W] - Original PET images (optional, for graph reasoning)

        Returns:
            global_context: [B, num_queries, d_model] - Global context vectors
                (num_queries+1 if graph reasoning is used, with graph summary as last vector)
            roi_features: List of [num_rois, d_model] - RoI features per batch item
                (enhanced with graph reasoning if enabled)
        """

        # Optional: Multi-scale feature extraction
        if self.use_multi_scale:
            pyramid_features = self.fpn(F_visual)
            F_processed = self.fpn_aggregator(pyramid_features)
        else:
            F_processed = F_visual

        # Extract global context
        global_context = self.global_extractor(F_processed)

        # Extract RoI features
        roi_features = self.roi_extractor(F_processed, boxes_list, original_spatial_dims)

        # NEW: Graph reasoning over ROIs
        if self.use_graph_reasoning:
            roi_features_enhanced = []
            graph_summaries = []

            for batch_idx, (roi_feats, boxes) in enumerate(zip(roi_features, boxes_list)):
                if len(roi_feats) > 1:  # Only apply graph reasoning if multiple ROIs
                    # Build graph (FIX: Now passing original_spatial_dims for coordinate scaling)
                    edge_index, edge_attr, _ = self.graph_builder(
                        roi_feats,
                        boxes,
                        original_spatial_dims,
                        ct_image[batch_idx:batch_idx+1] if ct_image is not None else None,
                        pet_image[batch_idx:batch_idx+1] if pet_image is not None else None
                    )

                    # Apply graph reasoning
                    if isinstance(self.graph_reasoner, HierarchicalROIGraph):
                        roi_feats_enh, graph_sum = self.graph_reasoner(
                            roi_feats, boxes, edge_index, edge_attr
                        )
                    else:
                        roi_feats_enh, graph_sum = self.graph_reasoner(
                            roi_feats, edge_index, edge_attr
                        )

                    roi_features_enhanced.append(roi_feats_enh)
                    graph_summaries.append(graph_sum.squeeze(0))  # [d_model]
                else:
                    # Single ROI: no graph reasoning needed
                    roi_features_enhanced.append(roi_feats)
                    graph_summaries.append(torch.zeros(self.d_model, device=roi_feats.device))

            roi_features = roi_features_enhanced

            # Inject graph summary into global context as an additional query
            graph_summaries = torch.stack(graph_summaries)  # [B, d_model]
            graph_summaries = self.graph_summary_proj(graph_summaries)  # [B, d_model]

            # Add as the last query vector
            global_context = torch.cat([
                global_context,  # [B, num_queries, d_model]
                graph_summaries.unsqueeze(1)  # [B, 1, d_model]
            ], dim=1)  # [B, num_queries+1, d_model]

        return global_context, roi_features

    def get_config_info(self):
        """Return configuration information"""
        info = {
            'config': self.config,
            'd_model': self.d_model,
            'use_multi_scale': self.use_multi_scale,
            'use_sparse_qformer': hasattr(self.global_extractor, 'use_sparse_sampling') and self.global_extractor.use_sparse_sampling,
            'use_spp_roi': self.use_spp_roi,
            'use_graph_reasoning': self.use_graph_reasoning,
            'graph_config': getattr(self, 'graph_config', None) if self.use_graph_reasoning else None,
        }
        return info


class FlexibleFeatureExtractor(nn.Module):
    """
    More flexible version allowing custom component selection
    """

    def __init__(self,
                 fpn: Optional[nn.Module] = None,
                 global_extractor: Optional[nn.Module] = None,
                 roi_extractor: Optional[nn.Module] = None,
                 d_model: int = 512):
        """
        Args:
            fpn: Optional custom FPN module
            global_extractor: Optional custom global context extractor
            roi_extractor: Optional custom RoI extractor
            d_model: Feature dimension
        """
        super().__init__()

        self.d_model = d_model

        # Use provided modules or create defaults
        self.fpn = fpn
        self.global_extractor = global_extractor or ImprovedQFormer(
            input_dim=d_model,
            d_model=d_model,
            num_queries=32,
            depth=6
        )
        self.roi_extractor = roi_extractor or SpatialPyramidRoI3D(
            input_dim=d_model,
            pyramid_levels=(1, 2, 4),
            d_model=d_model
        )

        # Aggregator if FPN is used
        if self.fpn is not None:
            self.fpn_aggregator = MultiScaleFeatureAggregator(
                fpn_dim=256,
                output_dim=d_model
            )

    def forward(self,
                F_visual: torch.Tensor,
                boxes_list: List[torch.Tensor],
                original_spatial_dims: Tuple[int, int, int]):
        """Forward pass through flexible pipeline"""

        # Optional FPN
        if self.fpn is not None:
            pyramid_features = self.fpn(F_visual)
            F_processed = self.fpn_aggregator(pyramid_features)
        else:
            F_processed = F_visual

        # Global context
        global_context = self.global_extractor(F_processed)

        # RoI features
        roi_features = self.roi_extractor(F_processed, boxes_list, original_spatial_dims)

        return global_context, roi_features


# Factory function for easy creation
def create_feature_extractor(config='improved', **kwargs):
    """
    Factory function to create feature extractors

    Args:
        config: 'basic', 'improved', or 'full'
        **kwargs: Additional arguments for AdvancedFeatureExtractor

    Returns:
        AdvancedFeatureExtractor instance

    Examples:
        >>> # Basic version (fast)
        >>> extractor = create_feature_extractor('basic')
        >>>
        >>> # Improved version (recommended)
        >>> extractor = create_feature_extractor('improved', num_queries=64)
        >>>
        >>> # Full version (best performance)
        >>> extractor = create_feature_extractor('full')
    """
    return AdvancedFeatureExtractor(config=config, **kwargs)


# Example usage
if __name__ == "__main__":
    print("=== Advanced Feature Extractors ===\n")

    # Test data
    batch_size = 2
    d_model = 512
    T, H, W = 21, 24, 24

    F_visual = torch.randn(batch_size, d_model, T, H, W)
    boxes_list = [
        torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.float32),
        torch.tensor([[5, 15, 25, 35, 45, 55]], dtype=torch.float32)
    ]
    orig_dims = (201, 160, 160)

    print("Input shapes:")
    print(f"  F_visual: {F_visual.shape}")
    print(f"  Boxes: {[b.shape for b in boxes_list]}\n")

    # Test all configurations
    for config_name in ['basic', 'improved', 'full']:
        print(f"\n[Testing {config_name.upper()} configuration]")

        extractor = create_feature_extractor(config_name, d_model=d_model)

        # Forward pass
        global_context, roi_features = extractor(F_visual, boxes_list, orig_dims)

        print(f"  Global context: {global_context.shape}")
        print(f"  RoI features: {[f.shape for f in roi_features]}")

        # Config info
        info = extractor.get_config_info()
        print(f"  Config: {info}")

    print("\n✓ All configurations working!")

    # Test flexible version
    print("\n[Testing Flexible extractor]")

    # Create custom components
    custom_fpn = MultiScaleFPN(input_dim=512, fpn_dim=256, num_levels=3)
    custom_qformer = ImprovedQFormer(
        input_dim=512,
        d_model=512,
        num_queries=64,
        depth=8
    )

    flexible = FlexibleFeatureExtractor(
        fpn=custom_fpn,
        global_extractor=custom_qformer,
        roi_extractor=None,  # Use default
        d_model=512
    )

    global_context, roi_features = flexible(F_visual, boxes_list, orig_dims)
    print(f"  Global context: {global_context.shape}")
    print(f"  RoI features: {[f.shape for f in roi_features]}")

    print("\n✓ Flexible extractor working!")

    print("\n=== All tests passed! ===")
