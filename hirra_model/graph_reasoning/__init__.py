"""
Graph Reasoning Module for HIRRA Model

This module provides explicit ROI-pair graph reasoning capabilities to model
spatial relationships, anatomical context, and disease progression patterns
between regions of interest.

Components:
- ROIPairGraphBuilder: Constructs graphs from ROI pairs with spatial and intensity relations
- ROIGraphAttentionNetwork: Graph attention network for message passing between ROIs
- HierarchicalROIGraph: Two-level graph reasoning (ROI-level + Organ-level)
"""

from .roi_graph_builder import ROIPairGraphBuilder
from .roi_gnn import ROIGraphAttentionNetwork
from .hierarchical_graph import HierarchicalROIGraph

__all__ = [
    'ROIPairGraphBuilder',
    'ROIGraphAttentionNetwork',
    'HierarchicalROIGraph',
]
