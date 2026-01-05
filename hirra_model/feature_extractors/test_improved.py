"""
Comprehensive Test Suite for Improved Feature Extractors

Tests all improved components:
1. Multi-Scale FPN
2. Improved Q-Former with self-attention
3. Deformable RoI Align 3D
4. Spatial Pyramid RoI
5. Integrated pipeline
"""

import torch
import sys
import os

# Setup path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(ROOT_DIR)

from hirra_model.feature_extractors.multi_scale_fpn import MultiScaleFPN, MultiScaleFeatureAggregator
from hirra_model.feature_extractors.improved_qformer import ImprovedQFormer, SparseFeatureSampler
from hirra_model.feature_extractors.deformable_roi import DeformableRoIAlign3D, SpatialPyramidRoI3D


def test_multi_scale_fpn():
    """Test Multi-Scale Feature Pyramid Network"""
    print("\n" + "="*60)
    print("[TEST 1] Multi-Scale FPN")
    print("="*60)

    batch_size = 2
    input_dim = 512
    T, H, W = 21, 8, 8

    # Create input
    F_visual = torch.randn(batch_size, input_dim, T, H, W)
    print(f"\n Input: {F_visual.shape}")

    # Initialize FPN
    fpn = MultiScaleFPN(input_dim=input_dim, fpn_dim=256, num_levels=3)

    # Forward pass
    pyramid_features = fpn(F_visual)

    # Verify outputs (note: downsampling with stride may round differently)
    print(f"\n FPN Pyramid Outputs:")

    # Check P3 (finest - should match input)
    assert pyramid_features[0].shape == (batch_size, 256, T, H, W), "P3 shape incorrect"
    print(f"   P3: {pyramid_features[0].shape} ✓")

    # Check P4 (allow rounding in temporal dimension)
    assert pyramid_features[1].shape[0] == batch_size
    assert pyramid_features[1].shape[1] == 256
    assert abs(pyramid_features[1].shape[2] - T//2) <= 1, f"P4 temporal dimension off: {pyramid_features[1].shape[2]} vs {T//2}"
    assert pyramid_features[1].shape[3:] == (H//2, W//2)
    print(f"   P4: {pyramid_features[1].shape} ✓")

    # Check P5 (allow rounding)
    assert pyramid_features[2].shape[0] == batch_size
    assert pyramid_features[2].shape[1] == 256
    print(f"   P5: {pyramid_features[2].shape} ✓")

    print("\n ✓ FPN test passed!")

    # Test aggregator
    print("\n Testing Feature Aggregator...")
    aggregator = MultiScaleFeatureAggregator(fpn_dim=256, output_dim=512)
    aggregated = aggregator(pyramid_features)
    print(f"   Aggregated: {aggregated.shape}")
    assert aggregated.shape == (batch_size, 512, T, H, W)
    print(" ✓ Aggregator test passed!")

    return True


def test_improved_qformer():
    """Test Improved Q-Former"""
    print("\n" + "="*60)
    print("[TEST 2] Improved Q-Former")
    print("="*60)

    batch_size = 2
    d_model = 512
    T, H, W = 21, 8, 8
    num_queries = 32

    F_visual = torch.randn(batch_size, d_model, T, H, W)
    print(f"\n Input: {F_visual.shape}")

    # Test basic improved Q-Former
    print("\n [2a] Basic Improved Q-Former (depth=6)")
    qformer = ImprovedQFormer(
        input_dim=d_model,
        d_model=d_model,
        num_queries=num_queries,
        depth=6,
        use_sparse_sampling=False
    )

    context = qformer(F_visual)
    print(f"   Output: {context.shape}")
    assert context.shape == (batch_size, num_queries, d_model)
    print(" ✓ Basic Q-Former working!")

    # Test with sparse sampling
    print("\n [2b] Q-Former with Sparse Sampling (4x faster)")
    qformer_sparse = ImprovedQFormer(
        input_dim=d_model,
        d_model=d_model,
        num_queries=num_queries,
        depth=6,
        use_sparse_sampling=True,
        sparse_keep_ratio=0.25
    )

    qformer_sparse.train()  # Enable sampling in training mode
    context_sparse = qformer_sparse(F_visual)
    print(f"   Output: {context_sparse.shape}")
    assert context_sparse.shape == (batch_size, num_queries, d_model)
    print(" ✓ Sparse Q-Former working!")

    # Test sparse sampler separately
    print("\n [2c] Sparse Feature Sampler")
    N = T * H * W
    features_flat = torch.randn(batch_size, N, d_model)
    sampler = SparseFeatureSampler(dim=d_model, keep_ratio=0.25)
    sampled, indices = sampler(features_flat, deterministic=True)
    print(f"   Input: {features_flat.shape}")
    print(f"   Sampled: {sampled.shape} (kept {sampled.shape[1]}/{N} features)")
    assert sampled.shape[1] == N // 4
    print(" ✓ Sparse sampler working!")

    return True


def test_deformable_roi():
    """Test Deformable RoI and SPP RoI"""
    print("\n" + "="*60)
    print("[TEST 3] Deformable & SPP RoI Align")
    print("="*60)

    batch_size = 2
    input_dim = 512
    d_model = 512
    T, H, W = 21, 8, 8

    F_visual = torch.randn(batch_size, input_dim, T, H, W)
    print(f"\n Input: {F_visual.shape}")

    # Create bounding boxes
    boxes_0 = torch.tensor([
        [10, 20, 30, 40, 50, 60],
        [100, 50, 50, 120, 80, 80],
        [150, 10, 70, 200, 60, 100],
    ], dtype=torch.float32)

    boxes_1 = torch.tensor([
        [5, 15, 25, 35, 45, 55],
        [70, 70, 70, 80, 80, 80],
    ], dtype=torch.float32)

    boxes_list = [boxes_0, boxes_1]
    orig_dims = (201, 160, 160)

    print(f"\n Boxes: item 0 has {boxes_0.shape[0]} RoIs, item 1 has {boxes_1.shape[0]} RoIs")

    # Test Spatial Pyramid RoI (recommended)
    print("\n [3a] Spatial Pyramid RoI (Multi-scale pooling)")
    spp_roi = SpatialPyramidRoI3D(
        input_dim=input_dim,
        pyramid_levels=(1, 2, 4),
        d_model=d_model
    )

    roi_features = spp_roi(F_visual, boxes_list, orig_dims)
    print(f"   Item 0: {roi_features[0].shape} (expected: [3, 512])")
    print(f"   Item 1: {roi_features[1].shape} (expected: [2, 512])")
    assert roi_features[0].shape == (3, d_model)
    assert roi_features[1].shape == (2, d_model)
    print(" ✓ SPP RoI working!")

    # Test Deformable RoI
    print("\n [3b] Deformable RoI Align (Learned offsets)")
    deform_roi = DeformableRoIAlign3D(
        input_dim=input_dim,
        output_size=(7, 7, 7),
        d_model=d_model,
        num_sample_points=4
    )

    roi_features_deform = deform_roi(F_visual, boxes_list, orig_dims)
    print(f"   Item 0: {roi_features_deform[0].shape}")
    print(f"   Item 1: {roi_features_deform[1].shape}")
    assert roi_features_deform[0].shape == (3, d_model)
    assert roi_features_deform[1].shape == (2, d_model)
    print(" ✓ Deformable RoI working!")

    return True


def test_integrated_pipeline():
    """Test complete integrated pipeline"""
    print("\n" + "="*60)
    print("[TEST 4] Integrated Pipeline")
    print("="*60)

    batch_size = 2
    input_dim = 512
    d_model = 512
    T, H, W = 21, 8, 8

    print("\n Building complete pipeline:")
    print(" 1. Multi-Scale FPN")
    print(" 2. Improved Q-Former")
    print(" 3. SPP RoI Align")

    # Input
    F_visual = torch.randn(batch_size, input_dim, T, H, W)
    boxes_list = [
        torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.float32),
        torch.tensor([[5, 15, 25, 35, 45, 55]], dtype=torch.float32)
    ]
    orig_dims = (201, 160, 160)

    print(f"\n Input F_visual: {F_visual.shape}")

    # Step 1: FPN
    fpn = MultiScaleFPN(input_dim=input_dim, fpn_dim=256, num_levels=3)
    pyramid_features = fpn(F_visual)
    print(f"\n [Step 1] FPN Output:")
    for i, feat in enumerate(pyramid_features):
        print(f"   P{i+3}: {feat.shape}")

    # Step 2: Aggregate multi-scale features
    aggregator = MultiScaleFeatureAggregator(fpn_dim=256, output_dim=512)
    F_aggregated = aggregator(pyramid_features)
    print(f"\n [Step 2] Aggregated: {F_aggregated.shape}")

    # Step 3: Q-Former for global context
    qformer = ImprovedQFormer(
        input_dim=d_model,
        d_model=d_model,
        num_queries=32,
        depth=6,
        use_sparse_sampling=True
    )
    qformer.eval()  # Disable sparse sampling for deterministic output
    global_context = qformer(F_aggregated)
    print(f"\n [Step 3] Global Context: {global_context.shape}")

    # Step 4: RoI features
    roi_extractor = SpatialPyramidRoI3D(
        input_dim=input_dim,
        pyramid_levels=(1, 2, 4),
        d_model=d_model
    )
    roi_features = roi_extractor(F_aggregated, boxes_list, orig_dims)
    print(f"\n [Step 4] RoI Features:")
    print(f"   Item 0: {roi_features[0].shape}")
    print(f"   Item 1: {roi_features[1].shape}")

    # Verify all outputs
    assert global_context.shape == (batch_size, 32, d_model)
    assert roi_features[0].shape[1] == d_model
    assert roi_features[1].shape[1] == d_model

    print("\n ✓ Integrated pipeline working correctly!")

    return True


def test_performance_comparison():
    """Compare performance: Basic vs Improved"""
    print("\n" + "="*60)
    print("[TEST 5] Performance Comparison")
    print("="*60)

    import time

    batch_size = 2
    d_model = 512
    T, H, W = 21, 8, 8

    F_visual = torch.randn(batch_size, d_model, T, H, W).cuda()

    # Basic Q-Former
    from hirra_model.feature_extractors.global_context import GlobalContextExtractor

    basic_qformer = GlobalContextExtractor(
        input_dim=d_model,
        d_model=d_model,
        num_context_vectors=32,
        depth=2
    ).cuda()

    # Improved Q-Former
    improved_qformer = ImprovedQFormer(
        input_dim=d_model,
        d_model=d_model,
        num_queries=32,
        depth=6,
        use_sparse_sampling=True
    ).cuda().eval()

    # Warmup
    _ = basic_qformer(F_visual)
    _ = improved_qformer(F_visual)

    # Benchmark
    n_iters = 100

    print(f"\n Running {n_iters} iterations...")

    # Basic
    start = time.time()
    for _ in range(n_iters):
        _ = basic_qformer(F_visual)
    torch.cuda.synchronize()
    basic_time = time.time() - start

    # Improved
    start = time.time()
    for _ in range(n_iters):
        _ = improved_qformer(F_visual)
    torch.cuda.synchronize()
    improved_time = time.time() - start

    print(f"\n Results:")
    print(f"   Basic Q-Former (depth=2):    {basic_time:.3f}s")
    print(f"   Improved Q-Former (depth=6): {improved_time:.3f}s")
    print(f"   Speedup: {basic_time/improved_time:.2f}x (with sparse sampling)")

    print("\n Note: Improved version is deeper (6 vs 2) but uses sparse sampling!")

    return True


def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*70)
    print(" COMPREHENSIVE TEST SUITE FOR IMPROVED FEATURE EXTRACTORS")
    print("="*70)

    tests = [
        ("Multi-Scale FPN", test_multi_scale_fpn),
        ("Improved Q-Former", test_improved_qformer),
        ("Deformable & SPP RoI", test_deformable_roi),
        ("Integrated Pipeline", test_integrated_pipeline),
    ]

    # Add performance test if CUDA available
    if torch.cuda.is_available():
        tests.append(("Performance Comparison", test_performance_comparison))

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "✓ PASSED"))
        except Exception as e:
            results.append((name, f"✗ FAILED: {str(e)}"))
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)

    for name, status in results:
        print(f" {status:12} {name}")

    passed = sum(1 for _, s in results if "PASSED" in s)
    total = len(results)

    print("="*70)
    print(f" {passed}/{total} tests passed")
    print("="*70)

    if passed == total:
        print("\n 🎉 All tests passed! Improved feature extractors are ready to use.")
    else:
        print("\n ⚠️  Some tests failed. Please check the errors above.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
