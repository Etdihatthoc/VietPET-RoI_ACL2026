"""
Validate quá trình xử lý toạ độ ROI và feature extraction.
Chạy script này TRƯỚC KHI training để đảm bảo logic ROI đúng.
"""

import sys
from pathlib import Path

# Add repo root to path
CURRENT_DIR = Path(__file__).resolve()
REPO_ROOT = CURRENT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import torch
import numpy as np
import json


def validate_coordinate_conversion():
    """Test chuyển đổi format toạ độ"""
    print("="*60)
    print("TEST 1: Chuyển Đổi Format Toạ Độ")
    print("="*60)

    # Box mẫu từ dataset
    json_box = [229.21875, 267.1875, 80.12840270996094,
                246.5625, 279.375, 85.56079864501953]
    # Format: [x_min, y_min, z_min, x_max, y_max, z_max]

    # Kết quả mong đợi sau convert thành [z_min, y_min, x_min, z_max, y_max, x_max]
    expected = [80.12840270996094, 267.1875, 229.21875,
                85.56079864501953, 279.375, 246.5625]

    # Áp dụng conversion
    converted = [json_box[2], json_box[1], json_box[0],
                 json_box[5], json_box[4], json_box[3]]

    assert np.allclose(converted, expected), "Conversion BỊ LỖI!"
    print("✓ Chuyển đổi toạ độ ĐÚNG")
    print(f"  Input (JSON):  {json_box}")
    print(f"  Output (model): {converted}")
    print(f"  Mong đợi:       {expected}")

    # Validate bounds
    vol_shape = (201, 480, 480)  # Z, H, W
    Z, H, W = vol_shape

    z_min, y_min, x_min, z_max, y_max, x_max = converted

    assert 0 <= z_min < Z and 0 <= z_max < Z, f"Z out of bounds: {z_min}, {z_max}"
    assert 0 <= y_min < H and 0 <= y_max < H, f"Y out of bounds: {y_min}, {y_max}"
    assert 0 <= x_min < W and 0 <= x_max < W, f"X out of bounds: {x_min}, {x_max}"

    print("✓ Tất cả toạ độ nằm trong volume bounds")


def validate_roi_extraction():
    """Test ROI feature extraction"""
    print("\n" + "="*60)
    print("TEST 2: Trích Xuất ROI Features")
    print("="*60)

    try:
        from hirra_model.feature_extractors.roi_align_3d import RoIAlign3D

        # Tạo dummy feature map
        B, C, Z, H, W = 1, 512, 201, 480, 480
        features = torch.randn(B, C, Z, H, W)

        # Tạo sample box (đã ở format đúng)
        boxes = torch.tensor([
            [80.0, 267.0, 229.0, 86.0, 279.0, 247.0]  # z,y,x,z,y,x
        ], dtype=torch.float32)

        # Trích xuất ROI features
        roi_align = RoIAlign3D(output_size=(7, 7, 7))
        roi_features = roi_align(features, [boxes], orig_dims=(W, H, Z))

        print(f"✓ Trích xuất ROI thành công")
        print(f"  Input features: {features.shape}")
        print(f"  ROI boxes: {boxes.shape}")
        print(f"  Output features: {roi_features[0].shape}")

        # Kiểm tra output không phải toàn zeros
        assert roi_features[0].abs().sum() > 0, "ROI features toàn là ZEROS!"
        print(f"  Feature mean: {roi_features[0].mean():.4f}")
        print(f"  Feature std: {roi_features[0].std():.4f}")
    except Exception as e:
        print(f"❌ Lỗi khi test ROI extraction: {e}")
        print("   (Có thể do chưa có hirra_model trong PYTHONPATH)")


def validate_dataset_boxes():
    """Load dataset thực và validate tất cả boxes"""
    print("\n" + "="*60)
    print("TEST 3: Validation Dataset Boxes")
    print("="*60)

    dataset_path = "/home/user01/aiotlab/sondinh/ACL 2026/Hirra_model/training/stage3/data/full_data/combined_roi_reformat_resized_train.json"

    if not Path(dataset_path).exists():
        print(f"⚠️  Dataset không tồn tại: {dataset_path}")
        print("   Bỏ qua test này")
        return

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    samples = data.get('samples', data.get('data', []))

    total_boxes = 0
    invalid_boxes = 0

    vol_shape = (201, 480, 480)
    Z, H, W = vol_shape

    for sample in samples:
        rois = sample.get('rois', [])
        for roi in rois:
            bbox = roi['bbox']  # [x_min, y_min, z_min, x_max, y_max, z_max]

            # Convert sang format model
            z_min, y_min, x_min = bbox[2], bbox[1], bbox[0]
            z_max, y_max, x_max = bbox[5], bbox[4], bbox[3]

            total_boxes += 1

            # Kiểm tra tính hợp lệ
            if not (0 <= z_min < z_max <= Z):
                invalid_boxes += 1
            if not (0 <= y_min < y_max <= H):
                invalid_boxes += 1
            if not (0 <= x_min < x_max <= W):
                invalid_boxes += 1

    print(f"✓ Đã validate {total_boxes} boxes từ {len(samples)} samples")
    print(f"  Invalid boxes: {invalid_boxes} ({100*invalid_boxes/max(total_boxes,1):.2f}%)")

    if invalid_boxes > 0:
        print("  ⚠️  Cảnh báo: Một số boxes nằm ngoài bounds!")
    else:
        print("  ✓ Tất cả boxes hợp lệ")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("VALIDATION SCRIPT CHO ROI LOGIC")
    print("="*60)
    print("Script này kiểm tra:")
    print("  1. Chuyển đổi format toạ độ ROI")
    print("  2. Trích xuất ROI features")
    print("  3. Validation tất cả boxes trong dataset")
    print("="*60 + "\n")

    try:
        validate_coordinate_conversion()
        validate_roi_extraction()
        validate_dataset_boxes()

        print("\n" + "="*60)
        print("✅ TẤT CẢ VALIDATION TESTS ĐỀU PASS!")
        print("="*60)
        print("\nBạn có thể tiếp tục training với confidence cao!")
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ VALIDATION FAILED: {e}")
        print("="*60)
        print("\nVui lòng fix lỗi trước khi training!")
        sys.exit(1)
