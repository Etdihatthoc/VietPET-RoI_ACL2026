import torch
import sys
import os

# --- Thiết lập Path để Import ---
# Thêm thư mục gốc của dự án (hirra_project) vào Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Đi lùi 2 cấp (từ feature_extractors -> hirra_model -> hirra_project)
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(ROOT_DIR)

# Import các module cần test
from hirra_model.feature_extractors.roi_align_3d import RoIAlign3D
from hirra_model.feature_extractors.global_context import GlobalContextExtractor
# ------------------------------


def test_feature_extractors():
    """
    Mục đích: Test toàn bộ Module 2 (Feature Extractors)
    1. Test RoIAlign3D: Trích xuất đặc trưng vùng.
    2. Test GlobalContextExtractor (QFormer): Trích xuất đặc trưng toàn cục.
    """
    print("--- [BẮT ĐẦU] Test Module 2: Feature Extractors ---")

    # --- Dữ liệu giả ---
    BATCH_SIZE = 2
    D_MODEL = 512 # Chiều đặc trưng của VisionEncoder và LLM

    # 1. Tạo F_visual giả (đầu ra của Module 1)
    # Shape: [B, D, T_feat, H_feat, W_feat]
    # (T=101, H=8, W=8 là từ config của CTViT [cite: ibrahimethemhamamci/generatect/GenerateCT-2a811356de351c67f89b2929c8bc9f2390797d9c/train_ctvit.py])
    F_visual_shape = (BATCH_SIZE, D_MODEL, 101, 8, 8)
    F_visual = torch.randn(F_visual_shape)
    print(f"Đã tạo F_visual giả shape: {F_visual.shape}")

    # 2. Tạo Bounding Boxes giả (tọa độ pixel gốc)
    # Kích thước ảnh gốc mà box được vẽ trên đó
    ORIG_DIMS = (201, 128, 128) # (T, H, W)
    
    # boxes_list là một list (dài BATCH_SIZE)
    # Item 0 có 3 RoI
    boxes_item_0 = torch.tensor([
        [10, 20, 30, 40, 50, 60],  # [z_min, y_min, x_min, z_max, y_max, x_max]
        [100, 50, 50, 120, 80, 80],
        [150, 10, 70, 200, 60, 100],
    ], dtype=torch.float32)
    
    # Item 1 có 2 RoI
    boxes_item_1 = torch.tensor([
        [5, 15, 25, 35, 45, 55],
        [70, 70, 70, 80, 80, 80],
    ], dtype=torch.float32)
    
    boxes_list = [boxes_item_0, boxes_item_1]
    print(f"Đã tạo Bounding Boxes giả: item 0 có {boxes_item_0.shape[0]} RoIs, item 1 có {boxes_item_1.shape[0]} RoIs.")

    # --- Test 1: RoIAlign3D ---
    print("\n[TEST 1] Đang test RoIAlign3D...")
    try:
        # Khởi tạo RoIAlign3D
        # (output_size không quá quan trọng vì ta dùng AdaptiveAvgPool3d)
        roi_aligner = RoIAlign3D(output_size=(4, 4, 4), d_model=D_MODEL)
        
        # Chạy forward
        all_rois_features = roi_aligner(F_visual, boxes_list, ORIG_DIMS)
        
        print(f"[THÀNH CÔNG] RoIAlign3D chạy forward.")

        # Kiểm tra shape đầu ra
        assert isinstance(all_rois_features, list)
        assert len(all_rois_features) == BATCH_SIZE
        print(f" - Đầu ra là list có độ dài {len(all_rois_features)} (Đúng)")
        
        # Kiểm tra item 0
        expected_shape_0 = (boxes_item_0.shape[0], D_MODEL) # (3, 512)
        assert all_rois_features[0].shape == expected_shape_0
        print(f" - Shape RoIs item 0: {all_rois_features[0].shape} (Mong đợi: {expected_shape_0}) (Đúng)")

        # Kiểm tra item 1
        expected_shape_1 = (boxes_item_1.shape[0], D_MODEL) # (2, 512)
        assert all_rois_features[1].shape == expected_shape_1
        print(f" - Shape RoIs item 1: {all_rois_features[1].shape} (Mong đợi: {expected_shape_1}) (Đúng)")

        print("[THÀNH CÔNG] Test RoIAlign3D!")

    except Exception as e:
        print(f"[THẤT BẠI] Test RoIAlign3D. Lỗi: {e}")
        raise e

    # --- Test 2: GlobalContextExtractor (QFormer) ---
    print("\n[TEST 2] Đang test GlobalContextExtractor (QFormer)...")
    try:
        NUM_CONTEXT_VECTORS = 32
        QFORMER_DEPTH = 2
        
        # Khởi tạo QFormer
        qformer = GlobalContextExtractor(
            input_dim=D_MODEL,
            d_model=D_MODEL,
            num_context_vectors=NUM_CONTEXT_VECTORS,
            depth=QFORMER_DEPTH
        )
        
        # Chạy forward
        global_context_vectors = qformer(F_visual)
        
        print(f"[THÀNH CÔNG] QFormer chạy forward.")

        # Kiểm tra shape đầu ra
        expected_shape = (BATCH_SIZE, NUM_CONTEXT_VECTORS, D_MODEL)
        assert global_context_vectors.shape == expected_shape
        print(f" - Shape đầu ra: {global_context_vectors.shape} (Mong đợi: {expected_shape}) (Đúng)")
        
        print("[THÀNH CÔNG] Test GlobalContextExtractor (QFormer)!")

    except Exception as e:
        print(f"[THẤT BẠI] Test GlobalContextExtractor (QFormer). Lỗi: {e}")
        raise e

    print("\n--- [KẾT THÚC] Module 2 đã vượt qua tất cả bài test! ---")


if __name__ == "__main__":
    # Để chạy file test này:
    # 1. Mở terminal
    # 2. Đứng từ thư mục gốc của dự án (hirra_project/)
    # 3. Gõ lệnh: python hirra_model/feature_extractors/test.py
    
    test_feature_extractors()
