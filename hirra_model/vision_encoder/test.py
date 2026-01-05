import torch
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(ROOT_DIR)

# Bây giờ chúng ta có thể import từ hirra_model
from hirra_model.vision_encoder.multimodal_encoder import MultimodalEncoder
# ------------------------------


def test_vision_encoder():
    """
    Mục đích: Test toàn bộ module vision_encoder.
    Các bước:
    1. Khởi tạo MultimodalEncoder.
    2. (Tùy chọn) Thử tải trọng số pretrain.
    3. Tạo dữ liệu giả (fake data) CT và PET 3D.
    4. Đẩy dữ liệu qua mô hình.
    5. Kiểm tra shape của đặc trưng đầu ra (F_visual).
    """
    print("--- [BẮT ĐẦU] Test Module Vision Encoder ---")

    # 1. Định nghĩa cấu hình cho CTViT
    # Config này được phân tích từ pretrained checkpoint:
    # /media/gpus/Data/HF_HOME_DICE/pretrained_models/noise_ctvit.79000.pt
    # Sử dụng script: check_pretrained_config.py
    #
    # LƯU Ý QUAN TRỌNG:
    # - Config phải KHỚP CHÍNH XÁC với checkpoint để load được pretrained weights
    # - patch_size=20, temporal_patch_size=10 (khác với train_ctvit.py mặc định)
    # - image_size=480 (vì 480 chia hết cho patch_size=20)
    # ctvit_config = dict(
    #     dim=512,
    #     codebook_size=8192,
    #     image_size=160,  # Thay đổi từ 128 → 480 để match với checkpoint
    #     patch_size=20,   # Thay đổi từ 16 → 20 để match với checkpoint
    #     temporal_patch_size=10,  # Thay đổi từ 2 → 10 để match với checkpoint
    #     spatial_depth=4,
    #     temporal_depth=4,
    #     dim_head=32,
    #     heads=8,
    #     channels=1  # Grayscale CT/PET
    # )
    
    ctvit_config = dict(
            dim = 512,
            codebook_size = 8192,
            image_size = 480,
            patch_size = 20,
            temporal_patch_size = 10,
            spatial_depth = 4,
            temporal_depth = 4,
            dim_head = 32,
            heads = 8
        )

    # 2. Khởi tạo MultimodalEncoder và chuyển lên GPU
    device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng device: {device}")

    try:
        model = MultimodalEncoder(ctvit_config, num_fusion_layers=4)
        model = model.to(device)
        print(f"[THÀNH CÔNG] Khởi tạo MultimodalEncoder (2x CTViT + 4x Fusion) và chuyển lên {device}.")
    except Exception as e:
        print(f"[THẤT BẠI] Không thể khởi tạo MultimodalEncoder. Lỗi: {e}")
        return

    # 3. (Tùy chọn) Thử tải trọng số pretrain
    # File này có thể không tồn tại, nhưng chúng ta test xem hàm có chạy không.
    print("Đang thử gọi hàm load_pretrained_ctvit...")
    model.load_pretrained_ctvit("/media/gpus/Data/HF_HOME_DICE/pretrained_models/noise_ctvit.79000.pt")
    # (Việc in ra lỗi "không thể tải" ở đây là BÌNH THƯỜNG)

    # 4. Định nghĩa kích thước dữ liệu giả (đã được resize)
    # Dựa trên config và công thức:
    # - Total frames = 1 (first frame) + N * temporal_patch_size
    # - Với temporal_patch_size=10, để có ~20 temporal patches: 1 + 20*10 = 201
    BATCH_SIZE = 2
    CHANNELS = 1
    FRAMES = 201  # (1 frame đầu + 20 * 10 frame patch)
    HEIGHT = 480  # Thay đổi từ 128 → 480 để match với image_size
    WIDTH = 480   # Thay đổi từ 128 → 480 để match với image_size

    # 5. Tạo dữ liệu giả và chuyển lên GPU
    # Lưu ý: Dữ liệu thật của bạn cần resize về shape này trong Dataloader:
    # - CT:  (313, 512, 512) → (201, 480, 480)
    # - PET: (313, 256, 256) → (201, 480, 480)
    fake_ct_image = torch.randn(BATCH_SIZE, CHANNELS, FRAMES, HEIGHT, WIDTH).to(device)
    fake_pet_image = torch.randn(BATCH_SIZE, CHANNELS, FRAMES, HEIGHT, WIDTH).to(device)

    print(f"Đã tạo dữ liệu giả với shape: {fake_ct_image.shape} trên {device}")

    # 6. Đẩy dữ liệu qua mô hình (Forward pass)
    try:
        # Bật no_grad() để tiết kiệm bộ nhớ khi test (không cần gradient)
        # Nếu cần huấn luyện thì bỏ torch.no_grad()
        with torch.no_grad():
            F_visual = model(fake_ct_image, fake_pet_image)
        print(f"[THÀNH CÔNG] Chạy forward pass.")
    except Exception as e:
        print(f"[THẤT BẠI] Lỗi khi chạy forward pass. Lỗi: {e}")
        import traceback
        traceback.print_exc()
        raise e

    # 7. Kiểm tra (Assert) shape của F_visual
    # D (dim): 512
    # T (time_patches): 1 + (200 / 10) = 21  # Thay đổi: temporal_patch_size=10
    # H' (height_patches): 480 / 20 = 24
    # W' (width_patches): 480 / 20 = 24
    # Shape mong đợi: [B, D, T, H', W']
    expected_shape = (BATCH_SIZE, 512, 21, 24, 24)  # Thay đổi T: 101 → 21

    print(f"      Shape đầu vào: {fake_ct_image.shape}")
    print(f"      Shape đầu ra F_visual: {F_visual.shape}")
    print(f"      Shape mong đợi: {expected_shape}")
    #print(f"      Đầu ra: {F_visual}")

    assert F_visual.shape == expected_shape, \
        f"Test Thất Bại! Shape đầu ra là {F_visual.shape}, \
          nhưng mong đợi là {expected_shape}"

    print("[THÀNH CÔNG] Shape đầu ra F_visual chính xác.")
    print("--- [KẾT THÚC] Test Module Vision Encoder ---")


if __name__ == "__main__":
    # Để chạy file test này:
    # 1. Mở terminal
    # 2. Đứng từ thư mục gốc của dự án (hirra_project/)
    # 3. Gõ lệnh: python hirra_model/vision_encoder/test.py
    #
    # (Cách `sys.path.append` ở trên sẽ đảm bảo nó tìm thấy các module)
    
    test_vision_encoder()
