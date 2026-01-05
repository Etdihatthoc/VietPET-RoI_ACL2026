#!/usr/bin/env python
"""
Script để phân tích pretrained checkpoint và tìm ra config đúng
"""
import torch
import math

def analyze_checkpoint(checkpoint_path):
    """
    Phân tích checkpoint và reverse engineer config từ shape của weights
    """
    print(f"=== Đang phân tích checkpoint: {checkpoint_path} ===\n")

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("✓ Load checkpoint thành công\n")
    except Exception as e:
        print(f"✗ Lỗi khi load checkpoint: {e}")
        return None

    # Lấy state_dict
    if isinstance(checkpoint, dict):
        print(f"Checkpoint keys: {list(checkpoint.keys())}\n")
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    print("=== PHÂN TÍCH SHAPE CỦA CÁC LAYERS QUAN TRỌNG ===\n")

    config = {}

    # 1. Phân tích patch_size từ to_patch_emb_first_frame
    key1 = 'to_patch_emb_first_frame.1.weight'
    if key1 in state_dict:
        shape = state_dict[key1].shape
        print(f"1. {key1}")
        print(f"   Shape: {shape}")

        # LayerNorm weight shape = [channels * patch_h * patch_w]
        total = shape[0]
        print(f"   Total features: {total}")

        # Giả sử channels = 1 (grayscale CT/PET)
        # Tính patch_size: sqrt(total / channels)
        patch_area = total  # vì channels = 1
        patch_size = int(math.sqrt(patch_area))

        if patch_size * patch_size == patch_area:
            print(f"   ✓ Detected: channels=1, patch_size={patch_size}")
            config['channels'] = 1
            config['patch_size'] = patch_size
        else:
            print(f"   ✗ Không thể xác định patch_size chính xác. patch_area={patch_area}")

    print()

    # 2. Phân tích temporal_patch_size từ to_patch_emb
    key2 = 'to_patch_emb.1.weight'
    if key2 in state_dict:
        shape = state_dict[key2].shape
        print(f"2. {key2}")
        print(f"   Shape: {shape}")

        # LayerNorm weight shape = [channels * temporal_patch_size * patch_h * patch_w]
        total = shape[0]
        print(f"   Total features: {total}")

        if 'patch_size' in config:
            # total = channels * temporal_patch_size * patch_size^2
            # temporal_patch_size = total / (channels * patch_size^2)
            channels = config['channels']
            patch_size = config['patch_size']
            temporal_patch_size = total // (channels * patch_size * patch_size)

            if total == channels * temporal_patch_size * patch_size * patch_size:
                print(f"   ✓ Detected: temporal_patch_size={temporal_patch_size}")
                config['temporal_patch_size'] = temporal_patch_size
            else:
                print(f"   ✗ Không thể xác định temporal_patch_size chính xác")

    print()

    # 3. Phân tích dim từ enc_spatial_transformer hoặc to_q
    key3 = 'enc_spatial_transformer.layers.0.1.to_q.weight'
    if key3 in state_dict:
        shape = state_dict[key3].shape
        print(f"3. {key3}")
        print(f"   Shape: {shape}")

        # to_q.weight shape = [inner_dim, dim]
        # inner_dim = dim_head * heads
        inner_dim, dim = shape
        print(f"   ✓ Detected: dim={dim}")
        config['dim'] = dim

    print()

    # 4. Phân tích codebook_size từ vq
    key4 = 'vq.codebook.weight'
    if key4 in state_dict:
        shape = state_dict[key4].shape
        print(f"4. {key4}")
        print(f"   Shape: {shape}")

        # codebook shape = [codebook_size, dim]
        codebook_size, dim_vq = shape
        print(f"   ✓ Detected: codebook_size={codebook_size}")
        config['codebook_size'] = codebook_size

    print()

    # 5. Phân tích spatial_depth (số layers trong spatial transformer)
    spatial_depth = 0
    for key in state_dict.keys():
        if 'enc_spatial_transformer.layers.' in key:
            layer_idx = int(key.split('.layers.')[1].split('.')[0])
            spatial_depth = max(spatial_depth, layer_idx + 1)

    if spatial_depth > 0:
        print(f"5. Spatial Transformer Depth")
        print(f"   ✓ Detected: spatial_depth={spatial_depth}")
        config['spatial_depth'] = spatial_depth

    print()

    # 6. Phân tích temporal_depth
    temporal_depth = 0
    for key in state_dict.keys():
        if 'enc_temporal_transformer.layers.' in key:
            layer_idx = int(key.split('.layers.')[1].split('.')[0])
            temporal_depth = max(temporal_depth, layer_idx + 1)

    if temporal_depth > 0:
        print(f"6. Temporal Transformer Depth")
        print(f"   ✓ Detected: temporal_depth={temporal_depth}")
        config['temporal_depth'] = temporal_depth

    print()

    # 7. Kiểm tra image_size (không có trong checkpoint, cần đoán)
    # Với patch_size=20, các image_size phổ biến: 160, 180, 200, 240, 256
    if 'patch_size' in config:
        ps = config['patch_size']
        print(f"7. Image Size (cần xác định)")
        print(f"   Patch size = {ps}")
        print(f"   Possible image_size values (phải chia hết cho {ps}):")
        possible_sizes = [size for size in [128, 160, 180, 200, 224, 240, 256] if size % ps == 0]
        for size in possible_sizes:
            patches = size // ps
            print(f"     - image_size={size} → {patches}×{patches} patches")

        # Giá trị mặc định thường gặp
        if ps == 20:
            config['image_size'] = 160  # hoặc 200
            print(f"   ⚠ Đề xuất: image_size=160 hoặc 200 (phổ biến với patch_size=20)")
        elif ps == 16:
            config['image_size'] = 128
            print(f"   ✓ Đề xuất: image_size=128 (phổ biến với patch_size=16)")

    print("\n" + "="*60)
    print("=== CONFIG ĐỀ XUẤT ===")
    print("="*60 + "\n")

    print("ctvit_config = dict(")
    for key, value in config.items():
        print(f"    {key}={value},")

    # Thêm các tham số còn thiếu với giá trị mặc định
    if 'dim_head' not in config:
        print(f"    dim_head=32,  # Giá trị mặc định")
    if 'heads' not in config:
        print(f"    heads=8,  # Giá trị mặc định")

    print(")")

    print("\n" + "="*60)

    return config


if __name__ == "__main__":
    checkpoint_path = "/media/gpus/Data/HF_HOME_DICE/pretrained_models/noise_ctvit.79000.pt"

    print("\n" + "="*60)
    print("SCRIPT PHÂN TÍCH PRETRAINED CHECKPOINT")
    print("="*60 + "\n")

    config = analyze_checkpoint(checkpoint_path)

    if config:
        print("\n✓ Phân tích hoàn tất!")
        print("\nHãy copy config trên vào test.py để sử dụng pretrained weights.")
    else:
        print("\n✗ Không thể phân tích checkpoint.")
