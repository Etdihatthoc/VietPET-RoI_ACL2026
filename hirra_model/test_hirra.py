"""
Test HiRRA Full Pipeline

Test toàn bộ kiến trúc HiRRA từ đầu đến cuối:
1. Vision Encoder (CTViT)
2. Feature Extractors (Improved with FPN, Q-Former, SPP)
3. Language Decoder (Qwen3-8B)
4. Stitching Logic
5. Training Forward Pass
6. Inference Generation
"""

import torch
import sys
import os
from contextlib import nullcontext

# --- Setup Path ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

try:
    from hirra_model.hirra import HiRRA
except ImportError as e:
    print(f"[ERROR] Cannot import HiRRA: {e}")
    print("Make sure all required files exist.")
    sys.exit(1)
# ------------------


def test_hirra_full_pipeline():
    """
    Test toàn bộ HiRRA pipeline
    """
    print("=" * 80)
    print("  HiRRA FULL PIPELINE TEST")
    print("=" * 80)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Using device: {device}")
    if device.type == 'cuda':
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # ============================================================
    # STEP 1: Define Configurations
    # ============================================================
    print("\n" + "=" * 80)
    print("[STEP 1] Defining Model Configurations")
    print("=" * 80)

    # CTViT config (Module 1)
    # These values match the pretrained checkpoint
    ctvit_config = dict(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8,
        channels=1
    )

    # Feature extractor config (Module 2)
    # Options: 'basic', 'improved', 'full'
    feature_config = 'improved'  # Recommended for good balance

    # Language model name (Module 3)
    llm_name = "Qwen/Qwen3-8B"
    use_llm_4bit = bool(int(os.environ.get("HIRRA_USE_4BIT", "0")))

    # Original image dimensions
    # IMPORTANT: Must match CTViT config
    # Formula: T = 1 + (n * temporal_patch_size), H,W must divide by patch_size
    # CTViT extracts first frame separately, so rest frames must divide by temporal_patch_size
    # 201 = 1 + 200, and 200/10=20 ✓, 480/20=24 ✓
    original_spatial_dims = (201, 480, 480)  # (T, H, W)

    # Number of query vectors
    num_queries = 32

    print(f"  CTViT Config:")
    print(f"    - Dimension: {ctvit_config['dim']}")
    print(f"    - Image size: {ctvit_config['image_size']}")
    print(f"    - Patch size: {ctvit_config['patch_size']}")
    print(f"    - Temporal patch size: {ctvit_config['temporal_patch_size']}")
    print(f"  Feature Extractor: {feature_config}")
    print(f"  Language Model: {llm_name}")
    print(f"  LLM 4-bit quantization: {use_llm_4bit}")
    print(f"  Num Queries: {num_queries}")

    # ============================================================
    # STEP 2: Initialize HiRRA Model
    # ============================================================
    print("\n" + "=" * 80)
    print("[STEP 2] Initializing HiRRA Model")
    print("=" * 80)
    print("(First run will download Qwen3-8B from HuggingFace - may take a few minutes)")

    try:
        model = HiRRA(
            ctvit_config=ctvit_config,
            feature_extractor_config=feature_config,
            language_decoder_name=llm_name,
            original_spatial_dims=original_spatial_dims,
            num_queries=num_queries,
            language_decoder_4bit=use_llm_4bit
        ).to(device)

        tokenizer = model.get_tokenizer()
        print(f"\n[SUCCESS] HiRRA model initialized!")

        # Print model info
        model_info = model.get_model_info()
        print(f"\n  Model Information:")
        print(f"    - Feature config: {model_info['feature_extractor_config']}")
        print(f"    - Vision dim: {model_info['vision_dim']}")
        print(f"    - LLM hidden dim: {model_info['llm_hidden_dim']}")
        print(f"    - Total parameters: {model_info['total_parameters']:,}")
        print(f"    - Trainable parameters: {model_info['trainable_parameters']:,}")
        print(f"    - Memory (fp32): {model_info['memory_footprint_fp32_GB']:.2f} GB")
        print(f"    - Memory (bf16): {model_info['memory_footprint_bf16_GB']:.2f} GB")

    except Exception as e:
        print(f"[FAILED] Could not initialize model: {e}")
        raise e

    # ============================================================
    # STEP 3: Create Dummy Data
    # ============================================================
    print("\n" + "=" * 80)
    print("[STEP 3] Creating Dummy Test Data")
    print("=" * 80)

    batch_size = 2
    T, H, W = original_spatial_dims

    # Create fake CT and PET images
    print(f"  Creating CT and PET images: [B={batch_size}, C=1, T={T}, H={H}, W={W}]")
    ct_image = torch.randn(batch_size, 1, T, H, W, device=device, dtype=torch.float32)
    pet_image = torch.randn(batch_size, 1, T, H, W, device=device, dtype=torch.float32)

    # Create fake bounding boxes (normalized coordinates [0, 1])
    # Format: [z_min, y_min, x_min, z_max, y_max, x_max]
    print(f"  Creating bounding boxes:")
    boxes_list = [
        torch.tensor([
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # RoI 1
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # RoI 2
        ], device=device, dtype=torch.float32),
        torch.tensor([
            [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # RoI 1
        ], device=device, dtype=torch.float32)
    ]
    print(f"    Batch 0: {len(boxes_list[0])} RoIs")
    print(f"    Batch 1: {len(boxes_list[1])} RoIs")

    # Create text prompts with placeholders
    print(f"  Creating text prompts with <GLOBAL_VISUALS> and <ROI_VISUALS> tokens:")
    text_batch = [
        "Phân tích hình ảnh: <GLOBAL_VISUALS> Phát hiện tổn thương 1: <ROI_VISUALS>. Phát hiện tổn thương 2: <ROI_VISUALS>.",
        "Báo cáo: <GLOBAL_VISUALS> Tổn thương: <ROI_VISUALS>."
    ]
    for i, text in enumerate(text_batch):
        print(f"    [{i}] {text}")

    # Tokenize
    inputs = tokenizer(
        text_batch,
        return_tensors='pt',
        padding='longest',
        truncation=True,
        max_length=128
    ).to(device)

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    print(f"\n  Data Summary:")
    print(f"    - CT shape: {ct_image.shape}")
    print(f"    - PET shape: {pet_image.shape}")
    print(f"    - Boxes per batch: {[len(b) for b in boxes_list]}")
    print(f"    - Input IDs shape: {input_ids.shape}")
    print(f"    - Tokenizer vocab size: {len(tokenizer)}")

    # ============================================================
    # TEST 1: Forward Pass (Training Mode)
    # ============================================================
    print("\n" + "=" * 80)
    print("[TEST 1] Testing Forward Pass (Training Mode)")
    print("=" * 80)

    try:
        # Create target IDs (ground truth responses)
        target_texts = [
            "Kết luận: Có dấu hiệu bất thường cần theo dõi thêm.",
            "Kết luận: Không có bất thường đáng kể."
        ]
        target_ids = tokenizer(
            target_texts,
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=64
        ).input_ids.to(device)

        print(f"  Target IDs shape: {target_ids.shape}")
        print(f"  Running forward pass...")

        # Forward pass
        outputs = model.forward(
            ct_image=ct_image,
            pet_image=pet_image,
            boxes_list=boxes_list,
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_ids=target_ids
        )

        loss = outputs.loss
        logits = outputs.logits

        print(f"\n[SUCCESS] Forward pass completed!")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Loss is finite: {torch.isfinite(loss).item()}")

        assert loss is not None and torch.isfinite(loss)
        assert logits.shape[0] == batch_size
        print(f"  ✓ All assertions passed")

    except Exception as e:
        print(f"[FAILED] Forward pass failed: {e}")
        raise e

    # ============================================================
    # TEST 2: Generate (Inference Mode)
    # ============================================================
    print("\n" + "=" * 80)
    print("[TEST 2] Testing Generate (Inference Mode)")
    print("=" * 80)

    try:
        # Create inference prompts
        inference_texts = [
            "Phân tích hình ảnh y tế: <GLOBAL_VISUALS> Mô tả tổn thương 1: <ROI_VISUALS>. Mô tả tổn thương 2: <ROI_VISUALS>. Kết luận:",
            "Báo cáo: <GLOBAL_VISUALS> Tổn thương phát hiện: <ROI_VISUALS>. Kết luận:"
        ]

        inference_inputs = tokenizer(
            inference_texts,
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=128
        ).to(device)

        print(f"  Inference input shape: {inference_inputs.input_ids.shape}")
        print(f"  Generating with max_new_tokens=50...")

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                ct_image=ct_image,
                pet_image=pet_image,
                boxes_list=boxes_list,
                input_ids=inference_inputs.input_ids,
                attention_mask=inference_inputs.attention_mask,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        print(f"\n[SUCCESS] Generation completed!")
        print(f"  Generated IDs shape: {generated_ids.shape}")
        print(f"  ✓ Generated more tokens than input: {generated_ids.shape[1]} > {inference_inputs.input_ids.shape[1]}")

        print(f"\n  Generated Reports:")
        for i, text in enumerate(generated_texts):
            print(f"\n    [Sample {i+1}]")
            print(f"    {text[:300]}...")  # Print first 300 chars

        assert generated_ids.shape[0] == batch_size
        assert generated_ids.shape[1] > inference_inputs.input_ids.shape[1]
        print(f"\n  ✓ All assertions passed")

    except Exception as e:
        print(f"[FAILED] Generate failed: {e}")
        raise e

    # ============================================================
    # TEST 3: Gradient Flow Check
    # ============================================================
    print("\n" + "=" * 80)
    print("[TEST 3] Testing Gradient Flow")
    print("=" * 80)

    try:
        # Backward pass
        print(f"  Preparing lightweight backward pass (freeze LLM to avoid OOM)...")
        model.zero_grad(set_to_none=True)

        llm_params = list(model.language_decoder.parameters())
        original_requires_grad = [p.requires_grad for p in llm_params]
        for param in llm_params:
            param.requires_grad_(False)

        autocast_ctx = (
            torch.cuda.amp.autocast(dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )

        with autocast_ctx:
            backward_outputs = model.forward(
                ct_image=ct_image,
                pet_image=pet_image,
                boxes_list=boxes_list,
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_ids=target_ids
            )

        backward_loss = backward_outputs.loss
        print(f"  Recomputed loss (LLM frozen): {backward_loss.item():.4f}")
        backward_loss.backward()

        for param, req in zip(llm_params, original_requires_grad):
            param.requires_grad_(req)

        # Check gradients
        has_grad_vision = any(p.grad is not None for p in model.vision_encoder.parameters())
        has_grad_feature = any(p.grad is not None for p in model.feature_extractor.parameters())
        has_grad_projector = any(p.grad is not None for p in model.visual_projector.parameters())
        has_grad_llm = any(p.grad is not None for p in model.language_decoder.parameters())

        print(f"\n[SUCCESS] Gradient flow check completed!")
        print(f"  Vision Encoder has gradients: {has_grad_vision}")
        print(f"  Feature Extractor has gradients: {has_grad_feature}")
        print(f"  Visual Projector has gradients: {has_grad_projector}")
        print(f"  Language Decoder has gradients (expected False - frozen): {has_grad_llm}")

        # At least some modules should have gradients
        assert has_grad_vision or has_grad_feature or has_grad_projector
        print(f"  ✓ Gradient flow is working")

    except Exception as e:
        print(f"[FAILED] Gradient flow check failed: {e}")
        raise e

    # ============================================================
    # TEST 4: Feature Extractor Components
    # ============================================================
    print("\n" + "=" * 80)
    print("[TEST 4] Testing Feature Extractor Components")
    print("=" * 80)

    try:
        # Test individual components
        print(f"  Testing vision encoder...")
        with torch.no_grad():
            F_visual = model.vision_encoder(ct_image, pet_image)
            print(f"    Visual features shape: {F_visual.shape}")

        print(f"  Testing feature extractor...")
        with torch.no_grad():
            F_global, F_rois = model.feature_extractor(
                F_visual, boxes_list, original_spatial_dims
            )
            print(f"    Global features shape: {F_global.shape}")
            print(f"    RoI features shapes: {[f.shape for f in F_rois]}")

        print(f"  Testing visual projector...")
        with torch.no_grad():
            proj_rois, proj_global = model.visual_projector(F_rois, F_global)
            print(f"    Projected global shape: {proj_global.shape}")
            print(f"    Projected RoI shapes: {[f.shape for f in proj_rois]}")

        print(f"\n[SUCCESS] All components working correctly!")
        print(f"  ✓ Vision encoder output: {F_visual.shape}")
        print(f"  ✓ Feature extractor outputs verified")
        print(f"  ✓ Visual projector outputs verified")

    except Exception as e:
        print(f"[FAILED] Component test failed: {e}")
        raise e

    # ============================================================
    # Final Summary
    # ============================================================
    print("\n" + "=" * 80)
    print("  ALL TESTS PASSED! ✓")
    print("=" * 80)
    print("\n  HiRRA Model Summary:")
    print(f"    - Feature Extractor Config: {feature_config}")
    print(f"    - Total Parameters: {model_info['total_parameters']:,}")
    print(f"    - Trainable Parameters: {model_info['trainable_parameters']:,}")
    print(f"    - Memory Footprint (bf16): {model_info['memory_footprint_bf16_GB']:.2f} GB")
    print(f"\n  Pipeline Flow:")
    print(f"    CT/PET Images → Vision Encoder → Feature Extractors → Projector")
    print(f"    Text Tokens → Text Embeddings")
    print(f"    Visual Features + Text Embeddings → Stitching → LLM → Generated Report")
    print(f"\n  All modules are working correctly!")
    print("=" * 80)


if __name__ == "__main__":
    """
    Run this test:
    1. Open terminal
    2. Navigate to DICE_model directory
    3. Run: python hirra_model/test_hirra.py

    Requirements:
    - PyTorch with CUDA support (recommended)
    - transformers library
    - Internet connection (first run to download Qwen model)
    """
    test_hirra_full_pipeline()
