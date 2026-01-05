import torch
import sys
import os
from typing import List

# --- Thiết lập Path để Import ---
# Thêm thư mục gốc của dự án (hirra_project) vào Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Đi lùi 2 cấp (từ language_decoder -> hirra_model -> hirra_project)
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(ROOT_DIR)

# Import các module cần test
try:
    from hirra_model.language_decoder.llm_loader import LanguageDecoder
    from Hirra_model.hirra_model.language_decoder.projection import VisualProjector
except ImportError:
    print("Lỗi: Không thể import. Hãy chắc chắn bạn đã tạo các file:")
    print(" - hirra_model/language_decoder/llm_loader.py")
    print(" - hirra_model/language_decoder/projection.py")
    print(" - và các file __init__.py cần thiết.")
    sys.exit(1)
# ------------------------------

def test_language_decoder_modules():
    """
    Mục đích: Test toàn bộ Module 3 (Language Decoder & Projector)
    """
    print("--- [BẮT ĐẦU] Test Module 3: Language Decoder ---")
    
    # Thiết lập device (ưu tiên GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng device: {device}")
    
    # Định nghĩa hằng số test
MODEL_NAME = "Qwen/Qwen3-8B"
    VISION_FEATURE_DIM = 512 # Từ Module 1 (CTViT)
    BATCH_SIZE = 2
    NUM_GLOBAL_VECTORS = 32 # Từ QFormer

    # --- Test 1: LanguageDecoder (Tải LLM và Tokenizer) ---
    print(f"\n[TEST 1] Đang tải LanguageDecoder (Model: {MODEL_NAME})...")
    print("(Lần chạy đầu tiên có thể mất vài phút để tải mô hình)")
    try:
        llm_decoder = LanguageDecoder(model_name=MODEL_NAME).to(device)
        tokenizer = llm_decoder.get_tokenizer()
        print(f"[THÀNH CÔNG] Đã tải LanguageDecoder và Tokenizer.")
    except Exception as e:
        print(f"[THẤT BẠI] Không thể tải mô hình từ Hugging Face. Lỗi: {e}")
        print("Vui lòng kiểm tra kết nối mạng và tên model.")
        return

    # Lấy các thông số quan trọng từ LLM đã tải
    LLM_HIDDEN_DIM = llm_decoder.llm_hidden_dim
    GLOBAL_TOKEN_ID = llm_decoder.global_token_id
    ROI_TOKEN_ID = llm_decoder.roi_token_id
    VOCAB_SIZE = llm_decoder.model.config.vocab_size

    print(f" - LLM Hidden Dim (d_model): {LLM_HIDDEN_DIM}")
    print(f" - ID Token <GLOBAL_VISUALS>: {GLOBAL_TOKEN_ID}")
    print(f" - ID Token <ROI_VISUALS>: {ROI_TOKEN_ID}")
    
    assert GLOBAL_TOKEN_ID is not None and ROI_TOKEN_ID is not None
    assert GLOBAL_TOKEN_ID != tokenizer.unk_token_id
    assert ROI_TOKEN_ID != tokenizer.unk_token_id
    print(f"[THÀNH CÔNG] Các token đặc biệt đã được thêm vào tokenizer.")

    # --- Test 2: VisualProjector (Chiếu Đặc Trưng) ---
    print(f"\n[TEST 2] Đang test VisualProjector...")
    try:
        projector = VisualProjector(
            vision_feature_dim=VISION_FEATURE_DIM,
            llm_hidden_dim=LLM_HIDDEN_DIM
        ).to(device).to(torch.bfloat16) # Chuyển projector sang bfloat16

        # Tạo dữ liệu hình ảnh giả
        fake_F_global_vis = torch.randn(
            BATCH_SIZE, NUM_GLOBAL_VECTORS, VISION_FEATURE_DIM, 
            device=device, dtype=torch.bfloat16
        )
        fake_F_rois_list = [
            torch.randn(3, VISION_FEATURE_DIM, device=device, dtype=torch.bfloat16), # Item 0 có 3 RoI
            torch.randn(0, VISION_FEATURE_DIM, device=device, dtype=torch.bfloat16)  # Item 1 có 0 RoI
        ]
        
        # Chạy forward projector
        proj_rois_list, proj_global = projector(fake_F_rois_list, fake_F_global_vis)
        
        # Kiểm tra shape đầu ra
        expected_global_shape = (BATCH_SIZE, NUM_GLOBAL_VECTORS, LLM_HIDDEN_DIM)
        expected_roi_shape_0 = (3, LLM_HIDDEN_DIM)
        expected_roi_shape_1 = (0, LLM_HIDDEN_DIM)
        
        assert proj_global.shape == expected_global_shape
        assert proj_rois_list[0].shape == expected_roi_shape_0
        assert proj_rois_list[1].shape == expected_roi_shape_1
        
        print(f"[THÀNH CÔNG] VisualProjector đã chiếu đặc trưng đúng shape.")
        print(f" - Shape Global: {proj_global.shape} (Mong đợi: {expected_global_shape})")
        print(f" - Shape RoI 0: {proj_rois_list[0].shape} (Mong đợi: {expected_roi_shape_0})")
        print(f" - Shape RoI 1: {proj_rois_list[1].shape} (Mong đợi: {expected_roi_shape_1}) (Test case rỗng)")

    except Exception as e:
        print(f"[THẤT BẠI] Test VisualProjector. Lỗi: {e}")
        raise e

    # --- Test 3: Logic "Khâu Vá" (Stitching) và Chạy LLM ---
    print(f"\n[TEST 3] Đang test logic 'khâu vá' (stitching) và LLM forward...")
    
    # Đây là logic phức tạp, mô phỏng những gì Module 4 (hirra.py) sẽ làm
    
    try:
        # 1. Tạo batch văn bản giả với các placeholder
        text_batch = [
            f"Báo cáo: <GLOBAL_VISUALS> Phát hiện 1: <ROI_VISUALS>. Hết.", # 1 Global, 1 RoI
            f"Báo cáo: <GLOBAL_VISUALS> Phát hiện 1: <ROI_VISUALS>. Phát hiện 2: <ROI_VISUALS>." # 1 Global, 2 RoI
        ]
        
        # 2. Tokenize văn bản
        inputs = tokenizer(
            text_batch, 
            return_tensors='pt', 
            padding='longest', 
            truncation=True, 
            max_length=64
        ).to(device)
        
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        # 3. Lấy embedding văn bản
        # Shape: [B, seq_len, D_LLM]
        text_embeds = llm_decoder.get_text_embeddings(input_ids)

        # 4. Tạo đặc trưng hình ảnh giả (đã được chiếu)
        # (Sử dụng lại từ Test 2, nhưng tạo lại RoI cho đúng số lượng)
        proj_rois_list_test3 = [
            torch.randn(1, LLM_HIDDEN_DIM, device=device, dtype=torch.bfloat16), # Item 0 có 1 RoI
            torch.randn(2, LLM_HIDDEN_DIM, device=device, dtype=torch.bfloat16)  # Item 1 có 2 RoI
        ]
        proj_global_test3 = torch.randn(
            BATCH_SIZE, NUM_GLOBAL_VECTORS, LLM_HIDDEN_DIM, 
            device=device, dtype=torch.bfloat16
        )

        # 5. Logic "Khâu Vá"
        final_embeds_batch = []
        final_mask_batch = []
        max_len = 0 # Để theo dõi độ dài lớn nhất cho padding

        for i in range(BATCH_SIZE):
            current_embeds_list = []
            current_mask_list = []
            roi_idx = 0
            
            # Lặp qua từng token ID trong chuỗi
            for token_idx in range(input_ids.shape[1]):
                
                # Dừng nếu là token padding
                if attention_mask[i, token_idx] == 0:
                    break
                    
                token_id = input_ids[i, token_idx].item()

                if token_id == GLOBAL_TOKEN_ID:
                    # Thay thế 1 token <GLOBAL_VISUALS>
                    # bằng N_VECTORS vector đặc trưng toàn cục
                    current_embeds_list.append(proj_global_test3[i])
                    current_mask_list.append(torch.ones(NUM_GLOBAL_VECTORS, device=device))
                
                elif token_id == ROI_TOKEN_ID:
                    # Thay thế 1 token <ROI_VISUALS>
                    # bằng 1 vector đặc trưng RoI
                    current_embeds_list.append(proj_rois_list_test3[i][roi_idx:roi_idx+1])
                    current_mask_list.append(torch.ones(1, device=device))
                    roi_idx += 1
                
                else:
                    # Giữ nguyên embedding văn bản
                    current_embeds_list.append(text_embeds[i, token_idx:token_idx+1])
                    current_mask_list.append(torch.ones(1, device=device))

            # Nối tất cả các embedding của 1 item lại
            final_embeds_batch.append(torch.cat(current_embeds_list, dim=0))
            final_mask_batch.append(torch.cat(current_mask_list, dim=0))
            
            # Cập nhật max_len
            if final_mask_batch[-1].shape[0] > max_len:
                max_len = final_mask_batch[-1].shape[0]

        # 6. Padding (sau khi "khâu vá")
        # Vì các chuỗi bây giờ có độ dài khác nhau
        final_embeds_padded = torch.zeros(BATCH_SIZE, max_len, LLM_HIDDEN_DIM, device=device, dtype=torch.bfloat16)
        final_mask_padded = torch.zeros(BATCH_SIZE, max_len, device=device, dtype=torch.long)

        for i in range(BATCH_SIZE):
            seq_len = final_embeds_batch[i].shape[0]
            final_embeds_padded[i, :seq_len] = final_embeds_batch[i]
            final_mask_padded[i, :seq_len] = final_mask_batch[i]

        print(f"[THÀNH CÔNG] Đã 'khâu vá' embedding.")
        print(f" - Shape text embeds gốc: {text_embeds.shape}")
        print(f" - Shape embeds 'khâu vá': {final_embeds_padded.shape} (chiều dài đã tăng lên {max_len})")

        # 7. Chạy LLM Forward với embedding đã "khâu vá"
        logits = llm_decoder.forward(
            inputs_embeds=final_embeds_padded,
            attention_mask=final_mask_padded
        )
        
        # 8. Kiểm tra shape logits
        # Shape phải là [B, max_len, vocab_size]
        # (vocab_size đã bao gồm 2 token đặc biệt)
        expected_logits_shape = (BATCH_SIZE, max_len, len(tokenizer))
        
        assert logits.shape == expected_logits_shape
        print(f"[THÀNH CÔNG] LLM forward chạy thành công.")
        print(f" - Shape logits: {logits.shape} (Mong đợi: {expected_logits_shape})")

    except Exception as e:
        print(f"[THẤT BẠI] Test 'khâu vá' / LLM forward. Lỗi: {e}")
        raise e

    print("\n--- [KẾT THÚC] Module 3 đã vượt qua tất cả bài test! ---")

if __name__ == "__main__":
    # Để chạy file test này:
    # 1. Mở terminal
    # 2. Đứng từ thư mục gốc của dự án (hirra_project/)
    # 3. Gõ lệnh: python hirra_model/language_decoder/test.py
    
    test_language_decoder_modules()
