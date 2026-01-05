import torch
from torch import nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint

# Import class CTViT gốc từ file vit_3d.py
from .vit_3d import CTViT
# Import lớp Fusion từ file fusion.py
from .fusion import CrossAttentionLayer

class MultimodalEncoder(nn.Module):
    def __init__(self, ctvit_config: dict, num_fusion_layers: int = 4, use_gradient_checkpointing: bool = True):
        """
        Mục đích: Module chính của bộ mã hóa 3D.
                  Chứa 2 bộ encoder (CT, PET) và các lớp "trộn" đặc trưng.

        Đầu vào:
            - ctvit_config (dict): Một dict chứa các tham số để khởi tạo CTViT
              (ví dụ: dim=512, codebook_size=8192, image_size=128, v.v...)

            - num_fusion_layers (int): Số tầng cross-attention để "trộn" PET và CT.

            - use_gradient_checkpointing (bool): Sử dụng gradient checkpointing để giảm VRAM (default: True)
        """
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Khởi tạo 2 encoder. Chúng có cùng kiến trúc nhưng trọng số sẽ khác nhau.
        # Chúng ta tắt use_vgg_and_gan=False để nó không khởi tạo VGG hay Discriminator
        #
        self.ct_encoder = CTViT(**ctvit_config, use_vgg_and_gan=False, use_gradient_checkpointing=use_gradient_checkpointing)
        self.pet_encoder = CTViT(**ctvit_config, use_vgg_and_gan=False, use_gradient_checkpointing=use_gradient_checkpointing)

        # for param in self.ct_encoder.parameters():
        #      param.requires_grad = False
        # for param in self.ct_encoder.enc_spatial_transformer.parameters():
        #     param.requires_grad = False
        # for param in self.ct_encoder.enc_temporal_transformer.parameters():
        #     param.requires_grad = False
            
        # for param in self.pet_encoder.parameters():
        #     param.requires_grad = False

        # print(f"======Freeze CT and PET encoder weight======")
        
        # Lấy chiều đặc trưng (dimension) từ config
        dim = ctvit_config.get('dim', 512)

        # Số heads cho Cross-Attention (giảm từ 8 xuống 4 để tiết kiệm VRAM)
        # Giảm heads không ảnh hưởng đến pretrained CTViT weights
        fusion_heads = ctvit_config.get('fusion_heads', 4)

        # Khởi tạo các lớp Fusion
        # Chúng ta sẽ làm xen kẽ: CT "hỏi" PET, và PET "hỏi" CT
        self.fusion_layers = nn.ModuleList([])
        for _ in range(num_fusion_layers):
            self.fusion_layers.append(nn.ModuleList([
                CrossAttentionLayer(dim=dim, heads=fusion_heads), # CT "hỏi" PET
                CrossAttentionLayer(dim=dim, heads=fusion_heads)  # PET "hỏi" CT
            ]))

    def load_pretrained_ctvit(self, checkpoint_path: str):
        """
        Mục đích: Tải trọng số pretrained (ví dụ: ctvit_pretrained.pt)
                  vào self.ct_encoder.
                  
        Lưu ý: Chúng ta chỉ tải cho ct_encoder. pet_encoder sẽ được
               huấn luyện từ đầu (hoặc bạn có thể tải cùng trọng số
               nếu muốn thử nghiệm).
        """
        try:
            # Tải checkpoint
            pkg = torch.load(checkpoint_path, map_location='cpu')

            # Trích xuất state_dict (trọng số)
            # File `inference_transformer.py`
            # và `train_ctvit.py`
            # cho thấy trọng số được lưu trực tiếp (không có key 'model')
            if 'model' in pkg:
                state_dict = pkg['model']
            else:
                state_dict = pkg
            
            # Tải trọng số vào self.ct_encoder
            # strict=False vì chúng ta đã tắt VGG và Discriminator,
            # nên một số trọng số (ví dụ: discr.*) sẽ bị thiếu.
            # Điều này là BÌNH THƯỜNG và mong đợi.
            self.ct_encoder.load_state_dict(state_dict, strict=False)
            print(f"Đã tải thành công trọng số CT-ViT pretrained từ: {checkpoint_path}")

        except Exception as e:
            print(f"LỖI: không thể tải trọng số pretrained CT-ViT. Lỗi: {e}")
            print("Mô hình CT-Encoder sẽ được huấn luyện từ đầu.")

    def forward(self, ct_image: torch.Tensor, pet_image: torch.Tensor):
        """
        Mục đích: Hàm forward chính, nhận ảnh CT và PET, trả về bản đồ
                  đặc trưng 3D (F_visual) đã được "trộn".

        Lưu ý quan trọng về kích thước:
            Mô hình này (dựa trên CTViT) 
            mong đợi đầu vào có kích thước cố định
            (ví dụ: 201x128x128).
            
            Bạn PHẢI resize/resample ảnh CT (313x512x512) và PET (313x256x256)
            của mình về kích thước này TRƯỚC KHI gọi hàm này.

        Đầu vào:
            - ct_image (torch.Tensor): Ảnh 3D CT (đã resize).
              Shape: [B, 1, F, H, W] (ví dụ: [B, 1, 201, 128, 128])
            - pet_image (torch.Tensor): Ảnh 3D PET (đã resize).
              Shape: [B, 1, F, H, W] (ví dụ: [B, 1, 201, 128, 128])

        Đầu ra:
            - F_visual (torch.Tensor): Bản đồ đặc trưng 3D cuối cùng.
              Shape: [B, D, T, H', W'] (ví dụ: [B, 512, 101, 8, 8])
              (B: Batch, D: Chiều đặc trưng, T, H', W': Kích thước
               giảm của đặc trưng 3D)
        """
        
        # 1. Trích xuất đặc trưng thô (raw features) từ mỗi encoder
        # Chúng ta dùng hàm `encode_features` mới thêm vào CTViT (trong vit_3d.py)
        # Hàm này đã được bọc trong @torch.no_grad()
        ct_tokens = self.ct_encoder.encode_features(ct_image)
        pet_tokens = self.pet_encoder.encode_features(pet_image)
        
        # ct_tokens và pet_tokens đang có shape: [B, T, H', W', D]
        # (ví dụ: [B, 101, 8, 8, 512])

        # 2. "Phẳng hóa" (Flatten) đặc trưng để đưa vào Transformer
        # Chuyển shape [B, T, H', W', D] -> [B, N, D]
        # trong đó N = T * H' * W'
        b, t, h, w, d = ct_tokens.shape
        ct_tokens_flat = rearrange(ct_tokens, 'b t h w d -> b (t h w) d')
        pet_tokens_flat = rearrange(pet_tokens, 'b t h w d -> b (t h w) d')

        # 3. Chạy qua các lớp Fusion (Cross-Attention)
        # Sử dụng gradient checkpointing để tiết kiệm VRAM khi training
        for (ct_attn_pet, pet_attn_ct) in self.fusion_layers:

            if self.training and self.use_gradient_checkpointing:
                # Gradient checkpointing: trade compute for memory
                # Giảm ~2x VRAM cho fusion layers
                temp_ct = checkpoint(
                    ct_attn_pet,
                    ct_tokens_flat,
                    pet_tokens_flat,
                    use_reentrant=False
                )
                temp_pet = checkpoint(
                    pet_attn_ct,
                    pet_tokens_flat,
                    ct_tokens_flat,
                    use_reentrant=False
                )
            else:
                # CT "hỏi" PET
                temp_ct = ct_attn_pet(
                    query=ct_tokens_flat,
                    key_value=pet_tokens_flat
                )

                # PET "hỏi" CT
                temp_pet = pet_attn_ct(
                    query=pet_tokens_flat,
                    key_value=ct_tokens_flat
                )

            # Cập nhật đặc trưng cho vòng lặp tiếp theo
            ct_tokens_flat = temp_ct
            pet_tokens_flat = temp_pet

        # 4. Kết hợp đặc trưng cuối cùng
        # Chúng ta lấy trung bình đặc trưng của CT và PET đã được "trộn"
        fused_tokens_flat = (ct_tokens_flat + pet_tokens_flat) / 2.0

        # 5. Định hình lại (Reshape) về dạng bản đồ 3D
        # Chuyển [B, N, D] -> [B, T, H', W', D]
        fused_tokens_3d = rearrange(
            fused_tokens_flat,
            'b (t h w) d -> b t h w d',
            t=t, h=h, w=w
        )

        # 6. Chuyển về định dạng $F_visual$
        # Chuyển [B, T, H', W', D] -> [B, D, T, H', W']
        # Đây là shape chuẩn để các lớp RoIAlign 3D hoạt động (kênh D ở dim 1)
        F_visual = rearrange(fused_tokens_3d, 'b t h w d -> b d t h w')

        return F_visual.contiguous()