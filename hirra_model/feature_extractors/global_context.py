import torch
import torch.nn as nn
from einops import rearrange, repeat
import sys
import os

# --- Thiết lập Path để Import ---
# Thêm thư mục gốc (hirra_project) vào path
# để import từ module 'vision_encoder'
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Đi lùi 2 cấp (từ feature_extractors -> hirra_model -> hirra_project)
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(ROOT_DIR)

# Import các khối Transformer cơ bản từ module 1
# (Những khối này vốn từ file attention.py)
from hirra_model.vision_encoder.attention_helper import Attention, FeedForward, LayerNorm

def exists(val):
    return val is not None

class QFormerBlock(nn.Module):
    """
    Một khối QFormer chuẩn: Gồm 1 lớp Cross-Attention và 1 lớp FeedForward.
    """
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        
        # Lớp Cross-Attention:
        # 'query' (từ learnable queries) sẽ "hỏi"
        # 'context' (từ F_visual)
        self.attn = Attention(
            dim=dim, 
            dim_head=dim_head, 
            heads=heads, 
            causal=False,
            # quan trọng: dim_context=dim vì query và F_visual
            # đều có chiều đặc trưng là 'dim'
            dim_context=dim 
        )
        
        # Lớp FeedForward
        self.ffn = FeedForward(dim=dim, mult=ff_mult)

    def forward(self, query, context):
        """
        Đầu vào:
        - query (torch.Tensor): [B, num_queries, D]
        - context (torch.Tensor): [B, N_features, D]
        """
        # Cross-Attention
        # Chú ý: x=query, context=context
        # (kết nối residual được thực hiện bên trong hàm Attention)
        query = self.attn(query, context=context) + query
        
        # FeedForward
        query = self.ffn(query) + query
        
        return query

class GlobalContextExtractor(nn.Module):
    def __init__(self, 
                 input_dim: int = 512, 
                 d_model: int = 512, 
                 num_context_vectors: int = 32,
                 depth: int = 2): # Thêm 'depth' cho QFormer
        """
        Mục đích: Kiến trúc QFormer (đơn giản hóa)
                  Tóm tắt bản đồ F_visual thành một số lượng
                  vector ngữ cảnh (queries) cố định.
                  
        Đầu vào Khởi tạo:
            - input_dim (int): Chiều đặc trưng (D) của F_visual (ví dụ: 512).
            - d_model (int): Chiều đặc trưng đầu ra (phải khớp với LLM).
            - num_context_vectors (int): Số lượng vector "truy vấn" (query).
            - depth (int): Số tầng (block) QFormer.
        """
        super().__init__()
        assert input_dim == d_model, "QFormer yêu cầu input_dim và d_model phải bằng nhau."
        
        self.num_context_vectors = num_context_vectors
        self.d_model = d_model
        
        # 1. Các vector truy vấn (queries) có thể học được
        self.queries = nn.Parameter(
            torch.randn(1, num_context_vectors, d_model)
        )
        
        # 2. Lớp chuẩn hóa (Normalization) cho F_visual
        # Sử dụng LayerNorm từ attention_helper
        self.context_norm = LayerNorm(input_dim)
        
        # 3. Các tầng QFormer
        self.layers = nn.ModuleList(
            [QFormerBlock(dim=d_model) for _ in range(depth)]
        )

    def forward(self, F_visual: torch.Tensor):
        """
        Mục đích: Hàm forward chính của QFormer.
        
        Đầu vào:
            - F_visual (torch.Tensor): Bản đồ đặc trưng 3D.
              Shape: [B, D, T_feat, H_feat, W_feat]
              Ví dụ: [2, 512, 101, 8, 8]
              
        Đầu ra:
            - context_vectors (torch.Tensor): Các vector ngữ cảnh toàn cục.
              Shape: [B, num_context_vectors, d_model]
              Ví dụ: [2, 32, 512]
        """
        B = F_visual.shape[0]
        
        # 1. "Phẳng hóa" (Flatten) F_visual
        # [B, D, T, H, W] -> [B, D, N] (N = T*H*W)
        F_visual_flat = rearrange(F_visual, 'b d ... -> b d (...)')
        # Chuyển [B, D, N] -> [B, N, D]
        F_visual_flat = rearrange(F_visual_flat, 'b d n -> b n d')
        
        # 2. Chuẩn hóa F_visual
        F_visual_flat = self.context_norm(F_visual_flat)
        
        # 3. Lặp lại (repeat) các queries cho batch size
        # [1, num_queries, D] -> [B, num_queries, D]
        queries = repeat(self.queries, '1 n d -> b n d', b=B)
        
        # 4. Chạy qua các tầng QFormer
        for layer in self.layers:
            queries = layer(queries, context=F_visual_flat)
            
        # 'queries' bây giờ đã chứa thông tin tóm tắt của F_visual
        return queries