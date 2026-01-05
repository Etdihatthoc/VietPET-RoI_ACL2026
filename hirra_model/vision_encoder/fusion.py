import torch
from torch import nn, einsum
from einops import rearrange, repeat

# Import LayerNorm từ file helper chúng ta đã tạo
from .attention_helper import LayerNorm
from torch.nn.functional import scaled_dot_product_attention
def exists(val):
    return val is not None

def rearrange_many(tensors, pattern, **kwargs):
    """Apply rearrange to multiple tensors with same pattern and kwargs"""
    return [rearrange(t, pattern, **kwargs) for t in tensors]

class CrossAttentionLayer(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, dropout=0.):
        """
        Mục đích: Một khối Cross-Attention chuẩn của Transformer.
                  Dùng để 'query' (ví dụ: CT) có thể "nhìn" vào 
                  'key' và 'value' (ví dụ: PET).
        """
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        # Chúng ta dùng LayerNorm từ file attention_helper
        #
        self.norm_query = LayerNorm(dim)
        self.norm_kv = LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Thêm một lớp FeedForward (FFN) theo sau Cross-Attention
        # để xử lý thông tin (giống như một khối Transformer hoàn chỉnh)
        self.ffn = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, key_value, mask=None):
        """
        Đầu vào:
            - query (torch.Tensor): Đặc trưng nguồn (ví dụ: CT).
              Shape: [B, N, D] (N = T * H' * W')
            - key_value (torch.Tensor): Đặc trưng đích (ví dụ: PET).
              Dùng cho cả Key và Value.
              Shape: [B, N, D]
        
        Đầu ra:
            - out (torch.Tensor): Đặc trưng 'query' đã được "trộn" thông tin.
              Shape: [B, N, D]
        """
        # Lưu lại query ban đầu cho kết nối residual (cộng)
        residual_query = query
        
        # Chuẩn hóa (Normalize)
        query = self.norm_query(query)
        kv_input = self.norm_kv(key_value)

        # Biến đổi tuyến tính (Linear projections)
        q = self.to_q(query)
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)

        # Tách thành các 'heads'
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=self.heads)

        # Sử dụng Flash Attention 2 (scaled_dot_product_attention)
        # Tiết kiệm memory ~3-4x so với vanilla attention
        attn_mask = None
        if exists(mask):
            # Reshape mask cho SDPA: [B, 1, 1, seq_len]
            attn_mask = rearrange(mask, 'b j -> b 1 1 j')

        # Flash Attention with memory-efficient kernel
        out = scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale
        )

        # Gộp các 'heads' lại
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # Lớp đầu ra + kết nối residual
        out = self.to_out(out) + residual_query
        
        # Lớp Feed-forward + kết nối residual
        out = self.ffn(out) + out
        return out