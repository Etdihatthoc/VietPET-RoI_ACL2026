import torch
import torch.nn as nn
from typing import List

class VisualProjector(nn.Module):
    def __init__(self,
                 vision_feature_dim: int,
                 llm_hidden_dim: int):
        """
        Mục đích: "Dịch" đặc trưng hình ảnh (từ Module 1 & 2) sang
                  không gian embedding của LLM (Module 3).
                  Đây là trái tim của việc "tiêm" (inject) hình ảnh.

        Đầu vào Khởi tạo:
            - vision_feature_dim (int): Chiều đặc trưng từ Module 1/2.
              Ví dụ: 512 (từ CTViT).
            - llm_hidden_dim (int): Chiều ẩn của LLM (ví dụ: 4096 cho Qwen-7B).
        """
        super().__init__()

        # Lớp "dịch" cho các đặc trưng ROI
        self.roi_projector = nn.Linear(vision_feature_dim, llm_hidden_dim)

        # Lớp "dịch" cho các đặc trưng toàn cục (từ QFormer)
        self.global_projector = nn.Linear(vision_feature_dim, llm_hidden_dim)

        # STAGE 4 FIX: Lớp chuẩn hóa RIÊNG BIỆT cho ROI và Global features
        # Điều này cho phép mỗi loại feature duy trì phân bố riêng của nó
        self.roi_norm = nn.LayerNorm(llm_hidden_dim)      # Riêng cho ROI
        self.global_norm = nn.LayerNorm(llm_hidden_dim)   # Riêng cho Global

    def forward(self, 
                F_rois_list: List[torch.Tensor], 
                F_global_vis: torch.Tensor):
        """
        Mục đích: Chạy forward qua các lớp "dịch".

        Đầu vào:
            - F_rois_list (List[torch.Tensor]): List các tensor đặc trưng RoI
              (đầu ra của RoIAlign3D - Module 2).
              Shape list: B x [num_rois, vision_feature_dim]
              Ví dụ: 2 x [Tensor[5, 512], Tensor[3, 512]]
              
            - F_global_vis (torch.Tensor): Tensor đặc trưng toàn cục
              (đầu ra của QFormer - Module 2).
              Shape: [B, num_context_vectors, vision_feature_dim]
              Ví dụ: [2, 32, 512]

        Đầu ra:
            - projected_rois_list (List[torch.Tensor]): List các
              đặc trưng RoI đã được "dịch".
              Shape list: B x [num_rois, llm_hidden_dim]
              Ví dụ: 2 x [Tensor[5, 4096], Tensor[3, 4096]]
              
            - projected_global (torch.Tensor): Đặc trưng toàn cục
              đã được "dịch".
              Shape: [B, num_context_vectors, llm_hidden_dim]
              Ví dụ: [2, 32, 4096]
        """

        # 1. "Dịch" đặc trưng Toàn cục (Global) - DÙNG global_norm
        projected_global = self.global_projector(F_global_vis)
        projected_global = self.global_norm(projected_global)  # STAGE 4 FIX: dùng global_norm

        # 2. "Dịch" đặc trưng Vùng (RoI) - DÙNG roi_norm
        projected_rois_list = []
        for F_rois in F_rois_list:
            if F_rois.shape[0] > 0:
                projected_rois = self.roi_projector(F_rois)
                projected_rois = self.roi_norm(projected_rois)  # STAGE 4 FIX: dùng roi_norm
                projected_rois_list.append(projected_rois)
            else:
                # Xử lý trường hợp không có RoI nào
                # Tạo tensor rỗng với đúng shape
                projected_rois_list.append(
                    torch.empty(0, self.roi_projector.out_features,
                                device=F_rois.device, dtype=F_rois.dtype)
                )

        return projected_rois_list, projected_global

# import torch
# import torch.nn as nn
# from typing import List

# class VisualProjector(nn.Module):
#     def __init__(self, 
#                  vision_feature_dim: int, 
#                  llm_hidden_dim: int):
#         """
#         Mục đích: "Dịch" đặc trưng hình ảnh (từ Module 1 & 2) sang
#                   không gian embedding của LLM (Module 3).
#                   Đây là trái tim của việc "tiêm" (inject) hình ảnh.

#         Đầu vào Khởi tạo:
#             - vision_feature_dim (int): Chiều đặc trưng từ Module 1/2.
#               Ví dụ: 512 (từ CTViT).
#             - llm_hidden_dim (int): Chiều ẩn của LLM (ví dụ: 4096 cho Qwen-7B).
#         """
#         super().__init__()
        
#         # Lớp "dịch" cho các đặc trưng ROI
#         self.roi_projector = nn.Linear(vision_feature_dim, llm_hidden_dim)
        
#         # Lớp "dịch" cho các đặc trưng toàn cục (từ QFormer)
#         self.global_projector = nn.Linear(vision_feature_dim, llm_hidden_dim)
        
#         # Lớp chuẩn hóa (normalize) sau khi "dịch"
#         self.norm = nn.LayerNorm(llm_hidden_dim)

#     def forward(self, 
#                 F_rois_list: List[torch.Tensor], 
#                 F_global_vis: torch.Tensor):
#         """
#         Mục đích: Chạy forward qua các lớp "dịch".

#         Đầu vào:
#             - F_rois_list (List[torch.Tensor]): List các tensor đặc trưng RoI
#               (đầu ra của RoIAlign3D - Module 2).
#               Shape list: B x [num_rois, vision_feature_dim]
#               Ví dụ: 2 x [Tensor[5, 512], Tensor[3, 512]]
              
#             - F_global_vis (torch.Tensor): Tensor đặc trưng toàn cục
#               (đầu ra của QFormer - Module 2).
#               Shape: [B, num_context_vectors, vision_feature_dim]
#               Ví dụ: [2, 32, 512]

#         Đầu ra:
#             - projected_rois_list (List[torch.Tensor]): List các
#               đặc trưng RoI đã được "dịch".
#               Shape list: B x [num_rois, llm_hidden_dim]
#               Ví dụ: 2 x [Tensor[5, 4096], Tensor[3, 4096]]
              
#             - projected_global (torch.Tensor): Đặc trưng toàn cục
#               đã được "dịch".
#               Shape: [B, num_context_vectors, llm_hidden_dim]
#               Ví dụ: [2, 32, 4096]
#         """
        
#         # 1. "Dịch" đặc trưng Toàn cục (Global)
#         projected_global = self.global_projector(F_global_vis)
#         projected_global = self.norm(projected_global)
        
#         # 2. "Dịch" đặc trưng Vùng (RoI)
#         projected_rois_list = []
#         for F_rois in F_rois_list:
#             if F_rois.shape[0] > 0:
#                 projected_rois = self.roi_projector(F_rois)
#                 projected_rois = self.norm(projected_rois)
#                 projected_rois_list.append(projected_rois)
#             else:
#                 # Xử lý trường hợp không có RoI nào
#                 # Tạo tensor rỗng với đúng shape
#                 projected_rois_list.append(
#                     torch.empty(0, self.roi_projector.out_features, 
#                                 device=F_rois.device, dtype=F_rois.dtype)
#                 )

#         return projected_rois_list, projected_global