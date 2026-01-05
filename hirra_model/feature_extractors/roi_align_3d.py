import torch
import torch.nn as nn
import torch.nn.functional as F

def exists(val):
    return val is not None

class RoIAlign3D(nn.Module):
    def __init__(self, output_size: tuple = (7, 7, 7), d_model: int = 512):
        """
        Mục đích: Trích xuất đặc trưng cho các vùng (RoI) 3D từ một
                  bản đồ đặc trưng 3D (F_visual).
                  Sử dụng F.grid_sample để nội suy 3 chiều (trilinear).

        Đầu vào Khởi tạo:
            - output_size (tuple): Kích thước (Depth, Height, Width)
                                   của đặc trưng RoI sau khi pool.
                                   Ví dụ: (7, 7, 7).
            - d_model (int): Chiều đặc trưng cuối cùng (phải khớp
                             với d_model của LLM).
        """
        super().__init__()
        self.output_size = output_size
        self.projection = nn.Linear(512, d_model) # Giả sử D (từ F_visual) là 512
        self.pooler = nn.AdaptiveAvgPool3d(output_size)
        
        # Tạo một lưới (grid) tọa độ chuẩn hóa 1 lần
        # Grid này có shape [Oz, Oy, Ox, 3] và giá trị từ -1 đến 1
        # F.grid_sample yêu cầu grid theo thứ tự (x, y, z)
        grid_z, grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, output_size[0]),
            torch.linspace(-1.0, 1.0, output_size[1]),
            torch.linspace(-1.0, 1.0, output_size[2]),
            indexing="ij"
        )
        # Sắp xếp lại thành [Oz, Oy, Ox, 3] với thứ tự (x, y, z)
        self.base_grid = torch.stack((grid_x, grid_y, grid_z), dim=-1)
        self.base_grid = nn.Parameter(self.base_grid.unsqueeze(0), requires_grad=False)
        
    def forward(self, 
                F_visual: torch.Tensor, 
                boxes_list: list, 
                original_spatial_dims: tuple):
        """
        Mục đích: Hàm forward chính.

        Đầu vào:
            - F_visual (torch.Tensor): Bản đồ đặc trưng 3D từ VisionEncoder.
              Shape: [B, D, T_feat, H_feat, W_feat]
              Ví dụ: [2, 512, 101, 8, 8]
              
            - boxes_list (list[torch.Tensor]): Một list (độ dài B)
              của các tensor. Mỗi tensor chứa các bounding box cho 1 item
              trong batch.
              Shape của mỗi tensor: [num_rois, 6]
              6 tọa độ là (z_min, y_min, x_min, z_max, y_max, x_max)
              ⚠️ LƯU Ý QUAN TRỌNG: Thứ tự toạ độ phải là (Z, Y, X) để khớp
              với shape của volume là (T/Z, H/Y, W/X)!
              Tọa độ này là tọa độ GỐC (pixel)
              ví dụ: (100, 50, 60, 120, 70, 80) trên ảnh 201x128x128.
              
            - original_spatial_dims (tuple): Kích thước (T, H, W)
              của ảnh gốc mà tọa độ box được tính trên đó.
              Ví dụ: (201, 128, 128)

        Đầu ra:
            - all_rois_features (list[torch.Tensor]): Một list (độ dài B)
              của các tensor. Mỗi tensor chứa đặc trưng đã được chiếu (project)
              cho các RoI của item đó.
              Shape của mỗi tensor: [num_rois, d_model]
        """
        
        all_rois_features = []
        B, D, T_feat, H_feat, W_feat = F_visual.shape
        device = F_visual.device
        
        # Chuyển kích thước gốc sang tensor
        # Shape [3] -> [1, 3]
        orig_dims_tensor = torch.tensor(
            [original_spatial_dims[2], original_spatial_dims[1], original_spatial_dims[0]],
            device=device, dtype=torch.float32
        ).unsqueeze(0) # Sắp xếp là (W, H, Z) để khớp với grid (x, y, z)
        
        # Chuyển grid cơ sở sang device
        base_grid = self.base_grid.to(device) # Shape: [1, Oz, Oy, Ox, 3]

        for i in range(B):
            F_visual_item = F_visual[i:i+1] # Shape: [1, D, T, H, W]
            boxes = boxes_list[i]           # Shape: [num_rois, 6]

            if boxes.shape[0] == 0:
                # Không có RoI nào cho item này
                all_rois_features.append(torch.empty(0, 512, device=device)) # 512 là d_model
                continue

            # 1. Chuẩn hóa tọa độ Bounding Box
            # Chuyển tọa độ (z_min, y_min, x_min, z_max, y_max, x_max)
            # sang tọa độ (x, y, z) và chuẩn hóa về [-1, 1]

            # FIX: Tính spatial_scale để account cho downsampling
            # Feature map dimensions: T_feat, H_feat, W_feat
            # Original image dimensions: T_orig, H_orig, W_orig
            T_orig, H_orig, W_orig = original_spatial_dims
            spatial_scale = torch.tensor(
                [W_feat / W_orig, H_feat / H_orig, T_feat / T_orig],
                device=device, dtype=torch.float32
            ).unsqueeze(0)  # Shape: [1, 3] as (x_scale, y_scale, z_scale)

            # Tách tọa độ
            boxes_min_zyx = boxes[:, 0:3] # (z, y, x)
            boxes_max_zyx = boxes[:, 3:6] # (z, y, x)

            # Đảo ngược thứ tự thành (x, y, z)
            boxes_min_xyz = torch.stack([boxes_min_zyx[:, 2], boxes_min_zyx[:, 1], boxes_min_zyx[:, 0]], dim=1)
            boxes_max_xyz = torch.stack([boxes_max_zyx[:, 2], boxes_max_zyx[:, 1], boxes_max_zyx[:, 0]], dim=1)

            # FIX: Scale boxes từ original image space → feature map space
            boxes_min_xyz_scaled = boxes_min_xyz * spatial_scale  # [num_rois, 3]
            boxes_max_xyz_scaled = boxes_max_xyz * spatial_scale  # [num_rois, 3]

            # Tạo feature map dimensions tensor để normalize
            feature_dims_tensor = torch.tensor(
                [W_feat, H_feat, T_feat],
                device=device, dtype=torch.float32
            ).unsqueeze(0)  # Shape: [1, 3] as (W, H, T)

            # Chuẩn hóa về [0, 1] trong feature map space
            norm_boxes_min_xyz = boxes_min_xyz_scaled / feature_dims_tensor # [num_rois, 3]
            norm_boxes_max_xyz = boxes_max_xyz_scaled / feature_dims_tensor # [num_rois, 3]

            # Chuẩn hóa về [-1, 1] để phù hợp với grid_sample
            norm_boxes_min_xyz = (norm_boxes_min_xyz * 2) - 1
            norm_boxes_max_xyz = (norm_boxes_max_xyz * 2) - 1
            
            # Tính toán tâm (center) và độ rộng (span) của box
            box_centers = (norm_boxes_max_xyz + norm_boxes_min_xyz) / 2.0
            box_spans = (norm_boxes_max_xyz - norm_boxes_min_xyz) / 2.0
            
            # Mở rộng dims để "broadcast"
            # [num_rois, 3] -> [num_rois, 1, 1, 1, 3]
            box_centers_view = box_centers.view(-1, 1, 1, 1, 3)
            box_spans_view = box_spans.view(-1, 1, 1, 1, 3)

            # 2. Tạo Sampling Grid cho từng Box
            # Sử dụng grid cơ sở [-1, 1] và biến đổi (scale/shift)
            # nó vào từng bounding box
            # base_grid: [1, Oz, Oy, Ox, 3]
            # sampling_grid: [num_rois, Oz, Oy, Ox, 3]
            sampling_grid = base_grid * box_spans_view + box_centers_view
            
            # 3. Lấy mẫu đặc trưng (Sample Features)
            # F.grid_sample yêu cầu input [N, C, Din, Hin, Win]
            # và grid [N, Dout, Hout, Wout, 3]
            # (Thứ tự D, H, W trong grid_sample hơi ngược, nó mong (x, y, z))
            
            # Lặp lại F_visual_item cho mỗi RoI
            # [1, D, T_feat, H_feat, W_feat] -> [num_rois, D, T_feat, H_feat, W_feat]
            F_visual_item_repeated = F_visual_item.repeat(boxes.shape[0], 1, 1, 1, 1)

            # Lấy mẫu!
            # FIX: Use 'trilinear' for 3D volumetric interpolation (not 'bilinear')
            roi_features = F.grid_sample(
                F_visual_item_repeated,
                sampling_grid,
                mode='bilinear',  # torch grid_sample 5D uses 'bilinear' for trilinear interp
                padding_mode='zeros',
                align_corners=False # Rất quan trọng khi dùng grid [-1, 1]
            )
            
            # roi_features shape: [num_rois, D, Oz, Oy, Ox]
            # ví dụ: [5, 512, 7, 7, 7]
            
            # 4. Pooling và Chiếu (Project)
            # Chúng ta dùng AdaptiveAvgPool3d để đảm bảo kích thước
            # (Mặc dù grid_sample đã làm, nhưng đây là bước "chắc chắn")
            pooled_features = self.pooler(roi_features) # [num_rois, D, 7, 7, 7]
            
            # Flatten và Project
            flattened_features = pooled_features.mean(dim=(2, 3, 4)) # [num_rois, D]
            projected_features = self.projection(flattened_features) # [num_rois, d_model]
            
            all_rois_features.append(projected_features)

        return all_rois_features


# Convenience wrapper for legacy imports/tests
def roi_align_3d_torch(
    F_visual: torch.Tensor,
    boxes_list: list,
    original_spatial_dims: tuple,
    output_size: tuple = (7, 7, 7),
    d_model: int = 512,
):
    """
    Thin wrapper to keep backward compatibility with older code/tests that
    expected a functional API. Internally instantiates RoIAlign3D and runs
    forward.
    """
    module = RoIAlign3D(output_size=output_size, d_model=d_model)
    return module(F_visual, boxes_list, original_spatial_dims)
