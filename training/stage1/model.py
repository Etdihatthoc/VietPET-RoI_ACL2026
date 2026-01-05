"""
MedicalCLIP Model for Stage 1 Pretraining

Components:
- Vision Encoder: MultimodalEncoder (CT + PET fusion) from hirra_model
- Text Encoder: PhoBERT (Vietnamese BERT)
- Projection Heads: Vision → Latent, Text → Latent
- CLIP Loss: Contrastive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import sys
from contextlib import nullcontext

# Import MultimodalEncoder từ hirra_model
sys.path.append('/home/user01/aiotlab/sondinh/ACL 2026/Hirra_model')
from hirra_model.vision_encoder.multimodal_encoder import MultimodalEncoder


class ProjectionHead(nn.Module):
    """Simple 2-layer MLP projection head."""

    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class MedicalCLIP(nn.Module):
    """
    CLIP-style model cho CT/PET medical imaging và Vietnamese text.

    Architecture:
        CT/PET → MultimodalEncoder → Vision Projection → Normalized [512D]
        Text → PhoBERT → Text Projection → Normalized [512D]
        Loss: Symmetric contrastive loss (InfoNCE)
    """

    def __init__(self, config):
        super().__init__()

        # ==================== Vision Encoder ====================
        # Sử dụng MultimodalEncoder từ hirra_model (CT + PET fusion)
        ctvit_config = {
            'dim': config['model']['ctvit_dim'],
            'codebook_size': 8192,
            'image_size': 480,  # Sau patch embedding
            'patch_size': config['model']['ctvit_patch_size'],
            'temporal_patch_size': config['model']['ctvit_temporal_patch_size'],
            'spatial_depth': 4,
            'temporal_depth': 4,
            'dim_head': 32,
            'heads': 8
        }

        # Get gradient checkpointing setting from config (default: True)
        use_gradient_checkpointing = config['model'].get('use_gradient_checkpointing', True)

        self.vision_encoder = MultimodalEncoder(
            ctvit_config=ctvit_config,
            num_fusion_layers=config['model']['num_fusion_layers'],
            use_gradient_checkpointing=use_gradient_checkpointing
        )

        # Load pretrained CTViT nếu có
        if config['model'].get('pretrained_ctvit'):
            print(f"[Model] Loading pretrained CTViT from {config['model']['pretrained_ctvit']}")
            self.vision_encoder.load_pretrained_ctvit(config['model']['pretrained_ctvit'])
        else:
            print("[Model] Training CTViT from scratch")

        # Convert vision encoder to specified dtype (e.g., bfloat16)
        vision_dtype_cfg = config['model'].get('vision_dtype')
        if vision_dtype_cfg == 'bfloat16':
            print("[Model] Converting vision encoder to bfloat16")
            self.vision_encoder = self.vision_encoder.to(torch.bfloat16)
        elif vision_dtype_cfg == 'float16':
            print("[Model] Converting vision encoder to float16")
            self.vision_encoder = self.vision_encoder.to(torch.float16)

        self.vision_dtype = next(self.vision_encoder.parameters()).dtype
        print(f"[Model] Vision encoder dtype: {self.vision_dtype}")

        # ==================== Text Encoder ====================
        # PhoBERT (Vietnamese BERT) with optional 4-bit quantization
        print(f"[Model] Loading {config['model']['phobert_model']}")

        text_model_kwargs = {}
        load_text_in_4bit = config['model'].get('load_text_in_4bit', False)
        load_text_in_8bit = config['model'].get('load_text_in_8bit', False)

        if load_text_in_4bit and load_text_in_8bit:
            raise ValueError("Both load_text_in_4bit and load_text_in_8bit are set; please enable only one.")

        if load_text_in_4bit:
            if BitsAndBytesConfig is None:
                raise ImportError(
                    "BitsAndBytesConfig not available. Install bitsandbytes and transformers>=4.30 to use 4-bit loading."
                )
            print("[Model] Using 4-bit quantization for text encoder")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            text_model_kwargs.update({
                "quantization_config": quant_config,
                "device_map": "auto",
            })
        elif load_text_in_8bit:
            if BitsAndBytesConfig is None:
                raise ImportError(
                    "BitsAndBytesConfig not available. Install bitsandbytes and transformers>=4.30 to use 8-bit loading."
                )
            print("[Model] Using 8-bit quantization for text encoder")
            text_model_kwargs.update({
                "load_in_8bit": True,
                "device_map": "auto",
            })

        self.text_encoder = AutoModel.from_pretrained(
            config['model']['phobert_model'],
            **text_model_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['model']['phobert_model']
        )

        # Freeze PhoBERT nếu config yêu cầu
        if config['model'].get('freeze_phobert', False):
            print("[Model] Freezing PhoBERT parameters")
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        else:
            print("[Model] PhoBERT parameters are trainable")

        # ==================== Projection Heads ====================
        latent_dim = config['model']['latent_dim']

        # Vision: 512D (CTViT output) → latent_dim
        self.vision_proj = ProjectionHead(
            input_dim=512,
            output_dim=latent_dim,
            dropout=0.1
        )

        # Text: 768D (PhoBERT output) → latent_dim
        self.text_proj = ProjectionHead(
            input_dim=768,
            output_dim=latent_dim,
            dropout=0.1
        )

        # ==================== Learnable Temperature ====================
        # CLIP temperature (learnable), initialized to 1/0.07
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6593)  # ln(1/0.07)

        print(f"[Model] MedicalCLIP initialized with latent_dim={latent_dim}")

    def encode_image(self, ct, pet):
        """
        Encode CT/PET images thành normalized embedding vector.

        Args:
            ct: [B, D, H, W] - CT volumes
            pet: [B, D, H, W] - PET volumes

        Returns:
            [B, latent_dim] - L2-normalized image embeddings
        """
        # Convert to vision dtype and add channel dimension
        ct = ct.to(self.vision_dtype).unsqueeze(1)  # [B, 1, D, H, W]
        pet = pet.to(self.vision_dtype).unsqueeze(1)  # [B, 1, D, H, W]

        # Use autocast if vision encoder is in half precision (fp16/bf16)
        use_autocast = (ct.device.type == 'cuda' and
                        self.vision_dtype in (torch.float16, torch.bfloat16))
        autocast_ctx = (torch.autocast(device_type='cuda', dtype=self.vision_dtype)
                        if use_autocast else nullcontext())

        with autocast_ctx:
            # MultimodalEncoder fusion - Output: [B, 512, D', H', W']
            fused = self.vision_encoder(ct, pet)

        # Global average pooling: [B, 512, D', H', W'] → [B, 512]
        pooled = F.adaptive_avg_pool3d(fused, output_size=1)
        pooled = pooled.view(pooled.size(0), -1)  # [B, 512]

        # Project to latent space
        projected = self.vision_proj(pooled)  # [B, latent_dim]

        # L2 normalize (CRITICAL for CLIP!)
        normalized = F.normalize(projected, dim=-1)

        return normalized

    def encode_text(self, texts):
        """
        Encode Vietnamese text thành normalized embedding vector.

        Args:
            texts: List of strings (Vietnamese descriptions)

        Returns:
            [B, latent_dim] - L2-normalized text embeddings
        """
        # Tokenize text
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        ).to(next(self.parameters()).device)

        # PhoBERT encoding
        # Output: last_hidden_state [B, seq_len, 768]
        outputs = self.text_encoder(**tokens)

        # Mean pooling over sequence dimension
        # [B, seq_len, 768] → [B, 768]
        pooled = outputs.last_hidden_state.mean(dim=1)

        # Convert to float32 để match với text_proj dtype
        # (PhoBERT với 4-bit quantization output float16)
        pooled = pooled.float()

        # Project to latent space
        projected = self.text_proj(pooled)  # [B, latent_dim]

        # L2 normalize
        normalized = F.normalize(projected, dim=-1)

        return normalized

    def forward(self, ct, pet, texts):
        """
        Forward pass for training.

        Args:
            ct: [B, D, H, W] - CT volumes
            pet: [B, D, H, W] - PET volumes
            texts: List of strings

        Returns:
            image_embeds: [B, latent_dim] - Normalized
            text_embeds: [B, latent_dim] - Normalized
            temperature: scalar - exp(logit_scale)
        """
        image_embeds = self.encode_image(ct, pet)
        text_embeds = self.encode_text(texts)

        # Temperature parameter (learnable)
        temperature = self.logit_scale.exp()

        return image_embeds, text_embeds, temperature


def clip_loss(image_embeds, text_embeds, temperature):
    """
    CLIP contrastive loss (symmetric InfoNCE).

    Ý tưởng:
    - Positive pairs (image_i, text_i) phải có similarity cao
    - Negative pairs (image_i, text_j) i≠j phải có similarity thấp

    Args:
        image_embeds: [B, D] - L2 normalized
        text_embeds: [B, D] - L2 normalized
        temperature: scalar - Temperature scaling parameter

    Returns:
        loss: scalar - Symmetric contrastive loss
    """
    batch_size = image_embeds.size(0)
    device = image_embeds.device

    # Compute similarity matrix (cosine similarity vì đã normalize)
    # [B, D] @ [D, B] = [B, B]
    logits = temperature * image_embeds @ text_embeds.T

    # Labels: diagonal elements là positive pairs
    labels = torch.arange(batch_size, device=device)

    # Image-to-Text loss
    # Mỗi image phải match đúng text của nó
    loss_i2t = F.cross_entropy(logits, labels)

    # Text-to-Image loss
    # Mỗi text phải match đúng image của nó
    loss_t2i = F.cross_entropy(logits.T, labels)

    # Symmetric average
    loss = (loss_i2t + loss_t2i) / 2

    return loss


# Test
if __name__ == '__main__':
    import yaml

    # Load config
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    # Create model
    model = MedicalCLIP(config).cuda()

    # Test forward
    # Batch size phải khớp với số texts trong CLIP
    batch_size = 3
    ct = torch.randn(batch_size, 201, 480, 480).cuda().half()
    pet = torch.randn(batch_size, 201, 480, 480).cuda().half()

    texts = [
        "Hình ảnh bắt xạ theo đặc điểm sinh lý ở gan, lách.",
        "Hình ảnh cho thấy sự tăng sinh mạch máu ở vùng tổn thương.",
        "Hình ảnh PET cho thấy sự chuyển hóa glucose tăng cao."
    ]

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    
    image_embeds, text_embeds, temp = model(ct, pet, texts)

    print(f"Image embeddings: {image_embeds.shape}")
    print(f"Text embeddings: {text_embeds.shape}")
    print(f"Temperature: {temp.item():.4f}")

    # Test loss
    loss = clip_loss(image_embeds, text_embeds, temp)
    print(f"CLIP loss: {loss.item():.6f}")
