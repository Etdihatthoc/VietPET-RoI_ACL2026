"""
HiRRA (Hierarchical Region-based Reporting Architecture)

File lắp ráp chính kết hợp tất cả các module:
- Module 1: Vision Encoder (CTViT for CT/PET fusion)
- Module 2: Feature Extractors (Improved version with FPN, Q-Former, SPP RoI)
- Module 3: Language Decoder (Qwen3-8B with projection)

Key innovation: "Stitching" logic - injecting visual features into text embeddings
to create a unified input sequence for the LLM.
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import List, Optional, Tuple, Dict
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import GenerationConfig

# --- Import các module con ---
# Module 1: Vision Encoder
from .vision_encoder.multimodal_encoder import MultimodalEncoder

# Module 2: Feature Extractors (IMPROVED VERSION)
from .feature_extractors.advanced_extractors import AdvancedFeatureExtractor

# Module 3: Language Decoder
from .language_decoder.llm_loader import LanguageDecoder
from .language_decoder.projection import VisualProjector
# ------------------------------


class HiRRA(nn.Module):
    """
    HiRRA: Hierarchical Region-based Reporting Architecture

    Kiến trúc multimodal cho medical report generation:
    - Nhận đầu vào: CT scans + PET scans + RoI boxes + text prompt
    - Xuất đầu ra: Generated medical report

    Sử dụng các module improved với SOTA techniques:
    - Multi-scale FPN for better multi-scale feature extraction
    - Improved Q-Former with self-attention and sparse sampling
    - Spatial Pyramid Pooling for RoI features
    """

    def __init__(self,
                 ctvit_config: Dict,
                 feature_extractor_config: str = 'improved',
                 language_decoder_name: str = "Qwen/Qwen3-8B",
                 original_spatial_dims: Tuple[int, int, int] = (201, 160, 160),
                 num_queries: int = 32,
                 language_decoder_4bit: bool = False,
                 language_decoder_kwargs: Optional[Dict] = None,
                 use_graph_reasoning: bool = True,
                 graph_config: str = 'basic'):
        """
        Khởi tạo HiRRA model

        Args:
            ctvit_config: Config cho CTViT (dim, patch_size, etc.)
            feature_extractor_config: 'basic', 'improved', or 'full'
                - 'basic': Original simple version (fastest)
                - 'improved': Medium improvements (recommended, good balance)
                - 'full': All SOTA improvements (best performance)
            language_decoder_name: Tên model LLM trên Hugging Face
            original_spatial_dims: Kích thước (T, H, W) của ảnh 3D gốc
            num_queries: Số query vectors cho global context
            language_decoder_4bit: Bật lượng tử 4-bit cho LLM (yêu cầu bitsandbytes)
            language_decoder_kwargs: Tham số bổ sung truyền cho LanguageDecoder
            use_graph_reasoning: Bật ROI-pair graph reasoning (NEW!)
            graph_config: 'basic' (GNN only) or 'hierarchical' (two-level graph) (NEW!)
        """
        super().__init__()

        self.original_spatial_dims = original_spatial_dims
        self.num_queries = num_queries
        self.feature_config = feature_extractor_config
        self.vision_dim = ctvit_config['dim']  # Store for later use

        print(f"[HiRRA] Initializing with config: {feature_extractor_config}")

        # --- Module 1: Vision Encoder ---
        print("[HiRRA] Loading Module 1: Vision Encoder (CTViT)...")
        self.vision_encoder = MultimodalEncoder(
            ctvit_config=ctvit_config
        )
        vision_dim = self.vision_dim  # e.g., 512
        print(f"[HiRRA] ✓ Vision Encoder loaded (dim={vision_dim})")

        # --- Module 3: Language Decoder (load first to get hidden_dim) ---
        print(f"[HiRRA] Loading Module 3: Language Decoder ({language_decoder_name})...")
        decoder_kwargs = language_decoder_kwargs or {}
        self.language_decoder = LanguageDecoder(
            model_name=language_decoder_name,
            load_in_4bit=language_decoder_4bit,
            **decoder_kwargs
        )
        llm_dim = self.language_decoder.llm_hidden_dim
        decoder_mode = "4-bit quantized" if language_decoder_4bit else "bf16"
        print(f"[HiRRA] ✓ Language Decoder loaded ({decoder_mode}, hidden_dim={llm_dim})")

        # Lấy các token ID đặc biệt
        self.global_token_id = self.language_decoder.global_token_id
        self.roi_token_id = self.language_decoder.roi_token_id
        self.tokenizer_pad_id = self.language_decoder.tokenizer.pad_token_id

        # --- Module 2: Feature Extractors (IMPROVED VERSION) ---
        print(f"[HiRRA] Loading Module 2: Feature Extractors ({feature_extractor_config})...")
        self.feature_extractor = AdvancedFeatureExtractor(
            config=feature_extractor_config,
            input_dim=vision_dim,
            d_model=vision_dim,
            num_queries=num_queries,
            use_multi_scale=True if feature_extractor_config in ['improved', 'full'] else False,
            use_sparse_qformer=True if feature_extractor_config == 'full' else False,
            use_spp_roi=True if feature_extractor_config in ['improved', 'full'] else False,
            use_graph_reasoning=use_graph_reasoning,
            graph_config=graph_config
        )

        # In thông tin config
        extractor_info = self.feature_extractor.get_config_info()
        print(f"[HiRRA] ✓ Feature Extractors loaded:")
        print(f"        - Multi-scale FPN: {extractor_info['use_multi_scale']}")
        print(f"        - Sparse Q-Former: {extractor_info['use_sparse_qformer']}")
        print(f"        - SPP RoI: {extractor_info['use_spp_roi']}")
        print(f"        - Graph Reasoning: {extractor_info['use_graph_reasoning']}")
        if extractor_info['use_graph_reasoning']:
            print(f"        - Graph Config: {extractor_info['graph_config']}")

        # --- Visual Projector (Bridge to LLM space) ---
        print(f"[HiRRA] Creating Visual Projector ({vision_dim} -> {llm_dim})...")
        self.visual_projector = VisualProjector(
            vision_feature_dim=vision_dim,
            llm_hidden_dim=llm_dim
        )
        print(f"[HiRRA] ✓ Visual Projector created")

        # Lấy LLM model cho generate()
        self.llm = self.language_decoder.model

        print(f"[HiRRA] ✓ HiRRA initialization complete!")
        print(f"[HiRRA] Total modules: Vision Encoder + Feature Extractors + Projector + LLM")

    def to(self, *args, **kwargs):
        """
        Ghi đè .to() để không chuyển LanguageDecoder 4-bit sang thiết bị khác
        (Params4bit không hỗ trợ .to()).
        """
        if getattr(self.language_decoder, "is_quantized_4bit", False):
            self.vision_encoder.to(*args, **kwargs)
            self.feature_extractor.to(*args, **kwargs)
            self.visual_projector.to(*args, **kwargs)
            return self
        return super().to(*args, **kwargs)

    def get_tokenizer(self):
        """
        Cung cấp tokenizer cho dataloader và inference
        """
        return self.language_decoder.tokenizer

    def load_pretrained_ctvit(self, checkpoint_path: str):
        """
        Tải trọng số pretrained cho CTViT encoder

        Args:
            checkpoint_path: Đường dẫn đến file checkpoint
        """
        print(f"[HiRRA] Loading pretrained CTViT from: {checkpoint_path}")
        self.vision_encoder.load_pretrained_ctvit(checkpoint_path)
        print(f"[HiRRA] ✓ Pretrained weights loaded successfully")

    def prepare_inputs_for_llm(self,
                               ct_image: torch.Tensor,
                               pet_image: torch.Tensor,
                               boxes_list: List[torch.Tensor],
                               input_ids: torch.Tensor,
                               attention_mask: torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        HÀM CỐT LÕI: "Stitching" Logic

        Thực hiện toàn bộ pipeline từ ảnh đến embedding sequence:
        1. Vision Encoder: Extract visual features
        2. Feature Extractors: Get global context + RoI features
        3. Visual Projector: Project to LLM space
        4. Text Embeddings: Get text embeddings
        5. Stitching: Merge visual + text embeddings

        Args:
            ct_image: [B, 1, T, H, W] - CT scan
            pet_image: [B, 1, T, H, W] - PET scan
            boxes_list: List of [num_rois, 6] bounding boxes per batch
            input_ids: [B, S_text] - Text token IDs
            attention_mask: [B, S_text] - Attention mask

        Returns:
            final_inputs_embeds: [B, S_final, D_llm] - Stitched embeddings
            final_attention_mask: [B, S_final] - Attention mask for stitched sequence
        """
        batch_size = ct_image.shape[0]
        embedding_layer = self.language_decoder.model.get_input_embeddings()
        llm_device = embedding_layer.weight.device
        target_dtype = embedding_layer.weight.dtype

        # --- STEP 1: Vision Encoder ---
        # Output: [B, D_vision, T_feat, H_feat, W_feat]
        # Example: [2, 512, 21, 8, 8] for patch_size=20, temporal_patch_size=10
        F_visual = self.vision_encoder(ct_image, pet_image)

        # --- STEP 2: Feature Extractors (IMPROVED) ---
        # Global: [B, N_queries, D_vision] - e.g., [2, 32, 512]
        # RoI: List[B] of [N_rois, D_vision]
        # NOTE: If graph reasoning is enabled, global will be [B, N_queries+1, D_vision]
        #       with the last vector being the graph summary
        F_global_vis, F_rois_list = self.feature_extractor(
            F_visual,
            boxes_list,
            self.original_spatial_dims,
            ct_image=ct_image,  # Pass for graph reasoning
            pet_image=pet_image  # Pass for graph reasoning
        )

        # --- STEP 3: Visual Projector ---
        # Project to LLM space
        # proj_global: [B, N_queries, D_llm] - e.g., [2, 32, 2048]
        # proj_rois_list: List[B] of [N_rois, D_llm]
        proj_rois_list, proj_global = self.visual_projector(
            F_rois_list,
            F_global_vis
        )

        proj_global = proj_global.to(device=llm_device, dtype=target_dtype)
        proj_rois_list = [
            roi.to(device=llm_device, dtype=target_dtype) for roi in proj_rois_list
        ]

        num_global_vectors = proj_global.shape[1]

        # --- STEP 4: Text Embeddings ---
        # [B, S_text, D_llm] - e.g., [2, 120, 2048]
        input_ids_llm = input_ids.to(llm_device)
        attention_mask_llm = attention_mask.to(llm_device)
        text_embeds = self.language_decoder.get_text_embeddings(input_ids_llm).to(
            dtype=target_dtype
        )

        # --- STEP 5: Stitching Logic ---
        # Replace <GLOBAL_VISUALS> and <ROI_VISUALS> tokens with actual visual features
        final_embeds_batch = []
        final_mask_batch = []
        max_len = 0

        for i in range(batch_size):
            # Get current item's data
            current_text_embeds = text_embeds[i]
            current_input_ids = input_ids_llm[i]
            current_attn_mask = attention_mask_llm[i]
            current_proj_rois = proj_rois_list[i]
            current_proj_global = proj_global[i]  # [N_queries, D_llm]

            current_embeds_list = []
            current_mask_list = []
            roi_idx = 0  # Counter for RoI usage

            # Iterate through each token
            for token_idx, token_id in enumerate(current_input_ids):
                if current_attn_mask[token_idx] == 0:
                    break  # Stop at padding

                token_id = token_id.item()

                if token_id == self.global_token_id:
                    # Replace <GLOBAL_VISUALS> with N_queries vectors
                    current_embeds_list.append(current_proj_global)
                    current_mask_list.append(
                        torch.ones(num_global_vectors, device=llm_device, dtype=torch.long)
                    )

                elif token_id == self.roi_token_id:
                    # Replace <ROI_VISUALS> with 1 RoI vector
                    if roi_idx < len(current_proj_rois):
                        current_embeds_list.append(
                            current_proj_rois[roi_idx:roi_idx+1]  # [1, D_llm]
                        )
                        current_mask_list.append(torch.ones(1, device=llm_device, dtype=torch.long))
                        roi_idx += 1
                    else:
                        # Error: More RoI tokens than provided boxes
                        # Skip this token or raise error
                        print(f"[WARNING] More <ROI_VISUALS> tokens than boxes in batch item {i}")
                        pass

                else:
                    # Regular text token
                    current_embeds_list.append(
                        current_text_embeds[token_idx:token_idx+1]  # [1, D_llm]
                    )
                    current_mask_list.append(torch.ones(1, device=llm_device, dtype=torch.long))

            # Concatenate all embeddings for this item
            final_embeds_batch.append(torch.cat(current_embeds_list, dim=0))
            final_mask_batch.append(torch.cat(current_mask_list, dim=0))

            # Track max length
            if final_mask_batch[-1].shape[0] > max_len:
                max_len = final_mask_batch[-1].shape[0]

        # --- STEP 6: Padding ---
        llm_dim = self.language_decoder.llm_hidden_dim
        final_inputs_embeds = torch.zeros(
            batch_size, max_len, llm_dim,
            device=llm_device, dtype=target_dtype
        )
        final_attention_mask = torch.zeros(
            batch_size, max_len,
            device=llm_device, dtype=torch.long
        )

        for i in range(batch_size):
            seq_len = final_embeds_batch[i].shape[0]
            final_inputs_embeds[i, :seq_len] = final_embeds_batch[i]
            final_attention_mask[i, :seq_len] = final_mask_batch[i]

        return final_inputs_embeds, final_attention_mask

    def forward(self,
                ct_image: torch.Tensor,
                pet_image: torch.Tensor,
                boxes_list: List[torch.Tensor],
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                target_ids: torch.Tensor
                ) -> CausalLMOutputWithPast:
        """
        Forward pass cho TRAINING

        Args:
            ct_image: [B, 1, T, H, W] - CT scan
            pet_image: [B, 1, T, H, W] - PET scan
            boxes_list: List of [num_rois, 6] bounding boxes
            input_ids: [B, S_prompt] - Prompt token IDs
            attention_mask: [B, S_prompt] - Prompt attention mask
            target_ids: [B, S_target] - Target token IDs (ground truth)

        Returns:
            CausalLMOutputWithPast: Contains loss and logits
        """
        # 1. Prepare prompt embeddings (text + visual features)
        prompt_embeds, prompt_mask = self.prepare_inputs_for_llm(
            ct_image, pet_image, boxes_list, input_ids, attention_mask
        )

        llm_device = prompt_embeds.device
        llm_dtype = prompt_embeds.dtype

        # 2. Prepare target embeddings
        target_ids_llm = target_ids.to(llm_device)
        target_embeds = self.language_decoder.get_text_embeddings(target_ids_llm).to(
            dtype=llm_dtype
        )
        target_mask = (target_ids_llm != self.tokenizer_pad_id)

        # 3. Concatenate prompt + target
        full_inputs_embeds = torch.cat([prompt_embeds, target_embeds], dim=1)
        full_attention_mask = torch.cat(
            [prompt_mask, target_mask.long()],
            dim=1
        )

        # 4. Prepare labels
        # We want LLM to predict target tokens, NOT prompt tokens
        # Set prompt labels to -100 (ignored by CrossEntropyLoss)
        prompt_labels = torch.full_like(
            prompt_mask, -100, dtype=torch.long, device=llm_device
        )
        target_labels = target_ids_llm.masked_fill(~target_mask, -100)
        full_labels = torch.cat([prompt_labels, target_labels], dim=1)

        # 5. Run LLM
        logits = self.language_decoder(
            inputs_embeds=full_inputs_embeds,
            attention_mask=full_attention_mask
        )

        # 6. Compute loss
        # Shift logits and labels (predict token i+1 from token i)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = full_labels[..., 1:].contiguous()

        loss = None
        if shift_labels is not None:
            label_smoothing = float(getattr(self, "label_smoothing", 0.0))
            loss_fct = CrossEntropyLoss(label_smoothing=label_smoothing)
            shift_logits = shift_logits.view(-1, self.llm.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,  # Disable KV cache for training
            hidden_states=None,
            attentions=None,
        )

    @torch.no_grad()
    def generate(self,
                 ct_image: torch.Tensor,
                 pet_image: torch.Tensor,
                 boxes_list: List[torch.Tensor],
                 input_ids: torch.Tensor,
                 attention_mask: torch.Tensor,
                 **generate_kwargs
                 ) -> torch.Tensor:
        """
        Generate cho INFERENCE

        Args:
            ct_image: [B, 1, T, H, W] - CT scan
            pet_image: [B, 1, T, H, W] - PET scan
            boxes_list: List of [num_rois, 6] bounding boxes
            input_ids: [B, S_prompt] - Prompt token IDs
            attention_mask: [B, S_prompt] - Prompt attention mask
            **generate_kwargs: Arguments for .generate()
                - max_new_tokens: Max tokens to generate
                - temperature: Sampling temperature
                - do_sample: Whether to use sampling
                - etc.

        Returns:
            generated_ids: [B, S_generated] - Generated token IDs
        """

        # 1. Prepare prompt embeddings (text + visual features)
        prompt_embeds, prompt_mask = self.prepare_inputs_for_llm(
            ct_image, pet_image, boxes_list, input_ids, attention_mask
        )

        # 2. ✅ FIX: Set critical generation parameters
        tokenizer = self.language_decoder.get_tokenizer()

        # Default generation config (can be overridden by generate_kwargs)
        default_config = {
            'max_new_tokens': 512,
            'min_new_tokens': 100,  # ✅ Force at least 100 new tokens for medical reports
            'do_sample': False,  # Use greedy decoding (deterministic)
            'num_beams': 1,  # No beam search for speed
        }

        # Set pad_token_id and eos_token_id
        if tokenizer.pad_token_id is not None:
            default_config['pad_token_id'] = tokenizer.pad_token_id
        elif tokenizer.eos_token_id is not None:
            default_config['pad_token_id'] = tokenizer.eos_token_id

        if tokenizer.eos_token_id is not None:
            default_config['eos_token_id'] = tokenizer.eos_token_id

        # Merge with user-provided kwargs (user kwargs take priority)
        final_config_dict = {**default_config, **generate_kwargs}

        # ✅ CRITICAL FIX: Create GenerationConfig object and pass as parameter
        # HuggingFace's .generate() doesn't respect unpacked kwargs for Qwen3
        # Must pass generation_config as an object parameter
        generation_config = GenerationConfig(**final_config_dict)

        # 3. Generate using LLM
        generated_ids = self.llm.generate(
            inputs_embeds=prompt_embeds,
            attention_mask=prompt_mask,
            generation_config=generation_config  # ✅ Pass as object, not **kwargs
        )

        return generated_ids

    def get_model_info(self) -> Dict:
        """
        Trả về thông tin về model configuration
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info = {
            'feature_extractor_config': self.feature_config,
            'vision_dim': self.vision_dim,
            'llm_hidden_dim': self.language_decoder.llm_hidden_dim,
            'num_queries': self.num_queries,
            'original_spatial_dims': self.original_spatial_dims,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'memory_footprint_fp32_GB': total_params * 4 / 1e9,
            'memory_footprint_bf16_GB': total_params * 2 / 1e9,
        }

        return info
