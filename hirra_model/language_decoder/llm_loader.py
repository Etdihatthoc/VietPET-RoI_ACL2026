import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None


class LanguageDecoder(nn.Module):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        *,
        load_in_4bit: bool = False,
        quantization_config: dict = None,
        device_map: str = "auto",
        model_kwargs: dict = None
    ):
        """
        Mục đích: Tải và chứa lõi LLM (Qwen-3-8B) từ Hugging Face.
                  Cung cấp các hàm tiện ích.

        Model info:
            - Qwen3-8B: ~8B parameters, hidden_size≈4096
            - Official Qwen model (không cần trust_remote_code)
        """
        super().__init__()
        model_kwargs = dict(model_kwargs or {})
        self.is_quantized_4bit = load_in_4bit

        # 1. Tải Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        # 2. Tải mô hình LLM
        if load_in_4bit:
            if BitsAndBytesConfig is None:
                raise ImportError(
                    "Không tìm thấy BitsAndBytesConfig. "
                    "Hãy cài đặt transformers>=4.34 và bitsandbytes để sử dụng load_in_4bit=True."
                )

            default_bnb_kwargs = dict(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            if quantization_config is not None:
                default_bnb_kwargs.update(quantization_config)

            quant_config = BitsAndBytesConfig(**default_bnb_kwargs)

            if "device_map" not in model_kwargs:
                model_kwargs["device_map"] = device_map

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                **model_kwargs
            )
        else:
            model_kwargs.setdefault("torch_dtype", torch.bfloat16)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )

        # 3. Lấy chiều ẩn của LLM
        # Qwen3-8B: hidden_size ≈ 4096
        self.llm_hidden_dim = self.model.config.hidden_size
        
        # 4. Thêm các token đặc biệt (placeholder) vào tokenizer
        # <ROI_VISUALS> là placeholder cho 1 vector đặc trưng RoI
        # <GLOBAL_VISUALS> là placeholder cho các vector đặc trưng toàn cục
        special_tokens = [
            "<ROI_VISUALS>",
            "<GLOBAL_VISUALS>"
        ]
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': special_tokens}
        )
        # Cập nhật kích thước embedding của LLM để khớp với 2 token mới
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Lấy ID của các token đặc biệt này
        self.roi_token_id = self.tokenizer.convert_tokens_to_ids("<ROI_VISUALS>")
        self.global_token_id = self.tokenizer.convert_tokens_to_ids("<GLOBAL_VISUALS>")

    def get_tokenizer(self):
        """
        Mục đích: Trả về tokenizer để Dataloader có thể sử dụng.
        """
        return self.tokenizer

    def get_text_embeddings(self, input_ids: torch.Tensor):
        """
        Mục đích: Chuyển đổi ID token văn bản sang embedding
                  (vector ngôn ngữ).

        Đầu vào:
            - input_ids (torch.Tensor): ID của các token văn bản.
              Shape: [B, text_seq_len]
              
        Đầu ra:
            - text_embeds (torch.Tensor): Embedding của các token văn bản.
              Shape: [B, text_seq_len, llm_hidden_dim]
              Ví dụ: [2, 120, 4096]
        """
        # Lấy lớp embedding gốc của LLM
        return self.model.get_input_embeddings()(input_ids)

    def forward(self, 
                inputs_embeds: torch.Tensor, 
                attention_mask: torch.Tensor):
        """
        Mục đích: Chạy LLM với một chuỗi embedding (đã kết hợp
                  cả văn bản và hình ảnh).

        Đầu vào:
            - inputs_embeds (torch.Tensor): Chuỗi embedding cuối cùng.
              Shape: [B, total_seq_len, llm_hidden_dim]
              Ví dụ: [2, 256, 4096]
              
            - attention_mask (torch.Tensor): Mặt nạ (mask)
              cho chuỗi cuối cùng.
              Shape: [B, total_seq_len]

        Đầu ra:
            - logits (torch.Tensor): Đầu ra "thô" của LLM.
              Shape: [B, total_seq_len, vocab_size]
        """
        # Đây là hàm forward chuẩn của Hugging Face LLM
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs.logits
