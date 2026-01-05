"""
Dataset loader cho preprocessed data (processed_480_npy).
Giữ nguyên cấu trúc thư mục từ JSON paths.
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


# Mapping body part (tiếng Anh → tiếng Việt)
BODY_PART_MAPPING = {
    'abdomen_pelvis': 'Ổ bụng - khung chậu',
    'chest': 'Lồng ngực',
    'head_neck': 'Đầu - cổ'
}

# Region names (tiếng Anh) phục vụ prompt ở các stage sau
REGION_NAMES_EN = {
    'abdomen_pelvis': 'vùng ổ bụng - khung chậu',
    'chest': 'vùng lồng ngực',
    'head_neck': 'vùng đầu - cổ'
}


class ViMedPETPreprocessedDatasetV2(Dataset):
    """
    Dataset cho preprocessed data (processed_480_npy).
    Đọc paths từ JSON, thay processed_npy → processed_480_npy.
    """

    def __init__(self, json_path, input_root, output_root, config):
        """
        Args:
            json_path: Path to JSON file (e.g., PETCT_parts_train_val_test.json)
            input_root: Input root (e.g., data/raw)
            output_root: Output root (e.g., training/stage1/data)
            config: Data config dict
        """
        self.input_root = input_root
        self.output_root = output_root
        self.config = config

        # Load JSON metadata
        with open(json_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)

        print(f"[Dataset] Loaded {len(self.samples)} samples from {json_path}")
        print(f"[Dataset] Input root: {input_root}")
        print(f"[Dataset] Output root: {output_root}")
        print(f"[Dataset] Reading from: processed_480_npy/")

    def _extract_body_part(self, img_path):
        """Extract body part từ path."""
        if 'abdomen_pelvis' in img_path:
            return 'abdomen_pelvis'
        elif 'chest' in img_path:
            return 'chest'
        elif 'head_neck' in img_path:
            return 'head_neck'
        return None

    def _load_report(self, report_path):
        """Đọc report JSON từ đường dẫn tương đối."""
        report_full_path = f"{self.input_root}/{report_path}"
        with open(report_full_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_text(self, report_data, body_part):
        """Extract text từ report tương ứng với body part."""
        # Lấy text từ "Mô tả hình ảnh"
        text_section = report_data.get('Mô tả hình ảnh', {})

        # Map body part sang tiếng Việt
        viet_body_part = BODY_PART_MAPPING.get(body_part, '')
        text = text_section.get(viet_body_part, '')

        # Fallback: nếu không tìm thấy, nối tất cả text
        if not text and isinstance(text_section, dict):
            text = " ".join(str(v) for v in text_section.values())

        return text.strip()

    def _build_prompt_context(self, report_data, body_part, report_rel_path):
        """Chuẩn bị metadata phục vụ prompt chi tiết cho Stage 2+."""
        clinical_history = report_data.get('Bệnh sử lâm sàng', '').strip()
        indication = report_data.get('Chỉ định', '').strip()

        # Parse patient ID từ đường dẫn: .../<patient>/report/<part>.json
        patient_id = None
        try:
            report_path = Path(report_rel_path)
            patient_id = report_path.parent.parent.name  # folder chứa report
        except Exception:
            patient_id = None

        return {
            'patient_id': patient_id,
            'body_part': body_part,
            'region_name_en': REGION_NAMES_EN.get(body_part, body_part or ''),
            'clinical_history': clinical_history or 'Không rõ',
            'indication': indication or 'Không rõ',
            'report_path': report_rel_path
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Get paths từ JSON (JSON đã có processed_480_npy)
        ct_rel_path = sample['ct_img_path']   # processed_480_npy/PETCT_2017/.../ct_...npy
        pet_rel_path = sample['pet_img_path']  # processed_480_npy/PETCT_2017/.../pet_...npy

        # Full paths cho CT/PET (đã trong processed_480_npy)
        ct_path = f"{self.output_root}/{ct_rel_path}"
        pet_path = f"{self.output_root}/{pet_rel_path}"

        # Load preprocessed data (float16 → float32 để tính toán)
        ct = np.load(ct_path)    # (201, 480, 480) float16
        pet = np.load(pet_path)  # (201, 480, 480) float16

        # Convert to tensor và cast sang float32 (quan trọng cho training!)
        ct = torch.from_numpy(ct).float()   # float16 → float32
        pet = torch.from_numpy(pet).float()  # float16 → float32

        # Extract text matching body part
        body_part = self._extract_body_part(sample['ct_img_path'])

        # Report path cần thay processed_480_npy → processed_npy (reports ở raw folder)
        report_rel_path = sample['report_path']
        report_data = self._load_report(report_rel_path)
        text = self._extract_text(report_data, body_part)
        prompt_context = self._build_prompt_context(report_data, body_part, report_rel_path)
        # print(f"[Dataset] Input text: {text[:100]}...")  # In 100 ký tự đầu
        # print(f"[Dataset] CT shape: {ct.shape}, PET shape: {pet.shape}")
        return {
            'ct': ct,           # [201, 480, 480] float32
            'pet': pet,         # [201, 480, 480] float32
            'text': text,
            'id': sample.get('id', idx),
            'body_part': body_part,
            'report_path': report_rel_path,
            'prompt_context': prompt_context
        }


def collate_fn(batch):
    """
    Collate function.
    Data đã ở shape đúng (201, 480, 480), chỉ cần stack.
    """
    ct_batch = [item['ct'] for item in batch]
    pet_batch = [item['pet'] for item in batch]

    return {
        'ct': torch.stack(ct_batch),           # [B, 201, 480, 480]
        'pet': torch.stack(pet_batch),         # [B, 201, 480, 480]
        'text': [item['text'] for item in batch],
        'id': [item['id'] for item in batch]
    }


def create_dataloaders(config):
    """
    Factory function to create dataloaders với preprocessed data v2.

    NOTE:
    - JSON từ: processed_480_npy/label/PETCT_parts_train_val_test.json
    - Images từ: processed_480_npy/PETCT_2017/.../ct_....npy (float16)
    - Reports từ: raw/processed_npy/.../report/....json
    """
     # Paths
    # json_path = "training/stage1/data/processed_480_npy/label/PETCT_parts_train_val_test.json"
    # input_root = "training/stage1/data"  # Cho reports
    # output_root = "training/stage1/data"  # Cho CT/PET

    data_cfg = config['data']
    data_root = data_cfg.get('root_dir', 'training/stage1/data/processed_480_npy')

    # Nếu root_dir chứa "processed_480_npy", lấy parent directory
    if 'processed_480_npy' in data_root:
        base_root = os.path.dirname(data_root) if data_root.endswith('processed_480_npy') else data_root.rsplit('/processed_480_npy', 1)[0]
    else:
        base_root = data_root

    input_root = base_root  # Cho reports
    output_root = base_root  # Cho CT/PET - QUAN TRỌNG: Phải giống input_root!

    def resolve_json_path(path_setting):
        if path_setting is None:
            return None
        path_setting = str(path_setting)
        if os.path.isabs(path_setting):
            return path_setting
        return os.path.join(data_root, path_setting)

    train_json_cfg = data_cfg.get('train_json')
    val_json_cfg = data_cfg.get('val_json')

    if train_json_cfg and val_json_cfg:
        train_json_path = resolve_json_path(train_json_cfg)
        val_json_path = resolve_json_path(val_json_cfg)

        print(f"[DataLoader] Train JSON: {train_json_path}")
        print(f"[DataLoader] Val JSON:   {val_json_path}")
        print(f"[DataLoader] Reports root: {input_root}")
        print(f"[DataLoader] Images root:  {output_root}")

        train_ds = ViMedPETPreprocessedDatasetV2(
            json_path=train_json_path,
            input_root=input_root,
            output_root=output_root,
            config=data_cfg
        )
        val_ds = ViMedPETPreprocessedDatasetV2(
            json_path=val_json_path,
            input_root=input_root,
            output_root=output_root,
            config=data_cfg
        )
    else:
        # Fallback: load single JSON và auto split (giữ compatibility)
        fallback_json = f"{base_root}/processed_480_npy/label/PETCT_parts_train_val_test.json"
        print(f"[DataLoader] JSON: {fallback_json}")
        print(f"[DataLoader] Reports root: {input_root}")
        print(f"[DataLoader] Images root: {output_root}")

        full_ds = ViMedPETPreprocessedDatasetV2(
            json_path=fallback_json,
            input_root=input_root,
            output_root=output_root,
            config=data_cfg
        )

        train_indices = []
        val_indices = []

        has_split_field = 'split' in full_ds.samples[0] if len(full_ds.samples) > 0 else False

        if has_split_field:
            print("[DataLoader] Using 'split' field from JSON")
            for idx, sample in enumerate(full_ds.samples):
                split = sample.get('split', 'train')
                if split == 'train':
                    train_indices.append(idx)
                elif split in ['val', 'validation']:
                    val_indices.append(idx)
        else:
            print("[DataLoader] ⚠️  JSON không có field 'split', chia 80/20")
            num_train = int(0.8 * len(full_ds))
            train_indices = list(range(num_train))
            val_indices = list(range(num_train, len(full_ds)))
            print(f"[DataLoader] Auto-split: {num_train} train, {len(full_ds) - num_train} val")

        from torch.utils.data import Subset
        train_ds = Subset(full_ds, train_indices)
        val_ds = Subset(full_ds, val_indices)

    print(f"[DataLoader] Train dataset: {len(train_ds)} samples")
    print(f"[DataLoader] Val dataset: {len(val_ds)} samples")

    # Training dataloader
    train_loader = DataLoader(
        train_ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=config['data']['pin_memory'],
        persistent_workers=True,
        prefetch_factor=config['data']['prefetch_factor'],
        drop_last=True
    )

    # Validation dataloader
    val_loader = DataLoader(
        val_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=config['data']['pin_memory'],
        persistent_workers=True,
        prefetch_factor=config['data']['prefetch_factor']
    )

    print(f"[DataLoader] Train batches: {len(train_loader)}")
    print(f"[DataLoader] Val batches: {len(val_loader)}")

    return train_loader, val_loader


if __name__ == '__main__':
    # Test
    import yaml

    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    train_loader, val_loader = create_dataloaders(config)

    print("\n[Test] Loading first batch...")
    batch = next(iter(train_loader))
    print(f"CT shape: {batch['ct'].shape}")
    print(f"PET shape: {batch['pet'].shape}")
    print(f"Text: {batch['text'][0][:100]}...")
    print(f"IDs: {batch['id']}")
    print("\n✓ Dataset test passed!")
