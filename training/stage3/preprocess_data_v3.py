#!/usr/bin/env python3
"""
Resize CT/PET volumes to 201×480×480 and export unified ROI JSON.

Usage:
  python preprocess_data_v3.py \
      --json_path "training/stage3/data/50samples_roi_reformat_resized.json" \
      --input_root "training/stage3/data/raw" \
      --output_root "training/stage3/data"

"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm


TARGET_SHAPE = (201, 480, 480)  # (depth, height, width)


def resize_volume(volume: np.ndarray, target_shape=TARGET_SHAPE) -> np.ndarray:
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)


def scale_boxes(
    boxes: List[List[float]],
    old_shape: tuple[int, int, int],
    target_shape=TARGET_SHAPE,
) -> List[List[float]]:
    scale = np.array(
        [
            target_shape[2] / old_shape[2],  # X axis
            target_shape[1] / old_shape[1],  # Y axis
            target_shape[0] / old_shape[0],  # Z axis
        ],
        dtype=np.float32,
    )
    sc = np.concatenate([scale, scale])
    return [np.round(np.array(box, dtype=np.float32) * sc, 4).tolist() for box in boxes]


def process_sample(sample: Dict[str, Any], input_root: Path, output_root: Path) -> Dict[str, Any]:
    ct_path = input_root / sample["modalities"]["ct"]
    pet_path = input_root / sample["modalities"]["pet"]

    ct = np.load(ct_path)
    pet = np.load(pet_path)

    ct_resized = resize_volume(ct)
    pet_resized = resize_volume(pet)

    # write volumes
    out_ct = output_root / sample["modalities"]["ct"]
    out_pet = output_root / sample["modalities"]["pet"]
    out_ct.parent.mkdir(parents=True, exist_ok=True)
    out_pet.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_ct, ct_resized.astype(np.float32))
    np.save(out_pet, pet_resized.astype(np.float32))

    # unify boxes
    boxes_ct = [roi["modalities"]["ct"] for roi in sample["rois"]]
    boxes_pet = [roi["modalities"]["pet"] for roi in sample["rois"]]
    scaled_ct = scale_boxes(boxes_ct, ct.shape)
    scaled_pet = scale_boxes(boxes_pet, pet.shape)

    unified_rois = []
    for idx, roi in enumerate(sample["rois"]):
        # average CT/PET bounding box in resized space
        merged = np.mean([scaled_ct[idx], scaled_pet[idx]], axis=0).tolist()
        unified_rois.append(
            {
                "index": roi["index"],
                "label": roi.get("label"),
                "description": roi.get("description"),
                "bbox": merged,  # both modalities share this bbox now
            }
        )

    return {
        "patient_id": sample["patient_id"],
        "region": sample["region"],
        "modalities": {
            "ct": str(out_ct.relative_to(output_root)),
            "pet": str(out_pet.relative_to(output_root)),
        },
        "rois": unified_rois,
        "z_offset": sample.get("z_offset"),
        "z_range_full": sample.get("z_range_full"),
        "cropped_shape": TARGET_SHAPE[::-1],  # (H, W) for display if needed
        "num_rois": len(unified_rois),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    json_path = Path(args.json_path)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    manifest_path = output_root / "50samples_roi_reformat_resized.json"

    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = raw["samples"]
    processed = []

    for sample in tqdm(samples, desc="Resizing volumes"):
        processed.append(process_sample(sample, input_root, output_root))

    manifest = {"metadata": raw.get("metadata", {}), "samples": processed}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Saved resized volumes under: {output_root}")
    print(f"New manifest: {manifest_path}")


if __name__ == "__main__":
    main()
