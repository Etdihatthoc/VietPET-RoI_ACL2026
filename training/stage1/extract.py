from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator

import numpy as np
from tqdm.auto import tqdm

RAW_ROOT = Path("/media/gpus/New Volume/ViMed-PET/raw")

CONFIG = {
    "ratios": {
        "head_neck": 0.25,
        "chest": 0.25,
        "abdomen_pelvis": 0.50,
    },
    "overlap": 15,
    "modalities": ("CT", "PET"),
    "save_dtype": np.int16,      # None nếu muốn giữ nguyên dtype
    "overwrite": False,
}

REPORT_SECTION_MAP = {
    "head_neck": "Đầu - cổ",
    "chest": "Lồng ngực",
    "abdomen_pelvis": "Ổ bụng - khung chậu",
}

# Thư mục chứa dữ liệu đầu ra (ảnh + report sau khi chia)
OUTPUT_BASE = RAW_ROOT / "processed_npy"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Các file index ban đầu (train/val/test) theo từng năm
JSON_SPLITS = {
    "PETCT_2017": RAW_ROOT / "PETCT_2017_train_val_test.json"#,
    # "PETCT_2018": RAW_ROOT / "PETCT_2018_train_val_test.json",
    # "PETCT_2019": RAW_ROOT / "PETCT_2019_train_val_test.json",
}

def greedy_segments(num_slices: int, ratios: Dict[str, float], overlap: int) -> Dict[str, slice]:
    names = list(ratios.keys())
    weights = np.array([ratios[name] for name in names], dtype=float)
    weights /= weights.sum()

    counts = np.floor(weights * num_slices).astype(int)
    remainder = num_slices - counts.sum()
    for idx in range(remainder):
        counts[idx % len(counts)] += 1

    bounds = np.concatenate(([0], np.cumsum(counts)))
    bounds[-1] = num_slices

    segments: Dict[str, slice] = {}
    for idx, name in enumerate(names):
        start = bounds[idx]
        end = bounds[idx + 1]
        if idx > 0:
            start = max(0, start - overlap)
        if idx < len(names) - 1:
            end = min(num_slices, end + overlap)
        if start >= end:
            continue
        segments[name] = slice(int(start), int(end))
    return segments


def split_volume_with_segments(volume: np.ndarray, segments: Dict[str, slice]) -> Dict[str, np.ndarray]:
    return {name: volume[slc].copy() for name, slc in segments.items()}


def load_npy_volumes(ct_path: Path, pet_path: Path) -> Dict[str, dict]:
    volumes = {}
    ct_array = np.load(ct_path)
    volumes["CT"] = {"array": ct_array, "original_dtype": ct_array.dtype}

    pet_array = np.load(pet_path)
    volumes["PET"] = {"array": pet_array, "original_dtype": pet_array.dtype}

    return volumes


def split_report_sections(report_data: dict, section_map: dict[str, str]) -> dict[str, dict]:
    base_fields = {k: v for k, v in report_data.items() if k != "Mô tả hình ảnh"}
    descriptions = report_data.get("Mô tả hình ảnh", {})
    part_reports = {}
    for part_key, section_name in section_map.items():
        part_reports[part_key] = {
            **base_fields,
            "Mô tả hình ảnh": {section_name: descriptions.get(section_name, "")},
        }
    return part_reports


def build_multilingual_reports(report_vi: dict, report_en: dict | None, section_map: dict[str, str]) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    if report_vi:
        reports.update(split_report_sections(report_vi, section_map))
    if report_en:
        for part_key, payload in split_report_sections(report_en, section_map).items():
            reports[f"{part_key}_en"] = payload
    return reports


def save_patient_arrays(
    splits: Dict[str, Dict[str, np.ndarray]],
    volumes_meta: Dict[str, Dict],
    patient_image_dir: Path,
    *,
    overwrite: bool,
    save_dtype,
) -> Dict[str, Dict[str, str]]:
    patient_image_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: Dict[str, Dict[str, str]] = defaultdict(dict)

    for modality, part_arrays in splits.items():
        target_dtype = save_dtype or volumes_meta[modality]["original_dtype"]
        for part_name, array in part_arrays.items():
            out_path = patient_image_dir / f"{modality.lower()}_{part_name}.npy"
            if out_path.exists() and not overwrite:
                saved_paths[modality][part_name] = str(out_path)
                continue
            np.save(out_path, array.astype(target_dtype, copy=False))
            saved_paths[modality][part_name] = str(out_path)

    return saved_paths


def save_patient_reports(part_reports: dict, patient_report_dir: Path, overwrite: bool) -> Dict[str, str]:
    patient_report_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for report_key in sorted(part_reports):
        out_path = patient_report_dir / f"{report_key}.json"
        if out_path.exists() and not overwrite:
            saved[report_key] = str(out_path)
            continue
        with out_path.open("w", encoding="utf-8") as fp:
            json.dump(part_reports[report_key], fp, ensure_ascii=False, indent=2)
        saved[report_key] = str(out_path)
    return saved


def iter_split_records() -> Iterator[tuple[str, dict]]:
    for year_tag, json_path in JSON_SPLITS.items():
        records = json.loads(json_path.read_text(encoding="utf-8"))
        for record in records:
            yield year_tag, record
            
def process_record(year_tag: str, record: dict) -> dict:
    year_dir = RAW_ROOT / year_tag
    ct_path = year_dir / record["ct_img_path"]
    pet_path = year_dir / record["pet_img_path"]
    report_path = year_dir / record["report_path"]

    if not ct_path.exists() or not pet_path.exists():
        raise FileNotFoundError(f"Missing npy volume: {ct_path} or {pet_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report file: {report_path}")

    volumes = load_npy_volumes(ct_path, pet_path)
    ct_slices = volumes["CT"]["array"].shape[0]
    pet_slices = volumes["PET"]["array"].shape[0]
    if ct_slices != pet_slices:
        raise ValueError(f"Slice mismatch CT={ct_slices}, PET={pet_slices} for {ct_path.name}")

    segments = greedy_segments(ct_slices, CONFIG["ratios"], CONFIG["overlap"])
    splits = {
        modality: split_volume_with_segments(info["array"], segments)
        for modality, info in volumes.items()
    }

    month_name = Path(record["ct_img_path"]).parts[0]
    patient_id = Path(ct_path).stem
    patient_root = OUTPUT_BASE / year_tag / month_name / patient_id

    image_paths = save_patient_arrays(
        splits,
        volumes,
        patient_root / "image",
        overwrite=CONFIG["overwrite"],
        save_dtype=CONFIG["save_dtype"],
    )

    report_json = json.loads(report_path.read_text(encoding="utf-8"))
    report_en_path = report_path.with_name(report_path.stem + "_en.json")
    report_json_en = json.loads(report_en_path.read_text(encoding="utf-8")) if report_en_path.exists() else None

    part_reports = build_multilingual_reports(report_json, report_json_en, REPORT_SECTION_MAP)
    report_paths = save_patient_reports(
        part_reports,
        patient_root / "report",
        overwrite=CONFIG["overwrite"],
    )

    return {
        "year": year_tag,
        "month": month_name,
        "source_record": record,
        "patient_id": patient_id,
        "segments": {k: (v.start, v.stop) for k, v in segments.items()},
        "image_parts": image_paths,
        "report_parts": report_paths,
    }

# %%
from zipfile import ZipFile
from pathlib import Path
import shutil
import json

RAW_ROOT = Path("/media/gpus/Do not enter/ViMed-PET")
TEMP_ROOT = Path("/mnt/disk1/SonDinh/SonDinh")
DEST_ROOT = Path("/mnt/disk1/SonDinh/SonDinh/DICE_model/training/stage1/processed_480_npy")

DEST_ROOT.mkdir(parents=True, exist_ok=True)
TEMP_ROOT.mkdir(parents=True, exist_ok=True)

def split_report_file(report_path: Path, dest_dir: Path, overwrite: bool = False):
    """Tách 1 report (và report_en nếu có) thành 3 phần và lưu vào dest_dir."""
    report_json = json.loads(report_path.read_text(encoding="utf-8"))
    report_en_path = report_path.with_name(report_path.stem + "_en.json")
    report_json_en = None
    if report_en_path.exists():
        report_json_en = json.loads(report_en_path.read_text(encoding="utf-8"))

    part_reports = build_multilingual_reports(report_json, report_json_en, REPORT_SECTION_MAP)
    save_patient_reports(part_reports, dest_dir, overwrite=overwrite)

# %%
import subprocess
from tqdm.auto import tqdm
zip_files = sorted(RAW_ROOT.glob("*.zip"))
print(f"Found {len(zip_files)} zip archives")

from subprocess import Popen, PIPE, STDOUT

def extract_with_7z(archive_path: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "7z", "x",
        str(archive_path),
        f"-o{dest_dir}",
        "-y",          # auto yes
        "-aoa"         # overwrite all
    ]

    with Popen(cmd, stdout=PIPE, stderr=STDOUT, text=True, bufsize=1) as proc:
        for line in proc.stdout:
            line = line.rstrip("\n")
            print(f"[7z] {line}")
        returncode = proc.wait()

    if returncode != 0:
        raise RuntimeError(f"7z failed on {archive_path} (exit code {returncode})")


for zip_path in tqdm(zip_files, desc="Zip archives"):
    print(f"\n[Processing] {zip_path.name}")
    temp_dir = TEMP_ROOT / zip_path.stem
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    extract_with_7z(zip_path, temp_dir)
    
    print(f"Extracted to {temp_dir}")
    # Các bước xử lý report như trước ...
    for report_file in tqdm(temp_dir.rglob("*.json"), desc=f"Reports {zip_path.name}", leave=False):
    #for report_file in temp_dir.rglob("*.json"):
        name = report_file.name
        if name.endswith("_en.json") or name.endswith("_metadata.json"):
            continue
        if "reports" not in report_file.parts:
            continue

        rel_path = report_file.relative_to(temp_dir)
        parts = rel_path.parts
        if len(parts) < 3:
            raise ValueError(f"Unexpected report path structure: {rel_path}")

        year_dir, month_dir = parts[0], parts[1]
        patient_dir = report_file.stem
        dest_report_dir = DEST_ROOT / year_dir / month_dir / patient_dir / "report"
        dest_report_dir.mkdir(parents=True, exist_ok=True)

        split_report_file(report_file, dest_report_dir, overwrite=False)

    shutil.rmtree(temp_dir)
    print(f"[Done] {zip_path.name}")
