"""
Evaluation utilities for Stage 3 training.
Includes metrics computation and prediction export to CSV.
"""

import math
import csv
import os
from typing import Dict, List, Optional
from pathlib import Path
import torch
from tqdm.auto import tqdm


def compute_metrics(loss_value: float) -> Dict[str, float]:
    """
    Compute evaluation metrics from loss.

    Args:
        loss_value: Average loss value

    Returns:
        Dictionary with loss and perplexity
    """
    if loss_value is None or math.isnan(loss_value):
        return {"loss": float("inf"), "perplexity": float("inf")}

    # Clamp loss for perplexity calculation to avoid overflow
    loss_clamped = min(50.0, max(1e-5, loss_value))

    return {
        "loss": loss_value,
        "perplexity": math.exp(loss_clamped)
    }


def export_predictions_to_csv(
    predictions: List[Dict[str, str]],
    output_path: str,
    fieldnames: Optional[List[str]] = None
):
    """
    Export predictions to CSV file.

    Args:
        predictions: List of prediction dictionaries with keys:
            - patient_id: Patient ID
            - region: Body region
            - ground_truth: Ground truth text
            - prediction: Predicted text
            - (optional) num_rois: Number of ROIs
        output_path: Path to output CSV file
        fieldnames: Optional list of field names (default: inferred from first prediction)
    """
    if not predictions:
        print("[Warning] No predictions to export")
        return

    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Infer fieldnames if not provided
    if fieldnames is None:
        fieldnames = list(predictions[0].keys())

    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    print(f"[Evaluation] Exported {len(predictions)} predictions to {output_path}")


@torch.no_grad()
def generate_predictions(
    model,
    dataloader,
    tokenizer,
    device: str = "cuda",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> List[Dict[str, str]]:
    """
    Generate predictions from validation dataloader.

    Args:
        model: HiRRA model
        dataloader: Validation DataLoader
        tokenizer: HuggingFace tokenizer
        device: Device to run on
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter

    Returns:
        List of prediction dictionaries
    """
    model.eval()
    predictions = []

    print(f"[Evaluation] Generating predictions for {len(dataloader)} batches...")
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Generating", leave=False)
    for batch_idx, batch in pbar:
        # Move batch to device
        # ct = batch["ct"].to(device)
        # pet = batch["pet"].to(device)
        # boxes_list = [boxes.to(device) for boxes in batch["boxes_list"]]
        # ct = batch["ct"].to(device, dtype=torch.float32, non_blocking=True)
        # pet = batch["pet"].to(device, dtype=torch.float32, non_blocking=True)
        # boxes_list = [boxes.to(device, dtype=torch.float32) for boxes in batch["boxes_list"]]
        # prompt_ids_cpu = batch["input_ids"]
        # input_ids = batch["input_ids"].to(device)
        # attention_mask = batch["attention_mask"].to(device)
        # target_ids = batch["target_ids"]  # Keep on CPU for decoding


        ct = batch["ct"].to(device, dtype=torch.float32, non_blocking=True)
        pet = batch["pet"].to(device, dtype=torch.float32, non_blocking=True)
        prompt_ids_cpu = batch["input_ids"]
        boxes_list = [boxes.to(device, dtype=torch.float32, non_blocking=True) for boxes in batch["boxes_list"]]
        input_ids = batch["input_ids"].long().to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].long().to(device, non_blocking=True)
        target_ids = batch["target_ids"].long()

        #print(ct.dtype, pet.dtype, boxes_list[0].dtype if boxes_list else None)

        # Generate (wrap in autocast để model dùng bf16 an toàn)
        try:

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type in ("cuda", "xpu")
            ):
                outputs = model.generate(
                    ct_image=ct,
                    pet_image=pet,
                    boxes_list=boxes_list,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p
                )

            # Decode predictions and ground truth
            for i in range(len(batch["patient_ids"])):
                # Ground truth (decode target_ids)
                ground_truth = tokenizer.decode(target_ids[i], skip_special_tokens=True)

                # Prediction (decode generated ids)
                # CRITICAL FIX: When using inputs_embeds, .generate() returns ONLY new tokens
                # Do NOT skip any tokens! The output is already the prediction.
                if isinstance(outputs, torch.Tensor):
                    # If outputs is a tensor of token IDs
                    pred_ids = outputs[i]
                    # Use full output as prediction (no skipping needed)
                    new_ids = pred_ids
                    prediction = tokenizer.decode(new_ids, skip_special_tokens=True)
                else:
                    # If outputs is already decoded text
                    prediction = outputs[i] if isinstance(outputs, list) else str(outputs)

                predictions.append({
                    "patient_id": batch["patient_ids"][i],
                    "region": batch["regions"][i],
                    "num_rois": batch["num_rois"][i],
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                })

        except Exception as e:
            print(f"[Warning] Failed to generate for batch {batch_idx}: {e}")
            # Add empty predictions for this batch
            for i in range(len(batch["patient_ids"])):
                prompt_text = tokenizer.decode(prompt_ids_cpu[i], skip_special_tokens=False)
                predictions.append({
                    "patient_id": batch["patient_ids"][i],
                    "region": batch["regions"][i],
                    "num_rois": batch["num_rois"][i],
                    "input": prompt_text,
                    "ground_truth": tokenizer.decode(target_ids[i], skip_special_tokens=True),
                    "prediction": "[GENERATION FAILED]",
                })

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")

    print(f"[Evaluation] Generated {len(predictions)} predictions")
    return predictions


# Example usage
if __name__ == "__main__":
    # Test metrics computation
    loss = 2.5
    metrics = compute_metrics(loss)
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Perplexity: {metrics['perplexity']:.4f}")

    # Test CSV export
    test_predictions = [
        {
            "patient_id": "patient_1",
            "region": "chest",
            "num_rois": 3,
            "ground_truth": "Test ground truth 1",
            "prediction": "Test prediction 1"
        },
        {
            "patient_id": "patient_2",
            "region": "abdomen",
            "num_rois": 5,
            "ground_truth": "Test ground truth 2",
            "prediction": "Test prediction 2"
        }
    ]

    output_path = "/tmp/test_predictions.csv"
    export_predictions_to_csv(test_predictions, output_path)
    print(f"Test CSV exported to {output_path}")
