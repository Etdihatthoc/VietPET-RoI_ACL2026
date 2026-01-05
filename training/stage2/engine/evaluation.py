"""
Evaluation utilities for Stage 2 training.
Includes metrics computation and prediction export to CSV.
"""

import math
import csv
from typing import Dict, List, Optional
from pathlib import Path
import torch


def summarize_validation(loss_value: float) -> Dict[str, float]:
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
    device: torch.device,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.9,
    max_samples: int = None
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
        max_samples: Maximum number of samples to generate (None = all)

    Returns:
        List of prediction dictionaries
    """
    model.eval()
    predictions = []

    total_batches = len(dataloader)
    if max_samples is not None:
        # Calculate number of batches needed
        max_batches = (max_samples + dataloader.batch_size - 1) // dataloader.batch_size
        max_batches = min(max_batches, total_batches)
        print(f"[Evaluation] Generating predictions for {max_batches}/{total_batches} batches (max_samples={max_samples})...")
    else:
        max_batches = total_batches
        print(f"[Evaluation] Generating predictions for {total_batches} batches...")

    for batch_idx, batch in enumerate(dataloader):
        # Stop if reached max_samples
        if max_samples is not None and len(predictions) >= max_samples:
            break
        # Move batch to device
        # IMPORTANT: Stage 2 uses [B, D, H, W] format (no channel dimension needed)
        ct = batch["ct"].to(device, dtype=torch.float32, non_blocking=True)
        pet = batch["pet"].to(device, dtype=torch.float32, non_blocking=True)

        # Add channel dimension [B, D, H, W] -> [B, 1, D, H, W] for model
        ct = ct.unsqueeze(1)
        pet = pet.unsqueeze(1)

        boxes_list = [boxes.to(device, dtype=torch.float32, non_blocking=True) for boxes in batch["boxes_list"]]
        input_ids = batch["input_ids"].long().to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].long().to(device, non_blocking=True)
        target_ids = batch["target_ids"].long()
        prompt_ids_cpu = batch["input_ids"]

        # Generate with autocast
        try:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda"
            ):
                # Generate predictions
                # Note: Generation config is now properly handled in hirra.py's generate() method
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
                # Input prompt (decode input_ids)
                input_text = tokenizer.decode(prompt_ids_cpu[i], skip_special_tokens=False)

                # Ground truth (decode target_ids)
                ground_truth = tokenizer.decode(target_ids[i], skip_special_tokens=True)

                # Prediction (decode generated ids)
                # CRITICAL: When using inputs_embeds, .generate() behavior varies:
                # - Some models return ONLY new tokens
                # - Some models return prompt + new tokens
                # We need to detect which case we're in
                if isinstance(outputs, torch.Tensor):
                    # If outputs is a tensor of token IDs
                    pred_ids = outputs[i]

                    # Debug on first batch: Check what .generate() returns
                    if batch_idx == 0 and i == 0:
                        print(f"\n[DEBUG] Decoding behavior check:")
                        print(f"  Input IDs length: {len(input_ids[i])}")
                        print(f"  Attention mask length: {attention_mask[i].sum().item()}")
                        print(f"  Output IDs length: {len(pred_ids)}")
                        print(f"  First 10 output tokens: {pred_ids[:10].tolist()}")
                        print(f"  Last 10 output tokens: {pred_ids[-10:].tolist()}")

                        # Try to decode the full output to see what it contains
                        full_decode = tokenizer.decode(pred_ids, skip_special_tokens=False)
                        print(f"  Full decode (first 200 chars): {full_decode[:200]}")

                    # CRITICAL FIX: When using inputs_embeds, .generate() returns ONLY new tokens
                    # Do NOT skip any tokens! The output is already the prediction.
                    new_ids = pred_ids

                    if batch_idx == 0 and i == 0:
                        print(f"  → Using full output as prediction (inputs_embeds mode)")
                        print(f"  → Output length: {len(new_ids)} tokens")

                    prediction = tokenizer.decode(new_ids, skip_special_tokens=True)

                    # Debug: Check if generation was successful
                    if batch_idx == 0 and i < 3:
                        print(f"  [Sample {i}] new_ids length: {len(new_ids)}, prediction length: {len(prediction)} chars")
                        if len(prediction) == 0:
                            print(f"  [Sample {i}] WARNING: Empty prediction! new_ids: {new_ids[:20].tolist()}")

                    if len(new_ids) < 10:
                        print(f"[WARNING] Sample {i}: Only {len(new_ids)} new tokens generated (input_len={input_ids_len}, output_len={output_len})")
                else:
                    # If outputs is already decoded text
                    prediction = outputs[i] if isinstance(outputs, list) else str(outputs)

                predictions.append({
                    "patient_id": batch["patient_ids"][i],
                    "region": batch["regions"][i],
                    "num_rois": batch["num_rois"][i],
                    "input": input_text,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                })

        except Exception as e:
            import traceback
            print(f"[ERROR] Failed to generate for batch {batch_idx}: {e}")
            print(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
            # Add empty predictions for this batch
            for i in range(len(batch["patient_ids"])):
                input_text = tokenizer.decode(prompt_ids_cpu[i], skip_special_tokens=False)
                predictions.append({
                    "patient_id": batch["patient_ids"][i],
                    "region": batch["regions"][i],
                    "num_rois": batch["num_rois"][i],
                    "input": input_text,
                    "ground_truth": tokenizer.decode(target_ids[i], skip_special_tokens=True),
                    "prediction": f"[GENERATION FAILED: {str(e)[:100]}]",
                })

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")

    print(f"[Evaluation] Generated {len(predictions)} predictions")
    return predictions
