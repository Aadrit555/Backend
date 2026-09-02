"""Native In-House Custom Vision Classifier Engine.

Powered by PyTorch and TorchVision. Provides lightning-fast interactive transfer learning,
custom classification heads, in-memory model caching, and sub-10ms real-time inference
for webcam streams and image files.
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as tv_models
import torchvision.transforms as transforms

from backend.config import settings

# Device configuration (CUDA GPU if available, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory to persist trained custom vision models
_MODELS_DIR = settings.storage_root / "models" / "custom_classifier"
_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory cache for active classifier
_ACTIVE_MODEL: nn.Module | None = None
_ACTIVE_MODEL_ID: str | None = None
_ACTIVE_CLASSES: list[str] = []
_ACTIVE_BACKBONE: str = "mobilenet_v3_small"

# Standard ImageNet normalization parameters
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

# Evaluator transforms for inference (224x224)
EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
])

# Training transforms with light data augmentation
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
])


class InMemoryDataset(Dataset):
    """PyTorch Dataset over in-memory PIL images and integer labels."""

    def __init__(self, images: list[Image.Image], labels: list[int], transform: Any = None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            tensor_img = self.transform(img)
        else:
            tensor_img = transforms.ToTensor()(img)
        return tensor_img, label


def build_classifier_model(num_classes: int, backbone_name: str = "mobilenet_v3_small") -> nn.Module:
    """Instantiate a pretrained backbone with an adapted classification head."""
    backbone_name = backbone_name.lower().strip()

    if "resnet" in backbone_name:
        model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        # Freeze feature extraction layers for ultra-fast transfer learning
        for param in model.parameters():
            param.requires_grad = False

        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
        )
    else:
        # Default: mobilenet_v3_small (ultra-compact and blazing fast)
        model = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT)
        # Freeze feature extraction layers
        for param in model.parameters():
            param.requires_grad = False

        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
        )

    return model.to(DEVICE)


def get_models_dir() -> Path:
    """Return root directory for custom vision model artifacts."""
    return _MODELS_DIR


def list_models() -> list[dict[str, Any]]:
    """List all previously trained custom vision classifier models."""
    models: list[dict[str, Any]] = []
    if not _MODELS_DIR.exists():
        return models

    for meta_file in _MODELS_DIR.glob("*/metadata.json"):
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            models.append(data)
        except Exception:
            continue

    # Sort descending by created_at
    models.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return models


def train_classifier(
    classes_data: dict[str, list[str]],
    backbone: str = "mobilenet_v3_small",
    epochs: int = 10,
    lr: float = 0.001,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Train a native transfer-learning vision classifier from interactive samples.

    Args:
        classes_data: Dict mapping class names to lists of base64-encoded image strings.
        backbone: "mobilenet_v3_small" or "resnet18".
        epochs: Number of training epochs (default 10).
        lr: Learning rate for AdamW optimizer (default 0.001).
        batch_size: Mini-batch size (default 8).

    Returns:
        Metadata dict with model_id, backbone, classes, top1_accuracy, fit_time_seconds, etc.
    """
    if len(classes_data) < 2:
        raise ValueError("Custom Vision Classifier requires at least 2 distinct classes to train.")

    for cname, imgs in classes_data.items():
        if not imgs:
            raise ValueError(f"Class '{cname}' contains no image samples.")

    model_id = f"cv_{uuid.uuid4().hex[:10]}"
    start_time = time.time()

    class_names = list(classes_data.keys())
    num_classes = len(class_names)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    train_images: list[Image.Image] = []
    train_labels: list[int] = []
    val_images: list[Image.Image] = []
    val_labels: list[int] = []

    total_images = 0

    # 1. Decode base64 images into memory
    for cname, b64_list in classes_data.items():
        c_idx = class_to_idx[cname]
        c_images: list[Image.Image] = []

        for b64_str in b64_list:
            try:
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                img_data = base64.b64decode(b64_str)
                pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
                c_images.append(pil_img)
            except Exception as e:
                print(f"[Custom Vision Engine] Warning skipping corrupt sample: {e}")
                continue

        if not c_images:
            raise ValueError(f"No valid images could be decoded for class '{cname}'")

        total_images += len(c_images)

        # Split 80% train, 20% validation
        if len(c_images) > 1:
            split_idx = max(1, int(len(c_images) * 0.8))
            train_part = c_images[:split_idx]
            val_part = c_images[split_idx:]
        else:
            # Single image: use for both train and val
            train_part = c_images
            val_part = [c_images[0].copy()]

        train_images.extend(train_part)
        train_labels.extend([c_idx] * len(train_part))

        val_images.extend(val_part)
        val_labels.extend([c_idx] * len(val_part))

    # 2. Build datasets and data loaders
    train_dataset = InMemoryDataset(train_images, train_labels, transform=TRAIN_TRANSFORMS)
    val_dataset = InMemoryDataset(val_images, val_labels, transform=EVAL_TRANSFORMS)

    train_loader = DataLoader(train_dataset, batch_size=max(1, batch_size), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=max(1, batch_size), shuffle=False)

    # 3. Initialize model & optimizer
    model = build_classifier_model(num_classes=num_classes, backbone_name=backbone)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    print(
        f"[Custom Vision Engine] Training {model_id} ({backbone}) on {DEVICE} with "
        f"{total_images} samples across {num_classes} classes for {epochs} epochs..."
    )

    # 4. Training loop
    model.train()
    history = []
    final_loss = 0.0

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        epoch_loss = running_loss / max(1, total)
        epoch_acc = correct / max(1, total)
        final_loss = epoch_loss
        history.append({"epoch": epoch, "loss": round(epoch_loss, 4), "acc": round(epoch_acc, 4)})

    # 5. Validation evaluation
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val = x_val.to(DEVICE)
            y_val = y_val.to(DEVICE)
            out = model(x_val)
            _, preds = torch.max(out, 1)
            val_correct += (preds == y_val).sum().item()
            val_total += y_val.size(0)

    val_acc = val_correct / max(1, val_total)
    fit_time = round(time.time() - start_time, 2)

    # 6. Save model checkpoint & metadata
    dest_dir = _MODELS_DIR / model_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = dest_dir / "model.pth"

    checkpoint = {
        "model_id": model_id,
        "backbone": backbone,
        "classes": class_names,
        "num_classes": num_classes,
        "state_dict": model.state_dict(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "top1_accuracy": round(val_acc, 4),
        "final_loss": round(final_loss, 4),
        "fit_time_seconds": fit_time,
        "device": str(DEVICE),
    }
    torch.save(checkpoint, str(checkpoint_path))

    metadata = {
        "model_id": model_id,
        "backbone": backbone,
        "classes": class_names,
        "sample_counts": {k: len(v) for k, v in classes_data.items()},
        "total_samples": total_images,
        "epochs": epochs,
        "top1_accuracy": round(val_acc, 4),
        "final_loss": round(final_loss, 4),
        "fit_time_seconds": fit_time,
        "created_at": checkpoint["created_at"],
        "checkpoint_path": str(checkpoint_path),
        "history": history,
    }
    (dest_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # 7. Cache active model in memory
    load_active_model(model_id, checkpoint_path=checkpoint_path)

    print(f"[Custom Vision Engine] Training complete in {fit_time}s! Val Acc: {val_acc:.2%}")
    return metadata


def load_active_model(model_id: str, checkpoint_path: Path | str | None = None) -> nn.Module:
    """Load and cache the specified custom vision model into memory."""
    global _ACTIVE_MODEL, _ACTIVE_MODEL_ID, _ACTIVE_CLASSES, _ACTIVE_BACKBONE

    if checkpoint_path is None:
        checkpoint_path = _MODELS_DIR / model_id / "model.pth"

    cp_path = Path(checkpoint_path)
    if not cp_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {cp_path}")

    print(f"[Custom Vision Engine] Loading active model {model_id} from {cp_path}...")
    cp_data = torch.load(str(cp_path), map_location=DEVICE, weights_only=False)

    backbone_name = cp_data.get("backbone", "mobilenet_v3_small")
    classes = cp_data.get("classes", [])
    num_classes = len(classes)

    model = build_classifier_model(num_classes=num_classes, backbone_name=backbone_name)
    model.load_state_dict(cp_data["state_dict"])
    model.eval()

    _ACTIVE_MODEL = model
    _ACTIVE_MODEL_ID = model_id
    _ACTIVE_CLASSES = classes
    _ACTIVE_BACKBONE = backbone_name

    return _ACTIVE_MODEL


def predict_classification(
    image_input: bytes | str | Image.Image,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Run real-time classification inference on an input image.

    Args:
        image_input: Raw image bytes, base64 data URL string, or PIL Image.
        model_id: Optional model ID. If None, uses currently active or latest model.

    Returns:
        Dict with status, model_id, backbone, predictions list, top_class, top_confidence, speed_ms.
    """
    global _ACTIVE_MODEL, _ACTIVE_MODEL_ID, _ACTIVE_CLASSES, _ACTIVE_BACKBONE

    # Resolve PIL Image
    if isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    elif isinstance(image_input, str):
        if "," in image_input:
            image_input = image_input.split(",", 1)[1]
        decoded = base64.b64decode(image_input)
        pil_img = Image.open(io.BytesIO(decoded)).convert("RGB")
    elif isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
    else:
        raise ValueError("Invalid image input type. Expected bytes, str (base64), or PIL Image.")

    # Resolve model
    if model_id and (_ACTIVE_MODEL_ID != model_id or _ACTIVE_MODEL is None):
        target_path = _MODELS_DIR / model_id / "model.pth"
        if target_path.exists():
            load_active_model(model_id, target_path)
        else:
            raise FileNotFoundError(f"Model ID '{model_id}' was not found in {_MODELS_DIR}")
    elif _ACTIVE_MODEL is None:
        models = list_models()
        if models:
            load_active_model(models[0]["model_id"])
        else:
            raise RuntimeError("No custom vision models have been trained yet. Please train a model first.")

    t0 = time.time()
    input_tensor = EVAL_TRANSFORMS(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = _ACTIVE_MODEL(input_tensor)
        probs = F.softmax(logits, dim=1)[0].tolist()

    latency_ms = round((time.time() - t0) * 1000, 1)

    predictions = []
    for idx, conf in enumerate(probs):
        cname = _ACTIVE_CLASSES[idx] if idx < len(_ACTIVE_CLASSES) else f"Class_{idx}"
        predictions.append({
            "class": cname,
            "confidence": round(float(conf), 4),
        })

    predictions.sort(key=lambda x: x["confidence"], reverse=True)
    top_pred = predictions[0] if predictions else {"class": "none", "confidence": 0.0}

    return {
        "status": "success",
        "model_id": _ACTIVE_MODEL_ID,
        "backbone": _ACTIVE_BACKBONE,
        "predictions": predictions,
        "top_class": top_pred["class"],
        "top_confidence": top_pred["confidence"],
        "speed_ms": latency_ms,
    }

