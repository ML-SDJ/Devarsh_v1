"""Vision pipeline for MedExplain AI – Multimodal Clinical Insight Assistant.

This module contains utilities to prepare the chest X-ray dataset, create a
transfer-learning model based on ResNet-18, run training/evaluation loops, and
produce Grad-CAM visualizations for interpretability.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.utils import make_grid

try:
    # torchcam is optional but provides a convenient Grad-CAM implementation.
    from torchcam.methods import SmoothGradCAMpp
except ImportError:  # pragma: no cover - fallback implementation
    SmoothGradCAMpp = None  # type: ignore

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "chest_xray"
MODEL_DIR = ROOT_DIR / "artifacts" / "vision"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Placeholder path for a fine-tuned checkpoint. Replace with your trained model.
FINETUNED_MODEL_PATH = MODEL_DIR / "resnet18_finetuned.pth"


def get_data_transforms(image_size: int = 224) -> Dict[str, transforms.Compose]:
    """Return torchvision transforms for training and validation/test splits."""
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalization,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return {"train": train_transform, "val": eval_transform, "test": eval_transform}


def create_dataloaders(
    data_dir: Path = DATA_DIR,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """Create PyTorch dataloaders for train, val, test splits using ImageFolder."""
    transforms_map = get_data_transforms()
    dataloaders: Dict[str, DataLoader] = {}
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Expected split directory '{split_dir}' not found. "
                "Download the Kaggle Chest X-Ray Pneumonia dataset and extract it to this location."
            )
        dataset = datasets.ImageFolder(split_dir, transform=transforms_map[split])
        shuffle = split == "train"
        dataloaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloaders


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Construct a ResNet-18 model for binary classification with transfer learning."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )
    return model


@dataclass
class TrainState:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


def _compute_accuracy(outputs: Tensor, labels: Tensor) -> float:
    preds = torch.argmax(outputs, dim=1)
    correct = torch.sum(preds == labels).item()
    return correct / labels.size(0)


def train_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler | None = None,
    num_epochs: int = 10,
    device: torch.device = DEVICE,
) -> Tuple[nn.Module, List[TrainState]]:
    """Train the model and return the best model along with training statistics."""
    model.to(device)
    best_model_wts = model.state_dict()
    best_acc = 0.0
    history: List[TrainState] = []

    for epoch in range(1, num_epochs + 1):
        epoch_train_loss = float("inf")
        epoch_train_acc = 0.0
        epoch_val_loss = float("inf")
        epoch_val_acc = 0.0

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = 0.0
            running_acc = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    acc = _compute_accuracy(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_acc += acc * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_acc / len(dataloaders[phase].dataset)

            if phase == "train" and scheduler:
                scheduler.step()

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

            if phase == "train":
                epoch_train_loss = epoch_loss
                epoch_train_acc = epoch_acc
            else:
                epoch_val_loss = epoch_loss
                epoch_val_acc = epoch_acc

        print(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}"
        )

        history.append(
            TrainState(
                epoch=epoch,
                train_loss=epoch_train_loss,
                train_acc=epoch_train_acc,
                val_loss=epoch_val_loss,
                val_acc=epoch_val_acc,
            )
        )

    model.load_state_dict(best_model_wts)
    return model, history


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device = DEVICE) -> float:
    """Evaluate the model on a dataloader and return accuracy."""
    model.to(device)
    model.eval()
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            running_corrects += torch.sum(preds == labels).item()
            total += labels.size(0)

    accuracy = running_corrects / total if total else 0.0
    print(f"Evaluation accuracy: {accuracy:.4f}")
    return accuracy


def generate_gradcam(
    model: nn.Module,
    input_tensor: Tensor,
    target_category: int | None = None,
    device: torch.device = DEVICE,
) -> Tensor:
    """Generate a Grad-CAM heatmap for the given input tensor.

    Parameters
    ----------
    model: Trained CNN model.
    input_tensor: A normalized input tensor of shape (1, C, H, W).
    target_category: Optional target class index for which to compute Grad-CAM.
    device: Device on which the model resides.
    """
    model.to(device)
    model.eval()

    if SmoothGradCAMpp is not None:
        cam_extractor = SmoothGradCAMpp(model, target_layer="layer4")
        cam = cam_extractor(input_tensor.to(device), class_idx=target_category)
        heatmap = cam[0].detach().cpu()
    else:
        # Manual Grad-CAM implementation (fallback).
        gradients: List[Tensor] = []
        activations: List[Tensor] = []

        def backward_hook(module: nn.Module, grad_input: Tuple[Tensor, ...], grad_output: Tuple[Tensor, ...]):
            gradients.append(grad_output[0])

        def forward_hook(module: nn.Module, input: Tuple[Tensor, ...], output: Tensor):
            activations.append(output)

        target_module = dict(model.named_modules())["layer4"]
        forward_handle = target_module.register_forward_hook(forward_hook)
        backward_handle = target_module.register_backward_hook(backward_hook)  # type: ignore[arg-type]

        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad = True

        outputs = model(input_tensor)
        if target_category is None:
            target_category = outputs.argmax(dim=1).item()
        loss = outputs[:, target_category].sum()
        model.zero_grad()
        loss.backward()

        gradients_ = gradients[0]
        activations_ = activations[0]
        pooled_gradients = torch.mean(gradients_, dim=[0, 2, 3])
        for i in range(activations_.shape[1]):
            activations_[:, i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(activations_, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap /= torch.max(heatmap) + 1e-8

        forward_handle.remove()
        backward_handle.remove()

    return heatmap


def save_model(model: nn.Module, path: Path = FINETUNED_MODEL_PATH) -> None:
    """Save the trained model weights to disk."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path: Path = FINETUNED_MODEL_PATH, num_classes: int = 2) -> nn.Module:
    """Load model weights from disk."""
    model = build_model(num_classes=num_classes)
    if path.exists():
        state_dict = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {path}")
    else:
        print(f"Warning: fine-tuned model not found at {path}. Using randomly initialized head.")
    return model.to(DEVICE)


def visualize_heatmap_on_image(heatmap: Tensor, input_tensor: Tensor) -> Tensor:
    """Overlay the Grad-CAM heatmap onto the original image tensor for display."""
    heatmap = heatmap.unsqueeze(0)
    heatmap = heatmap.repeat(3, 1, 1)
    heatmap = heatmap / heatmap.max()
    overlay = 0.3 * input_tensor.cpu() + 0.7 * heatmap
    overlay = overlay.clamp(0, 1)
    return make_grid(overlay, normalize=True)


def ensure_data_directories() -> None:
    """Ensure the expected folder structure exists."""
    for split in ["train", "val", "test"]:
        split_dir = DATA_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_data_directories()
    print(f"Using device: {DEVICE}")
    if FINETUNED_MODEL_PATH.exists():
        model = load_model()
        print("Loaded fine-tuned model ready for evaluation.")
    else:
        print(
            "Fine-tuned model not found. Configure dataset paths and run training via train_model() "
            "to create a checkpoint."
        )
