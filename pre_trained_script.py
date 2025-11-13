"""
Enhanced Chest X-Ray Pneumonia Classifier
Features: Better organization, logging, metrics tracking, and error handling
"""
import os
import logging
from pathlib import Path
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm


# ==========================================
# Configuration and Setup
# ==========================================

class Config:
    """Centralized configuration"""
    DATA_DIR = Path('data/chest_xray')
    BATCH_SIZE = 32
    NUM_EPOCHS = 25
    LEARNING_RATE = 1e-3
    PATIENCE = 4
    WEIGHT_DECAY = 1e-4
    IMG_SIZE = 224
    NUM_WORKERS = 4
    SEED = 42


def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ==========================================
# Data Preparation
# ==========================================

def get_transforms() -> Dict[str, transforms.Compose]:
    """Define data augmentation and normalization transforms"""
    return {
        'train': transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


def create_balanced_sampler(dataset):
    """Create weighted sampler to handle class imbalance"""
    targets = [s[1] for s in dataset.samples]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in targets]
    return WeightedRandomSampler(sample_weights, len(sample_weights))


def load_data(logger) -> Tuple[DataLoader, DataLoader, DataLoader, datasets.ImageFolder]:
    """Load and prepare datasets with balanced sampling"""
    data_transforms = get_transforms()

    train_dataset = datasets.ImageFolder(
        Config.DATA_DIR / 'train',
        transform=data_transforms['train']
    )
    val_dataset = datasets.ImageFolder(
        Config.DATA_DIR / 'val',
        transform=data_transforms['val']
    )
    test_dataset = datasets.ImageFolder(
        Config.DATA_DIR / 'test',
        transform=data_transforms['test']
    )

    # Log dataset statistics
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    logger.info(f"Classes: {train_dataset.classes}")

    # Create balanced sampler for training
    sampler = create_balanced_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_dataset


# ==========================================
# Model Architecture
# ==========================================

def create_model(num_classes: int, device: torch.device) -> nn.Module:
    """Create ResNet18 model with custom classifier"""
    from torchvision.models import resnet18, ResNet18_Weights

    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze early layers, unfreeze later ones for fine-tuning
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False

    # Enhanced classifier head
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    return model.to(device)


# ==========================================
# Training and Evaluation
# ==========================================

def train_epoch(model, loader, criterion, optimizer, device, logger):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': loss.item(), 'acc': correct / total})

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels, all_probs


def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler, device, logger):
    """Full training loop with early stopping"""
    best_val_loss = float('inf')
    best_val_acc = 0.0
    early_stop_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(Config.NUM_EPOCHS):
        logger.info(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")

        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, logger
        )

        # Validation
        val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        # Log metrics
        logger.info(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, 'best_model.pth')
            logger.info(f"✓ Saved best model (Val Loss: {val_loss:.4f})")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= Config.PATIENCE:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Save training history
    pd.DataFrame(history).to_csv('training_history.csv', index=False)
    logger.info(f"\nBest Val Loss: {best_val_loss:.4f} | Best Val Acc: {best_val_acc:.4f}")

    return model


def evaluate_model(model, test_loader, device, class_names, logger):
    """Comprehensive model evaluation"""
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.numpy())
            y_probs.extend(probs[:, 1].cpu().numpy())

    # Classification report
    logger.info("\n" + "=" * 50)
    logger.info("Classification Report")
    logger.info("=" * 50)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    logger.info(f"\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    logger.info("\nConfusion Matrix:")
    logger.info(f"\n{cm}")

    # ROC-AUC score
    if len(class_names) == 2:
        auc = roc_auc_score(y_true, y_probs)
        logger.info(f"\nROC-AUC Score: {auc:.4f}")

    return y_true, y_pred, y_probs


def save_predictions(model, dataset_path, transform, device, class_names, output_file):
    """Generate and save predictions for all test images"""
    model.eval()
    results = []

    for cls in class_names:
        folder = dataset_path / cls
        if not folder.exists():
            continue

        for img_file in folder.glob('*'):
            if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                try:
                    img = Image.open(img_file).convert('RGB')
                    input_tensor = transform(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.softmax(output, dim=1)
                        pred_idx = torch.argmax(output, dim=1).item()
                        confidence = probs[0, pred_idx].item()

                    results.append({
                        "image_name": img_file.name,
                        "true_class": cls,
                        "predicted_class": class_names[pred_idx],
                        "confidence": confidence,
                        "correct": cls == class_names[pred_idx]
                    })
                except Exception as e:
                    logging.error(f"Error processing {img_file}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    accuracy = df['correct'].mean()
    logging.info(f"\nPredictions saved to {output_file}")
    logging.info(f"Overall accuracy: {accuracy:.4f}")

    return df


# ==========================================
# Main Execution
# ==========================================

def main():
    """Main training pipeline"""
    # Setup
    logger = setup_logging()
    set_seed(Config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data
    train_loader, val_loader, test_loader, train_dataset = load_data(logger)

    # Create model
    model = create_model(num_classes=len(train_dataset.classes), device=device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5
    )

    # Train
    logger.info("\nStarting training...")
    model = train_model(model, train_loader, val_loader, criterion,
                        optimizer, scheduler, device, logger)

    # Load best model
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"\nLoaded best model from epoch {checkpoint['epoch'] + 1}")

    # Evaluate
    logger.info("\nEvaluating on test set...")
    evaluate_model(model, test_loader, device, train_dataset.classes, logger)

    # Save predictions
    logger.info("\nGenerating predictions...")
    save_predictions(
        model,
        Config.DATA_DIR / 'test',
        get_transforms()['test'],
        device,
        train_dataset.classes,
        'predictions_detailed.csv'
    )

    logger.info("\n✓ Training complete!")


if __name__ == "__main__":
    main()