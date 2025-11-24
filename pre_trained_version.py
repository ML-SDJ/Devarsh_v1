#
# """
# Enhanced Chest X-Ray Pneumonia Classifier
# Features: Better organization, logging, metrics tracking, and error handling
# """
# import os
# import logging
# from pathlib import Path
# from typing import Dict, Tuple
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, models, transforms
# from torch.utils.data import DataLoader, WeightedRandomSampler
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# import pandas as pd
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
#
#
# # ==========================================
# # Configuration and Setup
# # ==========================================
#
# class Config:
#     """Centralized configuration"""
#     DATA_DIR = Path('data/chest_xray')
#     BATCH_SIZE = 32
#     NUM_EPOCHS = 25
#     LEARNING_RATE = 1e-3
#     PATIENCE = 4
#     WEIGHT_DECAY = 1e-4
#     IMG_SIZE = 224
#     NUM_WORKERS = 4
#     SEED = 42
#
#
# def setup_logging():
#     """Configure logging"""
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler('training.log'),
#             logging.StreamHandler()
#         ]
#     )
#     return logging.getLogger(__name__)
#
#
# def set_seed(seed: int):
#     """Set random seeds for reproducibility"""
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#
#
# # ==========================================
# # Data Preparation
# # ==========================================
#
# def get_transforms() -> Dict[str, transforms.Compose]:
#     """Define data augmentation and normalization transforms"""
#     return {
#         'train': transforms.Compose([
#             transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomRotation(15),
#             transforms.ColorJitter(brightness=0.2, contrast=0.2),
#             transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         'val': transforms.Compose([
#             transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         'test': transforms.Compose([
#             transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
#     }
#
#
# def create_balanced_sampler(dataset):
#     """Create weighted sampler to handle class imbalance"""
#     targets = [s[1] for s in dataset.samples]
#     class_counts = np.bincount(targets)
#     class_weights = 1.0 / class_counts
#     sample_weights = [class_weights[t] for t in targets]
#     return WeightedRandomSampler(sample_weights, len(sample_weights))
#
#
# def load_data(logger) -> Tuple[DataLoader, DataLoader, DataLoader, datasets.ImageFolder]:
#     """Load and prepare datasets with balanced sampling"""
#     data_transforms = get_transforms()
#
#     train_dataset = datasets.ImageFolder(
#         Config.DATA_DIR / 'train',
#         transform=data_transforms['train']
#     )
#     val_dataset = datasets.ImageFolder(
#         Config.DATA_DIR / 'val',
#         transform=data_transforms['val']
#     )
#     test_dataset = datasets.ImageFolder(
#         Config.DATA_DIR / 'test',
#         transform=data_transforms['test']
#     )
#
#     # Log dataset statistics
#     logger.info(f"Train samples: {len(train_dataset)}")
#     logger.info(f"Val samples: {len(val_dataset)}")
#     logger.info(f"Test samples: {len(test_dataset)}")
#     logger.info(f"Classes: {train_dataset.classes}")
#
#     # Create balanced sampler for training
#     sampler = create_balanced_sampler(train_dataset)
#
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=Config.BATCH_SIZE,
#         sampler=sampler,
#         num_workers=Config.NUM_WORKERS,
#         pin_memory=True
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=Config.BATCH_SIZE,
#         shuffle=False,
#         num_workers=Config.NUM_WORKERS,
#         pin_memory=True
#     )
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=Config.BATCH_SIZE,
#         shuffle=False,
#         num_workers=Config.NUM_WORKERS,
#         pin_memory=True
#     )
#
#     return train_loader, val_loader, test_loader, train_dataset
#
#
# # ==========================================
# # Model Architecture
# # ==========================================
#
# def create_model(num_classes: int, device: torch.device) -> nn.Module:
#     """Create ResNet18 model with custom classifier"""
#     from torchvision.models import resnet18, ResNet18_Weights
#
#     model = resnet18(weights=ResNet18_Weights.DEFAULT)
#
#     # Freeze early layers, unfreeze later ones for fine-tuning
#     for name, param in model.named_parameters():
#         if 'layer4' not in name and 'fc' not in name:
#             param.requires_grad = False
#
#     # Enhanced classifier head
#     num_features = model.fc.in_features
#     model.fc = nn.Sequential(
#         nn.Linear(num_features, 512),
#         nn.ReLU(),
#         nn.BatchNorm1d(512),
#         nn.Dropout(0.5),
#         nn.Linear(512, 256),
#         nn.ReLU(),
#         nn.BatchNorm1d(256),
#         nn.Dropout(0.4),
#         nn.Linear(256, num_classes)
#     )
#
#     return model.to(device)
#
#
# # ==========================================
# # Training and Evaluation
# # ==========================================
#
# def train_epoch(model, loader, criterion, optimizer, device, logger):
#     """Train for one epoch"""
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#
#     pbar = tqdm(loader, desc='Training')
#     for images, labels in pbar:
#         images, labels = images.to(device), labels.to(device)
#
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#
#         # Gradient clipping for stability
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
#
#         running_loss += loss.item() * images.size(0)
#         _, preds = torch.max(outputs, 1)
#         correct += (preds == labels).sum().item()
#         total += labels.size(0)
#
#         pbar.set_postfix({'loss': loss.item(), 'acc': correct / total})
#
#     return running_loss / total, correct / total
#
#
# def validate(model, loader, criterion, device):
#     """Validate the model"""
#     model.eval()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     all_preds = []
#     all_labels = []
#     all_probs = []
#
#     with torch.no_grad():
#         for images, labels in tqdm(loader, desc='Validating'):
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#
#             running_loss += loss.item() * images.size(0)
#             probs = torch.softmax(outputs, dim=1)
#             _, preds = torch.max(outputs, 1)
#
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
#
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#             all_probs.extend(probs[:, 1].cpu().numpy())
#
#     return running_loss / total, correct / total, all_preds, all_labels, all_probs
#
#
# def train_model(model, train_loader, val_loader, criterion, optimizer,
#                 scheduler, device, logger):
#     """Full training loop with early stopping"""
#     best_val_loss = float('inf')
#     best_val_acc = 0.0
#     early_stop_counter = 0
#     history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
#
#     for epoch in range(Config.NUM_EPOCHS):
#         logger.info(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")
#
#         # Training
#         train_loss, train_acc = train_epoch(
#             model, train_loader, criterion, optimizer, device, logger
#         )
#
#         # Validation
#         val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, device)
#
#         # Update scheduler
#         scheduler.step(val_loss)
#
#         # Log metrics
#         logger.info(
#             f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
#             f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
#         )
#
#         # Store history
#         history['train_loss'].append(train_loss)
#         history['train_acc'].append(train_acc)
#         history['val_loss'].append(val_loss)
#         history['val_acc'].append(val_acc)
#
#         # Save best model
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_val_acc = val_acc
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'val_loss': val_loss,
#                 'val_acc': val_acc,
#             }, 'best_model.pth')
#             logger.info(f"✓ Saved best model (Val Loss: {val_loss:.4f})")
#             early_stop_counter = 0
#         else:
#             early_stop_counter += 1
#             if early_stop_counter >= Config.PATIENCE:
#                 logger.info(f"Early stopping triggered after {epoch + 1} epochs")
#                 break
#
#     # Save training history
#     pd.DataFrame(history).to_csv('training_history.csv', index=False)
#     logger.info(f"\nBest Val Loss: {best_val_loss:.4f} | Best Val Acc: {best_val_acc:.4f}")
#
#     return model
#
#
# def evaluate_model(model, test_loader, device, class_names, logger):
#     """Comprehensive model evaluation"""
#     model.eval()
#     y_true = []
#     y_pred = []
#     y_probs = []
#
#     with torch.no_grad():
#         for images, labels in tqdm(test_loader, desc='Testing'):
#             images = images.to(device)
#             outputs = model(images)
#             probs = torch.softmax(outputs, dim=1)
#             preds = torch.argmax(outputs, dim=1)
#
#             y_pred.extend(preds.cpu().numpy())
#             y_true.extend(labels.numpy())
#             y_probs.extend(probs[:, 1].cpu().numpy())
#
#     # Classification report
#     logger.info("\n" + "=" * 50)
#     logger.info("Classification Report")
#     logger.info("=" * 50)
#     report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
#     logger.info(f"\n{report}")
#
#     # Confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
#     logger.info("\nConfusion Matrix:")
#     logger.info(f"\n{cm}")
#
#     # ROC-AUC score
#     if len(class_names) == 2:
#         auc = roc_auc_score(y_true, y_probs)
#         logger.info(f"\nROC-AUC Score: {auc:.4f}")
#
#     return y_true, y_pred, y_probs
#
#
# def save_predictions(model, dataset_path, transform, device, class_names, output_file):
#     """Generate and save predictions for all test images"""
#     model.eval()
#     results = []
#
#     for cls in class_names:
#         folder = dataset_path / cls
#         if not folder.exists():
#             continue
#
#         for img_file in folder.glob('*'):
#             if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
#                 try:
#                     img = Image.open(img_file).convert('RGB')
#                     input_tensor = transform(img).unsqueeze(0).to(device)
#
#                     with torch.no_grad():
#                         output = model(input_tensor)
#                         probs = torch.softmax(output, dim=1)
#                         pred_idx = torch.argmax(output, dim=1).item()
#                         confidence = probs[0, pred_idx].item()
#
#                     results.append({
#                         "image_name": img_file.name,
#                         "true_class": cls,
#                         "predicted_class": class_names[pred_idx],
#                         "confidence": confidence,
#                         "correct": cls == class_names[pred_idx]
#                     })
#                 except Exception as e:
#                     logging.error(f"Error processing {img_file}: {e}")
#
#     df = pd.DataFrame(results)
#     df.to_csv(output_file, index=False)
#
#     accuracy = df['correct'].mean()
#     logging.info(f"\nPredictions saved to {output_file}")
#     logging.info(f"Overall accuracy: {accuracy:.4f}")
#
#     return df
#
#
# # ==========================================
# # Main Execution
# # ==========================================
#
# def main():
#     """Main training pipeline"""
#     # Setup
#     logger = setup_logging()
#     set_seed(Config.SEED)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")
#
#     # Load data
#     train_loader, val_loader, test_loader, train_dataset = load_data(logger)
#
#     # Create model
#     model = create_model(num_classes=len(train_dataset.classes), device=device)
#     logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
#     logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
#
#     # Loss and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.AdamW(
#         filter(lambda p: p.requires_grad, model.parameters()),
#         lr=Config.LEARNING_RATE,
#         weight_decay=Config.WEIGHT_DECAY
#     )
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', patience=2, factor=0.5
#     )
#
#     # Train
#     logger.info("\nStarting training...")
#     model = train_model(model, train_loader, val_loader, criterion,
#                         optimizer, scheduler, device, logger)
#
#     # Load best model
#     checkpoint = torch.load('best_model.pth', map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     logger.info(f"\nLoaded best model from epoch {checkpoint['epoch'] + 1}")
#
#     # Evaluate
#     logger.info("\nEvaluating on test set...")
#     evaluate_model(model, test_loader, device, train_dataset.classes, logger)
#
#     # Save predictions
#     logger.info("\nGenerating predictions...")
#     save_predictions(
#         model,
#         Config.DATA_DIR / 'test',
#         get_transforms()['test'],
#         device,
#         train_dataset.classes,
#         'predictions_detailed.csv'
#     )
#
#     logger.info("\n✓ Training complete!")
#
#
# if __name__ == "__main__":
#     main()

"""
Fast 3-Class Chest X-Ray Classifier (Normal vs Bacteria vs Virus)
Optimizations: Mixed Precision (AMP), OneCycleLR, Cached Loading
"""
import os
import logging
from pathlib import Path
from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

# ==========================================
# 1. Configuration
# ==========================================
class Config:
    DATA_DIR = Path('data/chest_xray')
    # Increased batch size because AMP reduces memory usage
    BATCH_SIZE = 64
    NUM_EPOCHS = 15 # Reduced epochs because OneCycleLR converges faster
    LEARNING_RATE = 1e-3
    IMG_SIZE = 224
    NUM_WORKERS = 4
    SEED = 42
    USE_AMP = True  # AUTOMATIC MIXED PRECISION (The Speed Booster)

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    return logging.getLogger(__name__)

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Optimization for fixed input sizes
    torch.backends.cudnn.benchmark = True

# ==========================================
# 2. Smart 3-Class Dataset
# ==========================================
class ThreeClassDataset(Dataset):
    """
    Parses filenames to create 3 classes:
    0: NORMAL
    1: BACTERIA
    2: VIRUS
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.classes = ['NORMAL', 'BACTERIA', 'VIRUS']

        # Scan directory
        for item in self.root_dir.glob('**/*'):
            if item.suffix.lower() not in ['.jpeg', '.jpg', '.png']:
                continue

            path_str = str(item).upper()
            filename = item.name.upper()
            parent = item.parent.name.upper()

            # Logic to determine class
            if 'NORMAL' in parent:
                label = 0
            elif 'PNEUMONIA' in parent:
                if 'BACTERIA' in filename:
                    label = 1
                elif 'VIRUS' in filename:
                    label = 2
                else:
                    # Fallback if filename doesn't specify (rare)
                    label = 1
            else:
                continue

            self.samples.append((str(item), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE)), label

# ==========================================
# 3. Data Loading & Transforms
# ==========================================
def get_transforms():
    return {
        'train': transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

def get_loaders(logger):
    tf = get_transforms()

    # Initialize Datasets
    train_ds = ThreeClassDataset(Config.DATA_DIR / 'train', tf['train'])
    val_ds = ThreeClassDataset(Config.DATA_DIR / 'val', tf['val'])
    test_ds = ThreeClassDataset(Config.DATA_DIR / 'test', tf['val'])

    logger.info(f"Dataset Size: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    # Balanced Sampler (Crucial for 3 classes)
    targets = [s[1] for s in train_ds.samples]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, sampler=sampler,
                              num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
                             num_workers=Config.NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.classes

# ==========================================
# 4. Model (ResNet18 Optimized)
# ==========================================
def create_model(num_classes, device):
    # Use ResNet18 for speed
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Unfreeze Layer 3 and 4 to learn texture differences bw Virus/Bacteria
    for name, param in model.named_parameters():
        if "layer3" in name or "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model.to(device)

# ==========================================
# 5. Training Loop (with AMP)
# ==========================================
def train_engine(model, train_loader, val_loader, device, logger):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Helps with noisy labels
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=Config.LEARNING_RATE, weight_decay=1e-2)

    # OneCycleLR is faster than ReduceLROnPlateau
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader), epochs=Config.NUM_EPOCHS
    )

    # Scaler for Mixed Precision
    scaler = torch.amp.GradScaler('cuda') if Config.USE_AMP else None

    best_acc = 0.0

    for epoch in range(Config.NUM_EPOCHS):
        # --- TRAINING ---
        model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS}", leave=False)
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            # FAST PATH: Mixed Precision
            if Config.USE_AMP:
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            scheduler.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(acc=train_correct/total)

        # --- VALIDATION ---
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        logger.info(f"Epoch {epoch+1}: Train Acc: {train_correct/total:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_3class_model.pth")

    return model

# ==========================================
# 6. Evaluation & Reporting
# ==========================================
def final_evaluate(model, loader, classes, device, logger):
    logger.info("\nRunning Final Evaluation...")
    model.load_state_dict(torch.load("best_3class_model.pth", map_location=device))
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Testing"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\n" + "=" * 50)
    print("FINAL 3-CLASS REPORT")
    print("=" * 50)

    # FIX: We explicitly pass the labels [0, 1, 2] so it doesn't crash if a class is missing
    print(classification_report(
        y_true,
        y_pred,
        target_names=classes,
        labels=list(range(len(classes))),  # <--- THIS LINE STOPS THE CRASH
        digits=4,
        zero_division=0
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
# ==========================================
# 7. Main
# ==========================================
if __name__ == "__main__":
    logger = setup_logging()
    set_seed(Config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Mixed Precision: {Config.USE_AMP}")

    # Load Data
    train_loader, val_loader, test_loader, class_names = get_loaders(logger)
    print(f"Classes Detected: {class_names}")

    # Create Model
    model = create_model(len(class_names), device)

    # Train
    model = train_engine(model, train_loader, val_loader, device, logger)

    # Test
    final_evaluate(model, test_loader, class_names, device, logger)