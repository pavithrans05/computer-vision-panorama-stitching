import os
import json
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import SmoothL1Loss, CrossEntropyLoss

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

from dataset import GCPDataset
from transforms import (
    get_train_transforms,
    get_val_transforms
)
from model import GCPModel


# ==========================================================
# CONFIG
# ==========================================================

TRAIN_DIR = "../train_dataset"
JSON_PATH = os.path.join(TRAIN_DIR, "gcp_marks.json")

BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5
IMG_SIZE = 512  # Used for denormalizing coordinates in PCK calculation

# Multi-task Loss Weights
REG_WEIGHT = 5.0
CLS_WEIGHT = 1.0

SAVE_DIR = "../weights"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ==========================================================
# LABEL MAPPING
# ==========================================================

shape_to_idx = {
    "Cross": 0,
    "Square": 1,
    "L-Shape": 2
}

idx_to_shape = {
    v: k for k, v in shape_to_idx.items()
}


# ==========================================================
# LOAD JSON
# ==========================================================

with open(JSON_PATH, "r") as f:
    data = json.load(f)

records = []

for path, info in data.items():

    if "verified_shape" not in info:
        continue

    records.append({
        "path": path,
        "x": info["mark"]["x"],
        "y": info["mark"]["y"],
        "shape": info["verified_shape"]
    })

df = pd.DataFrame(records)

print("Total usable samples:", len(df))


# ==========================================================
# TRAIN VAL SPLIT
# ==========================================================

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["shape"],
    random_state=42
)

print("Train:", len(train_df))
print("Validation:", len(val_df))


# ==========================================================
# DATASETS
# ==========================================================

train_dataset = GCPDataset(
    train_df,
    TRAIN_DIR,
    transform=get_train_transforms()
)

val_dataset = GCPDataset(
    val_df,
    TRAIN_DIR,
    transform=get_val_transforms()
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# ==========================================================
# MODEL
# ==========================================================

model = GCPModel().to(device)


# ==========================================================
# CLASS WEIGHTS
# ==========================================================

labels = train_df["shape"].map(shape_to_idx).values

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1, 2]),
    y=labels
)

weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(device)

print("Class weights:", weights)


# ==========================================================
# LOSS FUNCTIONS
# ==========================================================

# Regression loss for (x, y) coordinates
reg_loss_fn = SmoothL1Loss()

# Classification loss with balanced class weights
cls_loss_fn = CrossEntropyLoss(
    weight=weights
)


# ==========================================================
# OPTIMIZER
# ==========================================================

optimizer = AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ==========================================================
# TRAIN FUNCTION
# ==========================================================

def train_one_epoch():

    model.train()
    running_loss = 0

    for batch in train_loader:

        images = batch["image"].to(device)
        coords_gt = batch["coords"].to(device)
        labels_gt = batch["label"].to(device)

        optimizer.zero_grad()

        coords_pred, logits = model(images)

        # Calculate weighted losses
        reg_loss = reg_loss_fn(coords_pred, coords_gt)
        cls_loss = cls_loss_fn(logits, labels_gt)

        loss = (REG_WEIGHT * reg_loss) + (CLS_WEIGHT * cls_loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


# ==========================================================
# VALIDATION FUNCTION
# ==========================================================

@torch.no_grad()
def validate():

    model.eval()
    running_loss = 0
    
    # Tracking for metrics
    all_preds = []
    all_labels = []
    all_distances = []

    for batch in val_loader:

        images = batch["image"].to(device)
        coords_gt = batch["coords"].to(device)
        labels_gt = batch["label"].to(device)

        coords_pred, logits = model(images)

        # Validation Loss (Weighted)
        reg_loss = reg_loss_fn(coords_pred, coords_gt)
        cls_loss = cls_loss_fn(logits, labels_gt)
        loss = (REG_WEIGHT * reg_loss) + (CLS_WEIGHT * cls_loss)
        running_loss += loss.item()

        # Classification Metrics Gathering
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_gt.cpu().numpy())
        
        # Regression Metrics Gathering (PCK)
        pred_px = coords_pred * IMG_SIZE
        gt_px = coords_gt * IMG_SIZE
        
        distances = torch.norm(pred_px - gt_px, dim=1)
        all_distances.extend(distances.cpu().numpy())

    # Calculate final epoch metrics
    avg_loss = running_loss / len(val_loader)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_distances = np.array(all_distances)
    
    # Classification Metrics
    accuracy = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds, average="macro")
    
    # PCK Metrics
    pck10 = (all_distances <= 10).mean()
    pck25 = (all_distances <= 25).mean()
    pck50 = (all_distances <= 50).mean()

    return avg_loss, accuracy, f1, pck10, pck25, pck50


# ==========================================================
# TRAINING LOOP
# ==========================================================

history = {
    "train_loss": [],
    "val_loss": [],
    "acc": [],
    "f1": [],
    "pck10": [],
    "pck25": [],
    "pck50": []
}

best_val_loss = float("inf")
best_f1 = 0.0
patience_counter = 0

print("\nStarting Training...\n" + "="*50)

for epoch in range(EPOCHS):

    train_loss = train_one_epoch()
    val_loss, acc, f1, pck10, pck25, pck50 = validate()

    # Update history
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["acc"].append(acc)
    history["f1"].append(f1)
    history["pck10"].append(pck10)
    history["pck25"].append(pck25)
    history["pck50"].append(pck50)

    scheduler.step(val_loss)

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Acc: {acc:.4f} | "
        f"F1: {f1:.4f} | "
        f"PCK@10: {pck10:.4f} | "
        f"PCK@25: {pck25:.4f} | "
        f"PCK@50: {pck50:.4f}"
    )

    # Save Best Loss Model
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "f1": f1
        },
        os.path.join(
            SAVE_DIR,
            "best_loss_model.pth"
        ))

        print(" -> Saved best loss model")

        patience_counter = 0

    else:
        patience_counter += 1

    # Save Best F1 Model
    if f1 > best_f1:

        best_f1 = f1

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "f1": f1
        },
        os.path.join(
            SAVE_DIR,
            "best_f1_model.pth"
        ))

        print(" -> Saved best F1 model")

    # Early Stopping Check (based on val_loss)
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping triggered. Validation loss has not improved for {PATIENCE} epochs.")
        break

# Save Training History to CSV
pd.DataFrame(history).to_csv(
    os.path.join(SAVE_DIR, "training_history.csv"),
    index=False
)

print("\n" + "="*50)
print("Training completed!")
print(f"Best Validation Loss: {best_val_loss:.4f}")
print(f"Best F1 Score: {best_f1:.4f}")
print(f"History saved to {os.path.join(SAVE_DIR, 'training_history.csv')}")