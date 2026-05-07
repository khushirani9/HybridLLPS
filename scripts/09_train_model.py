"""
09 TRAIN MODEL
======================================================================
HybridLLPS Model Training — Four-Branch Fusion Architecture
======================================================================

Trains the HybridLLPS deep learning model on the prepared feature matrices.

Architecture — four independent branches, outputs concatenated then fused:
  ESM-2 branch        (1280 → 512 → 256 → 128):  captures evolutionary/structural context
  AlphaFold branch    (24   → 64  → 32):           structural disorder information
  Physicochemical branch (47 → 64  → 32):           charge patterning, aromaticity etc.
  Dipeptide branch    (410  → 128 → 64  → 32):     local sequence grammar
  Fusion layer        (224  → 64  → 1  → Sigmoid): final LLPS probability

Training settings:
  Optimiser: Adam (lr=1e-3, weight_decay=1e-4)
  Loss:      Binary Cross-Entropy
  Batch:     64 with weighted random sampling (addresses 1:2.9 class imbalance)
  Stopping:  Early stopping on validation AUROC, patience=10 epochs

Best checkpoint saved when validation AUROC improves.

Inputs
------
data/final/train_X.npy, train_y.npy, val_X.npy, val_y.npy

Outputs
-------
models/best_model.pt  (contains: model_state, val_auc, val_ap, epoch)

Usage
-----
python 09_train_model.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server
import matplotlib.pyplot as plt
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.expanduser("~/llps_project/data/final/")
MODEL_DIR  = os.path.expanduser("~/llps_project/models/")
RESULTS_DIR= os.path.expanduser("~/llps_project/results/plots/")
os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Load data ──────────────────────────────────────────────────────────────────
print("\nLoading data...")
X_train = np.load(os.path.join(DATA_DIR, 'train_X.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'train_y.npy'))
X_val   = np.load(os.path.join(DATA_DIR, 'val_X.npy'))
y_val   = np.load(os.path.join(DATA_DIR, 'val_y.npy'))
X_test  = np.load(os.path.join(DATA_DIR, 'test_X.npy'))
y_test  = np.load(os.path.join(DATA_DIR, 'test_y.npy'))

print(f"Train: {X_train.shape}, pos={y_train.sum()}, neg={(y_train==0).sum()}")
print(f"Val  : {X_val.shape}")
print(f"Test : {X_test.shape}")

# Feature block indices — must match order in 10_combine_features.py
# [ESM: 0-1279] [AlphaFold: 1280-1303] [Physicochemical: 1304-1350] [Traditional: 1351-1760]
ESM_END  = 1280
AF_END   = 1280 + 24   # 1304
PC_END   = 1304 + 47   # 1351
TRAD_END = 1351 + 410  # 1761

# ── Convert to PyTorch tensors ─────────────────────────────────────────────────
# PyTorch works with tensors, not numpy arrays
# float32 for features, long (int64) for labels
def to_tensor(X, y):
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32))  # float for BCELoss

X_train_t, y_train_t = to_tensor(X_train, y_train)
X_val_t,   y_val_t   = to_tensor(X_val,   y_val)
X_test_t,  y_test_t  = to_tensor(X_test,  y_test)

# DataLoader batches the data for mini-batch gradient descent
# shuffle=True randomizes order each epoch (prevents learning order artifacts)
train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=64, shuffle=True
)

# ── Model Architecture ─────────────────────────────────────────────────────────
class LLPSHybridModel(nn.Module):
    """
    Multi-Input Hybrid Neural Network for LLPS prediction.

    Four parallel branches process each feature block independently.
    This prevents high-dimensional ESM features from dominating
    the lower-dimensional but equally informative structural features.

    Architecture:
    ESM branch      : 1280 → 512 → 256 → 128
    AlphaFold branch:   24 →  64 →  32
    Physicochemical :   47 →  64 →  32
    Traditional     :  410 → 128 →  64 → 32
    Fusion          : 224 (128+32+32+32) → 64 → 1
    """

    def __init__(self, dropout_rate=0.3):
        super(LLPSHybridModel, self).__init__()

        # ── ESM-2 Branch (most complex — handles 1280D input) ─────────────────
        self.esm_branch = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),   # normalize across batch
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # ── AlphaFold Branch (small — only 24 features) ───────────────────────
        self.af_branch = nn.Sequential(
            nn.Linear(24, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        # ── Physicochemical Branch ─────────────────────────────────────────────
        self.pc_branch = nn.Sequential(
            nn.Linear(47, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        # ── Traditional/Dipeptide Branch ──────────────────────────────────────
        self.trad_branch = nn.Sequential(
            nn.Linear(410, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        # ── Fusion Layer ───────────────────────────────────────────────────────
        # 128 + 32 + 32 + 32 = 224 dimensional fusion vector
        fusion_dim = 128 + 32 + 32 + 32

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 1),
            nn.Sigmoid()  # output: LLPS probability [0,1]
        )

    def forward(self, x):
        """
        Forward pass — split input into feature blocks,
        process through each branch, concatenate, predict.
        """
        # Split the 1761-feature input into 4 blocks
        x_esm  = x[:, :ESM_END]           # columns 0-1279
        x_af   = x[:, ESM_END:AF_END]     # columns 1280-1303
        x_pc   = x[:, AF_END:PC_END]      # columns 1304-1350
        x_trad = x[:, PC_END:TRAD_END]    # columns 1351-1760

        # Process each branch independently
        out_esm  = self.esm_branch(x_esm)
        out_af   = self.af_branch(x_af)
        out_pc   = self.pc_branch(x_pc)
        out_trad = self.trad_branch(x_trad)

        # Concatenate all branch outputs → 224D fusion vector
        fused = torch.cat([out_esm, out_af, out_pc, out_trad], dim=1)

        # Final prediction
        return self.fusion(fused).squeeze(1)

# ── Initialize model ───────────────────────────────────────────────────────────
model = LLPSHybridModel(dropout_rate=0.3).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel initialized: {total_params:,} parameters")

# ── Class weighting for imbalanced dataset ────────────────────────────────────
# pos_weight tells BCELoss to weight positive samples more heavily
# Formula: num_negative / num_positive = 1411/485 = 2.91
pos_weight = torch.tensor([y_train.sum() / (len(y_train) - y_train.sum())])
pos_weight = (1 / pos_weight).to(device)  # inverse because we want to upweight positives
# Actually correct formula:
n_neg = (y_train == 0).sum()
n_pos = y_train.sum()
pos_weight_val = n_neg / n_pos  # = 2.91
pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
print(f"Class weight for positive samples: {pos_weight_val:.2f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Wait — we use Sigmoid in the model, so use BCELoss not BCEWithLogitsLoss
# Let's fix: remove Sigmoid from model and use BCEWithLogitsLoss
# OR keep Sigmoid and use BCELoss — we'll keep Sigmoid for interpretability
criterion = nn.BCELoss(reduction='mean')

# Apply manual class weighting in training loop instead
# ── Optimizer and Scheduler ───────────────────────────────────────────────────
# Adam optimizer — adaptive learning rate, works well for most deep learning
# lr=1e-3 is the standard starting point
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# ReduceLROnPlateau: if val AUC doesn't improve for 'patience' epochs,
# reduce learning rate by factor 0.5. Helps fine-tune convergence.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5, verbose=True
)

# ── Training function ─────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, pos_weight_val):
    model.train()  # enable dropout and batchnorm in training mode
    total_loss = 0
    all_preds, all_labels = [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()          # clear previous gradients
        preds = model(X_batch)         # forward pass
        
        # Manual class weighting
        # Positive samples get weight pos_weight_val, negatives get 1.0
        weights = torch.where(y_batch == 1,
                              torch.tensor(pos_weight_val, device=device),
                              torch.tensor(1.0, device=device))
        loss = (criterion(preds, y_batch) * weights).mean()

        loss.backward()                # backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
        optimizer.step()               # update weights

        total_loss += loss.item()
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(y_batch.detach().cpu().numpy())

    auc = roc_auc_score(all_labels, all_preds)
    return total_loss / len(loader), auc

def evaluate(model, X_t, y_t, device):
    model.eval()  # disable dropout, use running stats for batchnorm
    with torch.no_grad():
        preds = model(X_t.to(device)).cpu().numpy()
    auc = roc_auc_score(y_t.numpy(), preds)
    ap  = average_precision_score(y_t.numpy(), preds)
    return auc, ap, preds

# ── Training loop ──────────────────────────────────────────────────────────────
EPOCHS      = 100
best_val_auc= 0.0
best_epoch  = 0
patience    = 15   # early stopping: stop if no improvement for 15 epochs
no_improve  = 0

train_losses, train_aucs = [], []
val_aucs, val_aps        = [], []

print(f"\nStarting training for {EPOCHS} epochs...")
print(f"Early stopping patience: {patience} epochs")
print("-" * 55)

for epoch in range(1, EPOCHS + 1):
    # Train
    train_loss, train_auc = train_epoch(
        model, train_loader, optimizer, criterion, device, pos_weight_val
    )

    # Validate
    val_auc, val_ap, _ = evaluate(model, X_val_t, y_val_t, device)

    # Scheduler step — reduce LR if val_auc plateaus
    scheduler.step(val_auc)

    # Track metrics
    train_losses.append(train_loss)
    train_aucs.append(train_auc)
    val_aucs.append(val_auc)
    val_aps.append(val_ap)

    # Save best model
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_epoch   = epoch
        no_improve   = 0
        torch.save({
            'epoch'     : epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_auc'   : val_auc,
            'val_ap'    : val_ap,
        }, os.path.join(MODEL_DIR, 'best_model.pt'))
    else:
        no_improve += 1

    # Print progress every 5 epochs
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Loss={train_loss:.4f} | "
              f"Train AUC={train_auc:.4f} | "
              f"Val AUC={val_auc:.4f} | "
              f"Val AP={val_ap:.4f} | "
              f"Best={best_val_auc:.4f} (ep{best_epoch})")

    # Early stopping
    if no_improve >= patience:
        print(f"\nEarly stopping at epoch {epoch} — no improvement for {patience} epochs")
        break

print(f"\nTraining complete!")
print(f"Best Val AUC: {best_val_auc:.4f} at epoch {best_epoch}")

# ── Load best model and evaluate on test set ───────────────────────────────────
print("\nLoading best model for test evaluation...")
checkpoint = torch.load(os.path.join(MODEL_DIR, 'best_model.pt'), weights_only=False)
model.load_state_dict(checkpoint['model_state'])

test_auc, test_ap, test_preds = evaluate(model, X_test_t, y_test_t, device)
print(f"\nTEST RESULTS:")
print(f"  AUC-ROC            : {test_auc:.4f}")
print(f"  Average Precision  : {test_ap:.4f}")

# Threshold at 0.5 for binary predictions
test_binary = (test_preds > 0.5).astype(int)
from sklearn.metrics import classification_report, confusion_matrix
print(f"\nClassification Report (threshold=0.5):")
print(classification_report(y_test, test_binary,
                             target_names=['Non-LLPS', 'LLPS']))
print(f"Confusion Matrix:")
print(confusion_matrix(y_test, test_binary))

# ── Plot training curves ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(train_losses, label='Train Loss', color='blue')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(train_aucs, label='Train AUC', color='blue')
axes[1].plot(val_aucs,   label='Val AUC',   color='orange')
axes[1].axvline(best_epoch-1, color='red', linestyle='--',
                label=f'Best epoch {best_epoch}')
axes[1].axhline(0.88, color='green', linestyle='--',
                label='Target AUC=0.88')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('AUC-ROC')
axes[1].set_title('AUC-ROC During Training')
axes[1].legend()
axes[1].grid(True)
axes[1].set_ylim([0.5, 1.0])

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, 'training_curves.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\nTraining curves saved → {plot_path}")

# Save predictions for later analysis
np.save(os.path.join(MODEL_DIR, 'test_predictions.npy'), test_preds)
np.save(os.path.join(MODEL_DIR, 'test_labels.npy'), y_test)
print("Test predictions saved.")
print("\nReady for SHAP analysis and mutation scoring!")
