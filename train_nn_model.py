import copy
import random
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

# ================= CONFIG =================
MAX_ROBOTS = 15
INPUT_SIZE = MAX_ROBOTS * 6
OUTPUT_SIZE = MAX_ROBOTS

EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 12
MIN_DELTA = 1e-4
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= REPRODUCIBILITY =================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ================= LOAD DATA =================
print("Loading dataset...")
data = pd.read_csv("real_intersection_dataset.csv")

X = data.iloc[:, :INPUT_SIZE].values.astype(np.float32)
y = data.iloc[:, INPUT_SIZE:INPUT_SIZE + OUTPUT_SIZE].values.astype(np.float32)

# ================= SPLIT FIRST =================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# ================= NORMALIZATION =================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
joblib.dump(scaler, "scaler.pkl")

# ================= TENSORS =================
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False
)

# ================= MODEL =================
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, OUTPUT_SIZE)
        )

    def forward(self, x):
        return self.net(x)

model = NN().to(DEVICE)

# ================= LOSS =================
y_train_np = y_train.numpy()
pos_counts = y_train_np.sum(axis=0)
neg_counts = len(y_train_np) - pos_counts

pos_weight = np.clip(neg_counts / (pos_counts + 1e-6), 1.0, 10.0)
pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=4
)

# ================= METRICS =================
def evaluate_at_threshold(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    preds = (probs >= threshold).astype(np.float32)
    true = targets.detach().cpu().numpy()

    acc = (preds == true).mean()

    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        true, preds, average="micro", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        true, preds, average="macro", zero_division=0
    )

    return {
        "acc": acc,
        "p_micro": p_micro,
        "r_micro": r_micro,
        "f1_micro": f1_micro,
        "p_macro": p_macro,
        "r_macro": r_macro,
        "f1_macro": f1_macro,
    }

def find_best_threshold(logits, targets):
    best_t = 0.5
    best_f1 = -1.0

    for t in np.arange(0.30, 0.71, 0.02):
        metrics = evaluate_at_threshold(logits, targets, threshold=float(t))
        if metrics["f1_micro"] > best_f1:
            best_f1 = metrics["f1_micro"]
            best_t = float(t)

    return best_t, best_f1

# ================= TRAIN =================
train_losses = []
val_losses = []
val_f1_micro = []
val_f1_macro = []
val_recall_micro = []

best_val_loss = float("inf")
best_state = None
epochs_without_improve = 0

X_val_device = X_val.to(DEVICE)
y_val_device = y_val.to(DEVICE)

print("Training...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * len(xb)
        total_samples += len(xb)

    train_loss = total_loss / total_samples

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_device)
        val_loss = criterion(val_logits, y_val_device).item()
        metrics = evaluate_at_threshold(val_logits, y_val_device, threshold=0.5)

    scheduler.step(val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_f1_micro.append(metrics["f1_micro"])
    val_f1_macro.append(metrics["f1_macro"])
    val_recall_micro.append(metrics["r_micro"])

    print(
        f"Epoch {epoch+1:03d} | "
        f"TrainLoss: {train_loss:.4f} | "
        f"ValLoss: {val_loss:.4f} | "
        f"Acc: {metrics['acc']:.4f} | "
        f"F1_micro: {metrics['f1_micro']:.4f} | "
        f"F1_macro: {metrics['f1_macro']:.4f} | "
        f"Recall_micro: {metrics['r_micro']:.4f}"
    )

    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improve = 0
    else:
        epochs_without_improve += 1

    if epochs_without_improve >= PATIENCE:
        print("\nEarly stopping triggered.")
        break

# ================= RESTORE BEST =================
if best_state is not None:
    model.load_state_dict(best_state)

# ================= THRESHOLD TUNING =================
model.eval()
with torch.no_grad():
    final_val_logits = model(X_val_device)
    best_threshold, best_threshold_f1 = find_best_threshold(final_val_logits, y_val_device)
    final_metrics = evaluate_at_threshold(final_val_logits, y_val_device, threshold=best_threshold)

print("\nBest validation threshold:", round(best_threshold, 3))
print("Final validation metrics:")
for k, v in final_metrics.items():
    print(f"{k}: {v:.4f}")

# ================= SAVE =================
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_size": INPUT_SIZE,
        "output_size": OUTPUT_SIZE,
        "threshold": best_threshold
    },
    "nn_controller_best.pth"
)
print("\n BEST MODEL SAVED")

# ================= PLOTS =================
plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(val_f1_micro, label="Val F1 Micro")
plt.plot(val_f1_macro, label="Val F1 Macro")
plt.title("F1 Curves")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(val_recall_micro, label="Val Recall Micro")
plt.title("Recall Curve")
plt.legend()

plt.tight_layout()
plt.show()
