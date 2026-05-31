# src/benchmark.py  --  Train CNN autoencoder and compare with LSTM
# Run from project root: python -m src.benchmark
# Output: models/cnn_autoencoder.pt + models/benchmark.json

import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

sys.path.append(os.path.abspath("."))
from src.model import LSTMAutoencoder
from src.cnn_autoencoder import CNNAutoencoder


# ── Config ────────────────────────────────────────────────────
EPOCHS      = 10
BATCH_SIZE  = 64
LR          = 1e-3
THRESHOLD_PCT = 90   # same percentile used for LSTM
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ── Load data ─────────────────────────────────────────────────
print("Loading data...")
train_data    = np.load("data/train.npy")
normal_test   = np.load("data/test.npy")
anomaly_test  = np.load("data/anomaly.npy")

# Tensors -- shape (N, 187, 1)
def to_tensor(arr):
    return torch.tensor(arr[:, :, np.newaxis].astype(np.float32))

X_train   = to_tensor(train_data)
X_normal  = to_tensor(normal_test)
X_anomaly = to_tensor(anomaly_test)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train),
    batch_size=BATCH_SIZE, shuffle=True
)
print(f"Train: {len(X_train)}  |  Normal test: {len(X_normal)}  |  Anomaly test: {len(X_anomaly)}")


# ── Train a model ─────────────────────────────────────────────
def train_model(model, name):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f"\nTraining {name}...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for (batch,) in train_loader:
            batch = batch.to(DEVICE)
            out   = model(batch)
            loss  = criterion(out, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{EPOCHS}  loss={avg:.6f}")

    return model


# ── Evaluate a model ──────────────────────────────────────────
def evaluate_model(model, name):
    model.eval()
    criterion = nn.MSELoss(reduction="none")

    def get_errors(X):
        with torch.no_grad():
            out = model(X.to(DEVICE))
            err = criterion(out, X.to(DEVICE))
            return err.mean(dim=(1, 2)).cpu().numpy()

    # Compute threshold on training data
    with torch.no_grad():
        train_errors = []
        for (batch,) in train_loader:
            out = model(batch.to(DEVICE))
            err = nn.MSELoss(reduction="none")(out, batch.to(DEVICE))
            train_errors.append(err.mean(dim=(1, 2)).cpu().numpy())
    train_errors = np.concatenate(train_errors)
    threshold = float(np.percentile(train_errors, THRESHOLD_PCT))

    n_err = get_errors(X_normal)
    a_err = get_errors(X_anomaly)

    y_true   = np.concatenate([np.zeros(len(n_err)), np.ones(len(a_err))])
    y_scores = np.concatenate([n_err, a_err])
    y_pred   = (y_scores > threshold).astype(int)

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall    = float(recall_score(y_true, y_pred, zero_division=0))
    f1        = float(f1_score(y_true, y_pred, zero_division=0))
    auc       = float(roc_auc_score(y_true, y_scores))

    print(f"\n{name} Results (threshold={threshold:.6f} @ {THRESHOLD_PCT}th pct):")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1:        {f1*100:.2f}%")
    print(f"  ROC-AUC:   {auc:.4f}")

    return {
        "name": name,
        "threshold": threshold,
        "threshold_pct": THRESHOLD_PCT,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


# ── Load existing LSTM metrics ────────────────────────────────
print("\nLoading LSTM results from models/metrics.json...")
with open("models/metrics.json") as f:
    lstm_metrics_raw = json.load(f)

lstm_results = {
    "name": "LSTM Autoencoder",
    "threshold_pct": lstm_metrics_raw.get("threshold_percentile", 90),
    "precision": lstm_metrics_raw["precision"],
    "recall":    lstm_metrics_raw["recall"],
    "f1":        lstm_metrics_raw["f1"],
    "auc":       lstm_metrics_raw["auc"],
}
print(f"  LSTM -- P:{lstm_results['precision']*100:.2f}%  R:{lstm_results['recall']*100:.2f}%  F1:{lstm_results['f1']*100:.2f}%  AUC:{lstm_results['auc']:.4f}")


# ── Train and evaluate CNN ────────────────────────────────────
cnn_model = train_model(CNNAutoencoder(), "CNN Autoencoder")
torch.save(cnn_model.state_dict(), "models/cnn_autoencoder.pt")
print("Saved models/cnn_autoencoder.pt")

cnn_results = evaluate_model(cnn_model, "CNN Autoencoder")


# ── Print comparison table ────────────────────────────────────
print("\n" + "="*60)
print(f"{'Model':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
print("="*60)
for r in [lstm_results, cnn_results]:
    print(f"{r['name']:<22} {r['precision']*100:>9.2f}% {r['recall']*100:>9.2f}% {r['f1']*100:>9.2f}% {r['auc']:>10.4f}")
print("="*60)


# ── Save benchmark results ────────────────────────────────────
benchmark = {
    "lstm": lstm_results,
    "cnn":  cnn_results,
}
with open("models/benchmark.json", "w") as f:
    json.dump(benchmark, f, indent=2)
print("\nSaved models/benchmark.json")
print("Done! Run: streamlit run dashboard/app.py to see the comparison table.")
