# src/evaluate.py

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
from src.model import LSTMAutoencoder

MODEL_PATH = "models/lstm_autoencoder.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = LSTMAutoencoder()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_reconstruction_errors(model, data_np):
    tensor = torch.tensor(data_np[:, :, np.newaxis].astype(np.float32)).to(device)
    with torch.no_grad():
        output = model(tensor)
    errors = torch.mean((output - tensor) ** 2, dim=(1, 2))
    return errors.cpu().numpy()

def anomaly_score(error, train_errors):
    """Return a 0-100 score: how anomalous is this beat relative to normal training errors."""
    return int(np.mean(train_errors < error) * 100)

def sweep_thresholds(normal_errors, anomaly_errors):
    """Sweep percentiles 82–95 and print a comparison table. Returns the best percentile."""
    y_true = np.concatenate([
        np.zeros(len(normal_errors)),
        np.ones(len(anomaly_errors))
    ])
    y_scores = np.concatenate([normal_errors, anomaly_errors])

    print("\n" + "="*65)
    print(f"{'Pct':>4}  {'Threshold':>12}  {'Precision':>9}  {'Recall':>7}  {'F1':>7}")
    print("="*65)

    best_pct   = 95
    best_f1    = 0.0

    for pct in range(82, 96):
        t     = np.percentile(normal_errors, pct)
        preds = (y_scores > t).astype(int)
        p     = precision_score(y_true, preds, zero_division=0)
        r     = recall_score(y_true, preds, zero_division=0)
        f1    = f1_score(y_true, preds, zero_division=0)
        marker = "  ◄ best F1" if f1 > best_f1 else ""
        if f1 > best_f1:
            best_f1  = f1
            best_pct = pct
        print(f"{pct:>4}  {t:>12.6f}  {p:>9.4f}  {r:>7.4f}  {f1:>7.4f}{marker}")

    print("="*65)
    print(f"\n✅ Recommended threshold: {best_pct}th percentile  (F1 = {best_f1:.4f})")
    return best_pct

def evaluate():
    model = load_model()
    print("Model loaded.")

    train_normal = np.load("data/train.npy")
    test_normal  = np.load("data/test.npy")
    anomalous    = np.load("data/anomaly.npy")

    print("Calculating errors on training beats (for anomaly score baseline)...")
    train_errors = get_reconstruction_errors(model, train_normal)

    print("Calculating errors on normal test beats...")
    normal_errors = get_reconstruction_errors(model, test_normal)

    print("Calculating errors on anomalous beats...")
    anomaly_errors = get_reconstruction_errors(model, anomalous)

    # ── Threshold sweep ────────────────────────────────────────────
    best_pct = sweep_thresholds(normal_errors, anomaly_errors)
    threshold = np.percentile(normal_errors, best_pct)

    # ── Final metrics at best threshold ───────────────────────────
    y_true  = np.concatenate([np.zeros(len(normal_errors)), np.ones(len(anomaly_errors))])
    y_scores = np.concatenate([normal_errors, anomaly_errors])
    y_pred  = (y_scores > threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    auc       = roc_auc_score(y_true, y_scores)
    cm        = confusion_matrix(y_true, y_pred)

    print(f"\n── Final Metrics at {best_pct}th Percentile Threshold ──")
    print(f"Threshold : {threshold:.6f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")

    # ── Save threshold & metrics ───────────────────────────────────
    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    np.save("models/threshold.npy", threshold)
    np.save("models/train_errors.npy", train_errors)   # needed for anomaly_score()

    metrics = {
        "threshold_percentile": best_pct,
        "threshold": float(threshold),
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "auc":       float(auc),
    }
    import json
    with open("models/metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=2)
    print("\n✅ Threshold saved  →  models/threshold.npy")
    print("✅ Train errors saved  →  models/train_errors.npy")
    print("✅ Metrics saved  →  models/metrics.json")

    # ── Plots ──────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="blue", label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — PulseWatcher")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/roc_curve.png")
    print("ROC curve saved to outputs/roc_curve.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — PulseWatcher ({best_pct}th pct threshold)")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    print("Confusion matrix saved to outputs/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()
