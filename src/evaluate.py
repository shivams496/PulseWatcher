# src/evaluate.py

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os
from src.model import LSTMAutoencoder

MODEL_PATH = "models/lstm_autoencoder.pt"
THRESHOLD_PERCENTILE = 95

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

def evaluate():
    model = load_model()
    print("Model loaded.")

    test_normal = np.load("data/test.npy")
    anomalous = np.load("data/anomaly.npy")

    print("Calculating errors on normal beats...")
    normal_errors = get_reconstruction_errors(model, test_normal)

    print("Calculating errors on anomalous beats...")
    anomaly_errors = get_reconstruction_errors(model, anomalous)

    threshold = np.percentile(normal_errors, THRESHOLD_PERCENTILE)
    print(f"\nThreshold (95th percentile): {threshold:.6f}")

    normal_flagged = np.sum(normal_errors > threshold)
    anomaly_flagged = np.sum(anomaly_errors > threshold)

    tp = anomaly_flagged
    fp = normal_flagged
    fn = len(anomaly_errors) - anomaly_flagged
    tn = len(normal_errors) - normal_flagged

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"\nPrecision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # ✅ NEW: Confusion Matrix
    y_true = np.concatenate([np.zeros(len(normal_errors)), np.ones(len(anomaly_errors))])
    y_pred = np.concatenate([
        (normal_errors > threshold).astype(int),
        (anomaly_errors > threshold).astype(int)
    ])
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")

    # ✅ NEW: ROC-AUC
    y_scores = np.concatenate([normal_errors, anomaly_errors])
    auc = roc_auc_score(y_true, y_scores)
    print(f"\nROC-AUC Score: {auc:.4f}")

    # ✅ NEW: Save plots
    os.makedirs("outputs", exist_ok=True)

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
    plt.title("Confusion Matrix — PulseWatcher")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    print("Confusion matrix saved to outputs/confusion_matrix.png")

    np.save("models/threshold.npy", threshold)
    print(f"\nThreshold saved to models/threshold.npy")

if __name__ == "__main__":
    evaluate()