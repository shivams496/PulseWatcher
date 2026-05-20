# src/evaluate.py

import torch
import numpy as np
from src.model import LSTMAutoencoder

MODEL_PATH = "models/lstm_autoencoder.pt"
THRESHOLD_PERCENTILE = 95

def load_model():
    model = LSTMAutoencoder()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model

def get_reconstruction_errors(model, data_np):
    """Pass beats through model, return reconstruction error per beat."""
    tensor = torch.tensor(data_np[:, :, np.newaxis].astype(np.float32))
    with torch.no_grad():
        output = model(tensor)
    errors = torch.mean((output - tensor) ** 2, dim=(1, 2))
    return errors.numpy()

def evaluate():
    model = load_model()
    print("Model loaded.")

    # Load data
    test_normal = np.load("data/test.npy")
    anomalous = np.load("data/anomaly.npy")

    # Get reconstruction errors
    print("Calculating errors on normal beats...")
    normal_errors = get_reconstruction_errors(model, test_normal)

    print("Calculating errors on anomalous beats...")
    anomaly_errors = get_reconstruction_errors(model, anomalous)

    # Set threshold at 95th percentile of normal errors
    threshold = np.percentile(normal_errors, THRESHOLD_PERCENTILE)
    print(f"\nThreshold (95th percentile): {threshold:.6f}")

    # Detect anomalies
    normal_flagged = np.sum(normal_errors > threshold)
    anomaly_flagged = np.sum(anomaly_errors > threshold)

    print(f"\nNormal beats flagged as anomaly: {normal_flagged}/{len(normal_errors)}")
    print(f"Anomalous beats correctly flagged: {anomaly_flagged}/{len(anomaly_errors)}")

    # Metrics
    tp = anomaly_flagged
    fp = normal_flagged
    fn = len(anomaly_errors) - anomaly_flagged

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Save threshold for dashboard
    np.save("models/threshold.npy", threshold)
    print(f"\nThreshold saved to models/threshold.npy")

if __name__ == "__main__":
    evaluate()