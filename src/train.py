import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.model import LSTMAutoencoder
import matplotlib.pyplot as plt
import os

# --- Config ---
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MODEL_PATH = "models/lstm_autoencoder.pt"

# ✅ NEW: GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_data():
    """Load preprocessed training data and reshape for LSTM."""
    train = np.load("data/train.npy")   # shape: (59816, 187)
    test = np.load("data/test.npy")     # shape: (14955, 187)

    # ✅ NEW: Shuffle to avoid data leakage
    np.random.shuffle(train)
    np.random.shuffle(test)

    # Add the feature dimension: (samples, timesteps, 1)
    train = train[:, :, np.newaxis].astype(np.float32)
    test = test[:, :, np.newaxis].astype(np.float32)

    return train, test

def train_model():
    train_data, val_data = load_data()

    train_tensor = torch.tensor(train_data)
    val_tensor = torch.tensor(val_data)

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # ✅ NEW: Validation DataLoader
    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # ✅ NEW: Move model to GPU if available
    model = LSTMAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ✅ NEW: Scheduler - reduces LR when val loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=True
    )

    print(f"Training on {len(train_data)} beats for {EPOCHS} epochs...")
    print(f"Validating on {len(val_data)} beats")
    print("-" * 50)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):

        # --- Training ---
        model.train()
        total_train_loss = 0

        for (batch,) in train_loader:
            batch = batch.to(device)  # ✅ NEW: Move to GPU
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # ✅ NEW: Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # ✅ NEW: Step scheduler
        scheduler.step(avg_val_loss)

        # ✅ NEW: Save only the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  → Best model saved (val loss: {best_val_loss:.6f})")

    # ✅ NEW: Plot and save training curve
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Val Loss", color="orange", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/training_curve.png")
    print("\nTraining curve saved to outputs/training_curve.png")
    print(f"Best validation loss: {best_val_loss:.6f}")

    return model

if __name__ == "__main__":
    train_model()