import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.model import LSTMAutoencoder

# --- Config ---
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MODEL_PATH = "models/lstm_autoencoder.pt"

def load_data():
    """Load preprocessed training data and reshape for LSTM."""
    train = np.load("data/train.npy")   # shape: (59816, 187)
    test = np.load("data/test.npy")     # shape: (14955, 187)

    # Add the feature dimension: (samples, timesteps, 1)
    train = train[:, :, np.newaxis].astype(np.float32)
    test = test[:, :, np.newaxis].astype(np.float32)

    return train, test

def train_model():
    # Load data
    train_data, test_data = load_data()

    # Convert to PyTorch tensors
    train_tensor = torch.tensor(train_data)
    test_tensor = torch.tensor(test_data)

    # Create DataLoaders
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Initialize model, loss, optimizer
    model = LSTMAutoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training on {len(train_data)} beats for {EPOCHS} epochs...")
    print("-" * 50)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for (batch,) in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.6f}")

    # Save the trained model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    return model

if __name__ == "__main__":
    train_model()