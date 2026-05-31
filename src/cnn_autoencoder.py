# src/cnn_autoencoder.py  --  1D CNN Autoencoder for ECG anomaly detection
# Same input/output shape as LSTMAutoencoder: (batch, 187, 1)

import torch
import torch.nn as nn


class CNNAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),   # (B, 16, 187)
            nn.ReLU(),
            nn.MaxPool1d(2),                               # (B, 16, 93)
            nn.Conv1d(16, 32, kernel_size=5, padding=2),  # (B, 32, 93)
            nn.ReLU(),
            nn.MaxPool1d(2),                               # (B, 32, 46)
            nn.Conv1d(32, 64, kernel_size=3, padding=1),  # (B, 64, 46)
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1),  # (B, 32, 46)
            nn.ReLU(),
            nn.Upsample(size=93),                                   # (B, 32, 93)
            nn.ConvTranspose1d(32, 16, kernel_size=5, padding=2),  # (B, 16, 93)
            nn.ReLU(),
            nn.Upsample(size=187),                                  # (B, 16, 187)
            nn.ConvTranspose1d(16, 1, kernel_size=7, padding=3),   # (B, 1, 187)
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, 187, 1) -- same as LSTM input
        x = x.permute(0, 2, 1)   # -> (batch, 1, 187)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.permute(0, 2, 1)   # -> (batch, 187, 1)
        return x
