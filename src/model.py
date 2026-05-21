# src/model.py

import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMAutoencoder, self).__init__()

        # ✅ NEW: Dropout added for regularization
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        _, (hidden, cell) = self.encoder(x)
        repeated = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(repeated)
        output = self.output_layer(decoded)
        return output