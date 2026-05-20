import torch 
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for ECG anomaly detection.
    Encoder compresses the heartbeat into a small representation.
    Decoder reconstructs it back. High reconstruction error = anomaly.
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        
        # Encoder: reads the heartbeat sequence, compresses it
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Decoder: takes the compressed form, reconstructs the sequences
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        #Final layer: maps hidden_size back to 1 value per timestep
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # x shape: (batch, 187, 1)

        #Encode
        _, (hidden, cell) = self.encoder(x)

        #Repeat the hidden state 187 times - one for each timestep 
        #hidden[-1] shape: (batch, hidden_size)
        repeated = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        #repeated shape: (batch, 187, hidden_size)

        #Decode
        decoded, _ = self.decoder(repeated)
        #decoded shape: (batch, 187, hidden_size)

        #Map to output 
        output = self.output_layer(decoded)
        #output shape: (batch, 187, 1)
        return output