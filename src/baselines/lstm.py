import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """LSTM baseline for well-log lithology classification.

    A unidirectional multi-layer LSTM followed by a two-layer fully-connected
    head that operates on the last hidden state.

    Args:
        d_input: Number of input features per time step.
        num_classes: Number of lithology classes.
        lstm_hidden: Hidden size of the LSTM.
        lstm_layers: Number of stacked LSTM layers.
        dropout: Dropout probability (applied between LSTM layers and in the FC head).
        fc_hidden: Hidden size of the fully-connected head.
    """

    def __init__(
        self,
        d_input,
        num_classes=5,
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.3,
        fc_hidden=128,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=d_input,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_input).

        Returns:
            Logits of shape (batch, num_classes).
        """
        # LSTM output: (batch, seq_len, lstm_hidden)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take the last hidden state from the top layer
        last_hidden = lstm_out[:, -1, :]  # (batch, lstm_hidden)

        logits = self.fc(last_hidden)
        return logits
