import torch
import torch.nn as nn


class RNNModel(nn.Module):
    """Vanilla RNN baseline for well-log lithology classification.

    A multi-layer vanilla RNN (tanh activation) followed by a two-layer
    fully-connected head that operates on the last hidden state.

    Args:
        d_input: Number of input features per time step.
        num_classes: Number of lithology classes.
        rnn_hidden: Hidden size of the RNN.
        rnn_layers: Number of stacked RNN layers.
        dropout: Dropout probability (applied between RNN layers and in the FC head).
        fc_hidden: Hidden size of the fully-connected head.
    """

    def __init__(
        self,
        d_input,
        num_classes=5,
        rnn_hidden=128,
        rnn_layers=2,
        dropout=0.3,
        fc_hidden=128,
    ):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=d_input,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden, fc_hidden),
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
        # RNN output: (batch, seq_len, rnn_hidden)
        rnn_out, h_n = self.rnn(x)

        # Take the last hidden state from the top layer
        last_hidden = rnn_out[:, -1, :]  # (batch, rnn_hidden)

        logits = self.fc(last_hidden)
        return logits
