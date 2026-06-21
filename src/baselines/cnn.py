import torch
import torch.nn as nn


class CNNModel(nn.Module):
    """1D-CNN baseline for well-log lithology classification.

    Three Conv1d blocks (Conv1d -> BatchNorm1d -> ReLU -> Dropout) followed
    by global average pooling and a two-layer fully-connected head.

    Args:
        d_input: Number of input features per time step.
        num_classes: Number of lithology classes.
        conv1_out: Output channels for the first convolution.
        conv2_out: Output channels for the second convolution.
        conv3_out: Output channels for the third convolution.
        kernel_size: Kernel size for all Conv1d layers.
        fc_hidden: Hidden size of the fully-connected head.
        dropout: Dropout probability applied after each convolution and in the FC head.
    """

    def __init__(
        self,
        d_input,
        num_classes=5,
        conv1_out=64,
        conv2_out=128,
        conv3_out=256,
        kernel_size=3,
        fc_hidden=128,
        dropout=0.3,
    ):
        super().__init__()

        padding = kernel_size // 2  # same-padding so length is preserved

        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv1d(d_input, conv1_out, kernel_size, padding=padding),
            nn.BatchNorm1d(conv1_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Block 2
            nn.Conv1d(conv1_out, conv2_out, kernel_size, padding=padding),
            nn.BatchNorm1d(conv2_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Block 3
            nn.Conv1d(conv2_out, conv3_out, kernel_size, padding=padding),
            nn.BatchNorm1d(conv3_out),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(conv3_out, fc_hidden),
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
        # Conv1d expects (batch, channels, length)
        x = x.transpose(1, 2)
        x = self.conv_block(x)
        x = self.global_pool(x)  # (batch, conv3_out, 1)
        x = x.squeeze(-1)  # (batch, conv3_out)
        x = self.fc(x)
        return x
