"""TCN-BiLSTM-Transformer model for lithology identification.

This module implements a hybrid deep learning architecture that combines:
- Temporal Convolutional Network (TCN) for local feature extraction
- Bidirectional LSTM for sequential dependency modeling
- Transformer encoder for global self-attention
- Multi-scale prediction head with learned gating

The model is designed for well-log based lithology classification tasks.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation attention module.

    Adaptively recalibrates channel-wise feature responses by modelling
    inter-channel dependencies through a bottleneck architecture.

    Args:
        channels: Number of input channels.
        reduction: Channel reduction ratio for the bottleneck. Default: 4.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, length).

        Returns:
            Channel-attention-weighted tensor of the same shape.
        """
        b, c, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class TemporalBlock(nn.Module):
    """TCN temporal block with optional SE attention.

    Implements a dilated causal convolution block with two convolution layers,
    batch normalisation, ReLU activation, dropout, and a residual connection.
    Optionally applies squeeze-and-excitation attention after the convolutions.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the 1D convolution kernel. Default: 3.
        dilation: Dilation factor for the convolutions. Default: 1.
        dropout: Dropout probability. Default: 0.2.
        use_se: Whether to apply SE attention. Default: True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout)

        self.se = SEBlock(out_channels) if use_se else None

        # Residual downsample when channel dimensions differ
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor of shape (batch, in_channels, length).

        Returns:
            Output tensor of shape (batch, out_channels, length).
        """
        out = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        out = self.drop2(self.relu2(self.bn2(self.conv2(out))))
        if self.se is not None:
            out = self.se(out)
        res = x if self.downsample is None else self.downsample(x)
        if out.size(2) != res.size(2):
            out = out[:, :, :res.size(2)]
        return out + res


class TCN(nn.Module):
    """Temporal Convolutional Network.

    Stacks temporal blocks with exponentially growing dilation factors to
    capture long-range temporal dependencies with a large receptive field.

    Args:
        in_channels: Number of input features per time step.
        channels: List of channel sizes for each temporal block.
        kernel_size: Convolution kernel size. Default: 3.
        dropout: Dropout probability. Default: 0.2.
        use_se: Whether to apply SE attention in each block. Default: True.
    """

    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        layers = []
        for i, ch in enumerate(channels):
            in_ch = in_channels if i == 0 else channels[i - 1]
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=ch,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                    use_se=use_se,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, length, in_channels).

        Returns:
            Output tensor of shape (batch, length, channels[-1]).
        """
        x = x.transpose(1, 2)  # (B, C, L) for Conv1d
        y = self.network(x)
        return y.transpose(1, 2)  # (B, L, C) for subsequent layers


class MultiScaleHead(nn.Module):
    """Multi-scale feature extraction with learned gating.

    Combines a single-layer linear projection and a two-layer MLP through a
    learned softmax gate, allowing the model to adaptively weight features
    from different scales.

    Args:
        in_dim: Input feature dimension.
        out_dim: Output feature dimension.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.scale1 = nn.Linear(in_dim, out_dim)
        self.scale2 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(out_dim * 2, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gated multi-scale combination.

        Args:
            x: Input tensor of shape (batch, length, in_dim).

        Returns:
            Output tensor of shape (batch, length, out_dim).
        """
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        combined = torch.cat([s1, s2], dim=-1)
        weights = self.gate(combined)
        return weights[..., 0:1] * s1 + weights[..., 1:2] * s2


class Model_TCN_BiLSTM_Transformer(nn.Module):
    """Hybrid TCN-BiLSTM-Transformer model for lithology identification.

    Architecture flow:
        Input -> TCN (local features) -> BiLSTM (sequential dependencies)
        -> Positional Encoding -> Transformer Encoder (global attention)
        -> Multi-Scale Head -> LayerNorm -> FC (classification at last position)

    The model takes the last time-step output for classification, producing
    a (batch, num_classes) tensor of logits.

    Args:
        d_input: Number of input features per time step.
        num_classes: Number of lithology classes. Default: 5.
        window_size: Input sequence length. Default: 41.
        tcn_channels: Channel sizes for each TCN block. Default: [32, 64].
        lstm_hidden: BiLSTM hidden size (per direction). Default: 128.
        lstm_layers: Number of BiLSTM layers. Default: 1.
        nhead: Number of attention heads in the Transformer. Default: 4.
        trans_fwd: Transformer feed-forward dimension. Default: 128.
        trans_layers: Number of Transformer encoder layers. Default: 2.
        dropout: Dropout probability. Default: 0.2.
        use_se: Whether to use SE attention in TCN blocks. Default: True.
    """

    def __init__(
        self,
        d_input: int,
        num_classes: int = 5,
        window_size: int = 41,
        tcn_channels: List[int] = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        nhead: int = 4,
        trans_fwd: int = 128,
        trans_layers: int = 2,
        dropout: float = 0.2,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        if tcn_channels is None:
            tcn_channels = [32, 64]

        tcn_out = tcn_channels[-1]
        lstm_out = lstm_hidden * 2  # bidirectional

        # --- TCN backbone ---
        self.tcn = TCN(d_input, tcn_channels, dropout=dropout, use_se=use_se)

        # --- BiLSTM ---
        self.lstm = nn.LSTM(
            input_size=tcn_out,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # --- Positional encoding ---
        self.pos = nn.Parameter(torch.randn(1, window_size, lstm_out) * 0.02)

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_out,
            nhead=nhead,
            dim_feedforward=trans_fwd,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=trans_layers
        )

        # --- Multi-scale prediction head ---
        self.multi_scale_head = MultiScaleHead(lstm_out, lstm_out)
        self.layer_norm = nn.LayerNorm(lstm_out)

        # --- Classification layer ---
        self.fc = nn.Linear(lstm_out, num_classes)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[List[torch.Tensor]]]]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, length, d_input).
            return_attention: If True, also return attention weights from
                each Transformer encoder layer. Default: False.

        Returns:
            If return_attention is False:
                logits: Tensor of shape (batch, num_classes).
            If return_attention is True:
                (logits, attentions): logits of shape (batch, num_classes)
                and a list of attention weight tensors.
        """
        # Local feature extraction
        x = self.tcn(x)  # (B, L, tcn_out)

        # Sequential dependency modelling
        x, _ = self.lstm(x)  # (B, L, lstm_out)

        # Positional encoding
        x = x + self.pos

        # Global self-attention
        if return_attention:
            attentions: Optional[List[torch.Tensor]] = []
            for layer in self.transformer.layers:
                x_norm = layer.norm1(x)
                attn_out, attn_w = layer.self_attn(
                    x_norm, x_norm, x_norm, need_weights=True,
                    average_attn_weights=True,
                )
                x = x + layer.dropout1(attn_out)
                x = x + layer.dropout2(
                    layer.linear2(
                        layer.activation(layer.linear1(layer.norm2(x)))
                    )
                )
                attentions.append(attn_w)
        else:
            attentions = None
            x = self.transformer(x)

        # Multi-scale head and normalisation
        x = self.multi_scale_head(x)  # (B, L, lstm_out)
        x = self.layer_norm(x)

        # Classification at last position
        logits = self.fc(x[:, -1])  # (B, num_classes)

        if return_attention:
            return logits, attentions
        return logits
