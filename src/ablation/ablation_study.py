"""Structural ablation study for TCN-BiLSTM-Transformer lithology identification.

Systematically removes or replaces architectural components to evaluate their
individual contribution to overall model performance.  Each variant is
evaluated using leave-one-well-out cross-validation so that results are
directly comparable with the full model baseline.
"""

import os
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from src.core.model import (
    Model_TCN_BiLSTM_Transformer,
    SEBlock,
    TemporalBlock,
    TCN,
    MultiScaleHead,
    LearnablePositionalEncoding,
)


# ---------------------------------------------------------------------------
# Ablation variant model definitions
# ---------------------------------------------------------------------------


class TemporalBlockNoResidual(nn.Module):
    """TCN temporal block *without* residual connections.

    Identical to :class:`TemporalBlock` except that the skip connection from
    input to output is removed.  This isolates the contribution of residual
    learning within the TCN backbone.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.  Default: 3.
        dilation: Dilation factor.  Default: 1.
        dropout: Dropout probability.  Default: 0.2.
        use_se: Whether to apply SE attention.  Default: True.
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

        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                       dilation=dilation, padding=padding)
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)
        self.se1 = SEBlock(out_channels) if use_se else nn.Identity()

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size,
                       dilation=dilation, padding=padding)
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout)
        self.se2 = SEBlock(out_channels) if use_se else nn.Identity()

        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass without residual connection."""
        out = self.conv1(x)[:, :, :-self.padding].contiguous()
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.se1(out)

        out = self.conv2(out)[:, :, :-self.padding].contiguous()
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.se2(out)

        return out


class TCNNoResidual(nn.Module):
    """TCN built from :class:`TemporalBlockNoResidual` blocks.

    Args:
        in_channels: Number of input features per time step.
        channels: List of channel sizes for each temporal block.
        kernel_size: Convolution kernel size.  Default: 3.
        dropout: Dropout probability.  Default: 0.2.
        use_se: Whether to apply SE attention.  Default: True.
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
                TemporalBlockNoResidual(
                    in_channels=in_ch,
                    out_channels=ch,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    dropout=dropout,
                    use_se=use_se,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, C, L)
        x = self.network(x)
        x = x.transpose(1, 2)  # (B, L, C)
        return x


class AblationModel(nn.Module):
    """Configurable model that supports all ablation variants.

    Each boolean flag controls whether a specific component is included.
    When a component is disabled, a minimal replacement is used so that
    tensor shapes remain consistent throughout the forward pass.

    Args:
        d_input: Number of input features per time step.
        num_classes: Number of lithology classes.
        window_size: Input sequence length.
        tcn_channels: Channel sizes for each TCN block.
        lstm_hidden: LSTM hidden size (per direction).
        lstm_layers: Number of LSTM layers.
        nhead: Number of Transformer attention heads.
        trans_fwd: Transformer feed-forward dimension.
        trans_layers: Number of Transformer encoder layers.
        dropout: Dropout probability.
        use_se: Whether to use SE attention in TCN blocks.
        use_tcn_residual: Whether TCN blocks include residual connections.
        bidirectional: Whether LSTM is bidirectional.
        use_transformer: Whether to include the Transformer encoder.
        use_multiscale: Whether to use MultiScaleHead (else simple linear).
        use_pos_enc: Whether to add learnable positional encoding.
    """

    def __init__(
        self,
        d_input: int,
        num_classes: int = 5,
        window_size: int = 41,
        tcn_channels: Optional[List[int]] = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        nhead: int = 4,
        trans_fwd: int = 128,
        trans_layers: int = 2,
        dropout: float = 0.2,
        use_se: bool = True,
        use_tcn_residual: bool = True,
        bidirectional: bool = True,
        use_transformer: bool = True,
        use_multiscale: bool = True,
        use_pos_enc: bool = True,
    ) -> None:
        super().__init__()
        if tcn_channels is None:
            tcn_channels = [32, 64]

        tcn_out = tcn_channels[-1]
        lstm_out = lstm_hidden * 2 if bidirectional else lstm_hidden

        # --- TCN backbone ---
        if use_tcn_residual:
            self.tcn = TCN(d_input, tcn_channels, dropout=dropout, use_se=use_se)
        else:
            self.tcn = TCNNoResidual(d_input, tcn_channels, dropout=dropout, use_se=use_se)

        # --- LSTM ---
        self.bilstm = nn.LSTM(
            input_size=tcn_out,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # --- Positional encoding ---
        self.use_pos_enc = use_pos_enc
        if use_pos_enc:
            self.pos_enc = LearnablePositionalEncoding(lstm_out, max_len=window_size)

        # --- Transformer encoder ---
        self.use_transformer = use_transformer
        if use_transformer:
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

        # --- Multi-scale head ---
        self.use_multiscale = use_multiscale
        if use_multiscale:
            self.head = MultiScaleHead(lstm_out, lstm_out)
        else:
            self.head = nn.Linear(lstm_out, lstm_out)

        self.norm = nn.LayerNorm(lstm_out)
        self.fc = nn.Linear(lstm_out, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise weights with Xavier uniform for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, length, d_input).
            return_attention: If True, return Transformer attention weights.

        Returns:
            Tuple of (logits, attentions).
        """
        x = self.tcn(x)
        x, _ = self.bilstm(x)

        if self.use_pos_enc:
            x = self.pos_enc(x)

        attentions = None
        if self.use_transformer:
            if return_attention:
                attentions = []
                for layer in self.transformer.layers:
                    x_norm = layer.norm1(x)
                    attn_out, attn_w = layer.self_attn(
                        x_norm, x_norm, x_norm, need_weights=True
                    )
                    x = x + layer.dropout1(attn_out)
                    x = x + layer.dropout2(
                        layer.linear2(
                            layer.activation(layer.linear1(layer.norm2(x)))
                        )
                    )
                    attentions.append(attn_w)
            else:
                x = self.transformer(x)

        x = self.head(x)
        x = self.norm(x)
        logits = self.fc(x)

        return logits, attentions


# ---------------------------------------------------------------------------
# Variant name → AblationModel keyword mapping
# ---------------------------------------------------------------------------

_VARIANT_KWARGS = {
    "full": {},
    "no_tcn_residual": {"use_tcn_residual": False},
    "no_se": {"use_se": False},
    "no_bidirectional": {"bidirectional": False},
    "no_transformer": {"use_transformer": False},
    "no_multiscale": {"use_multiscale": False},
    "no_pos_enc": {"use_pos_enc": False},
}


class StructuralAblationStudy:
    """Structural ablation study for TCN-BiLSTM-Transformer.

    Systematically removes or replaces components to evaluate their contribution:
    - Remove TCN residual connections
    - Remove SE attention from TCN
    - Replace BiLSTM with unidirectional LSTM
    - Remove Transformer encoder
    - Remove MultiScale head
    - Remove positional encoding
    """

    def __init__(self, d_input, num_classes=5, window_size=41, device='cuda'):
        self.d_input = d_input
        self.num_classes = num_classes
        self.window_size = window_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def build_variant(self, variant_name, base_params):
        """Build a model variant for ablation.

        Variants:
            'full': Complete model (baseline)
            'no_tcn_residual': TCN without residual connections
            'no_se': TCN without SE attention
            'no_bidirectional': Unidirectional LSTM instead of BiLSTM
            'no_transformer': Remove Transformer encoder
            'no_multiscale': Replace MultiScaleHead with simple linear
            'no_pos_enc': Remove learnable positional encoding

        Args:
            variant_name: One of the above strings.
            base_params: Dict with tcn_channels, lstm_hidden, lstm_layers,
                nhead, trans_fwd, trans_layers, dropout.

        Returns:
            nn.Module model on the configured device.
        """
        if variant_name not in _VARIANT_KWARGS:
            raise ValueError(
                f"Unknown variant '{variant_name}'. "
                f"Available: {list(_VARIANT_KWARGS.keys())}"
            )

        # Merge base hyperparameters with variant-specific overrides
        model_kwargs = dict(
            d_input=self.d_input,
            num_classes=self.num_classes,
            window_size=self.window_size,
            tcn_channels=base_params.get("tcn_channels", [32, 64]),
            lstm_hidden=base_params.get("lstm_hidden", 128),
            lstm_layers=base_params.get("lstm_layers", 1),
            nhead=base_params.get("nhead", 4),
            trans_fwd=base_params.get("trans_fwd", 128),
            trans_layers=base_params.get("trans_layers", 2),
            dropout=base_params.get("dropout", 0.2),
        )
        model_kwargs.update(_VARIANT_KWARGS[variant_name])

        model = AblationModel(**model_kwargs)
        return model.to(self.device)

    def run_single_fold(
        self,
        model,
        X_train,
        Y_train,
        X_val,
        Y_val,
        criterion,
        num_epochs=30,
        batch_size=512,
        lr=0.001,
        patience=8,
    ):
        """Train and evaluate a single ablation variant on one fold.

        Args:
            model: An :class:`AblationModel` instance.
            X_train: Training features of shape (n_train, window_size, n_features).
            Y_train: Training labels of shape (n_train,).
            X_val: Validation features of shape (n_val, window_size, n_features).
            Y_val: Validation labels of shape (n_val,).
            criterion: Loss function.
            num_epochs: Maximum training epochs.  Default: 30.
            batch_size: Mini-batch size.  Default: 512.
            lr: Learning rate.  Default: 0.001.
            patience: Early-stopping patience (epochs).  Default: 8.

        Returns:
            Dict with accuracy, macro_f1, weighted_f1, balanced_acc,
            per_class_f1.
        """
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(Y_train, dtype=torch.long),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                   drop_last=False)

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            # --- Training ---
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits, _ = model(xb)
                # Predict at centre position
                centre = logits.size(1) // 2
                loss = criterion(logits[:, centre, :], yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

            # --- Validation ---
            model.eval()
            with torch.no_grad():
                X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
                logits_val, _ = model(X_val_t)
                centre = logits_val.size(1) // 2
                val_logits = logits_val[:, centre, :]
                Y_val_t = torch.tensor(Y_val, dtype=torch.long, device=self.device)
                val_loss = criterion(val_logits, Y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(self.device)

        # --- Final evaluation ---
        model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
            logits_val, _ = model(X_val_t)
            centre = logits_val.size(1) // 2
            val_logits = logits_val[:, centre, :]
            preds = val_logits.argmax(dim=-1).cpu().numpy()

        accuracy = float(accuracy_score(Y_val, preds))
        macro_f1 = float(f1_score(Y_val, preds, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(Y_val, preds, average="weighted", zero_division=0))
        balanced_acc = float(balanced_accuracy_score(Y_val, preds))

        per_class_f1_vals = f1_score(Y_val, preds, average=None, zero_division=0)
        per_class_f1 = {int(c): float(v) for c, v in enumerate(per_class_f1_vals)}

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "balanced_acc": balanced_acc,
            "per_class_f1": per_class_f1,
        }

    def run_full_ablation(self, wells, base_params, output_dir, n_folds=8):
        """Run full ablation study with leave-one-well-out cross-validation.

        Args:
            wells: List of (X, Y, depth) tuples, one per well.
            base_params: Base hyperparameters dict.
            output_dir: Directory to save results.
            n_folds: Number of CV folds (uses first *n_folds* wells).

        Returns:
            Dict mapping variant_name -> list of fold results.
        """
        os.makedirs(output_dir, exist_ok=True)
        variant_names = list(_VARIANT_KWARGS.keys())
        all_results: Dict[str, List[dict]] = {v: [] for v in variant_names}

        n_wells = min(n_folds, len(wells))
        fold_wells = wells[:n_wells]

        for variant_name in variant_names:
            print(f"\n{'=' * 60}")
            print(f"  Ablation variant: {variant_name}")
            print(f"{'=' * 60}")

            for fold_idx in range(n_wells):
                # Leave-one-well-out split
                val_well = fold_wells[fold_idx]
                X_val, Y_val, _ = val_well

                train_parts = [
                    fold_wells[j] for j in range(n_wells) if j != fold_idx
                ]
                if not train_parts:
                    print(f"  [SKIP] Fold {fold_idx}: no training wells")
                    continue

                X_train = np.concatenate([w[0] for w in train_parts], axis=0)
                Y_train = np.concatenate([w[1] for w in train_parts], axis=0)

                # Build fresh model for this fold
                model = self.build_variant(variant_name, base_params)

                # Compute class-balanced loss weights
                class_counts = np.bincount(Y_train, minlength=self.num_classes)
                samples_per_class = class_counts.tolist()
                from src.core.loss import ClassBalancedFocalLoss
                criterion = ClassBalancedFocalLoss(
                    samples_per_class=samples_per_class,
                    num_classes=self.num_classes,
                )

                print(f"  Fold {fold_idx + 1}/{n_wells} — "
                      f"train: {len(X_train)}, val: {len(X_val)}")

                fold_result = self.run_single_fold(
                    model, X_train, Y_train, X_val, Y_val,
                    criterion,
                    num_epochs=30,
                    batch_size=512,
                    lr=0.001,
                    patience=8,
                )
                fold_result["fold"] = fold_idx
                all_results[variant_name].append(fold_result)

                print(f"    accuracy={fold_result['accuracy']:.4f}  "
                      f"macro_f1={fold_result['macro_f1']:.4f}  "
                      f"weighted_f1={fold_result['weighted_f1']:.4f}")

        # Save results
        results_path = os.path.join(output_dir, "ablation_results.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAblation results saved to {results_path}")

        # Print summary table
        self._print_summary(all_results)

        return all_results

    @staticmethod
    def _print_summary(all_results: Dict[str, List[dict]]) -> None:
        """Print a summary table of ablation results."""
        header = f"{'Variant':<20} {'Accuracy':>10} {'Macro F1':>10} {'Wtd F1':>10} {'Bal Acc':>10}"
        print(f"\n{'=' * len(header)}")
        print(header)
        print(f"{'=' * len(header)}")

        for variant, folds in all_results.items():
            if not folds:
                continue
            avg_acc = np.mean([f["accuracy"] for f in folds])
            avg_mf1 = np.mean([f["macro_f1"] for f in folds])
            avg_wf1 = np.mean([f["weighted_f1"] for f in folds])
            avg_bacc = np.mean([f["balanced_acc"] for f in folds])
            print(f"{variant:<20} {avg_acc:>10.4f} {avg_mf1:>10.4f} "
                  f"{avg_wf1:>10.4f} {avg_bacc:>10.4f}")

        print(f"{'=' * len(header)}")
