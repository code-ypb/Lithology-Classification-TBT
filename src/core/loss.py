"""Loss functions for lithology identification.

Provides class-balanced focal loss and standard focal loss implementations
for handling class imbalance commonly encountered in well-log lithology
classification tasks.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassBalancedFocalLoss(nn.Module):
    """Class-Balanced Focal Loss with effective number weighting.

    Combines class-balanced weighting (Cui et al., 2019) with focal loss
    (Lin et al., 2017) to handle class imbalance in lithology identification.

    The effective number of samples for class i is defined as:
        E_i = (1 - beta^n_i) / (1 - beta)
    where n_i is the number of samples in class i. The class weight is then:
        w_i = 1 / E_i, normalised so that sum(w_i) = num_classes.

    Reference:
        - Cui et al., "Class-Balanced Loss Based on Effective Number of
          Samples", CVPR 2019
        - Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017

    Args:
        samples_per_class: List or tensor with the number of samples per class.
        num_classes: Total number of classes. Default: 5.
        beta: Effective number coefficient in (0, 1). Higher values give
            more weight to rare classes. Default: 0.9999.
        gamma: Focal loss focusing parameter. Higher values down-weight
            easy examples more aggressively. Default: 2.0.
        label_smoothing: Label smoothing factor in [0, 1). Default: 0.1.
    """

    def __init__(
        self,
        samples_per_class: List[int],
        num_classes: int = 5,
        beta: float = 0.9999,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.label_smoothing = label_smoothing

        # Compute effective number weights
        samples = torch.tensor(samples_per_class, dtype=torch.float32)
        effective_num = 1.0 - beta ** samples
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes

        # Register as a buffer so it moves with the model but is not a parameter
        self.register_buffer("alpha", weights)

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute the class-balanced focal loss.

        Args:
            inputs: Logits of shape (N, C) where C is the number of classes.
            targets: Ground-truth class indices of shape (N,).

        Returns:
            Scalar loss value.
        """
        # Softmax probabilities
        probs = F.softmax(inputs, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)

        # Class-balanced alpha weights
        alpha_t = self.alpha.to(inputs.device).gather(0, targets)  # (N,)

        # Focal modulation
        focal_weight = alpha_t * (1.0 - pt) ** self.gamma

        if self.label_smoothing > 0:
            # One-hot with label smoothing
            one_hot = torch.zeros_like(inputs)
            one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
            smooth = one_hot * (1.0 - self.label_smoothing) + (
                self.label_smoothing / self.num_classes
            )
            log_probs = F.log_softmax(inputs, dim=-1)
            loss = -(smooth * log_probs).sum(dim=-1)  # (N,)
        else:
            loss = F.cross_entropy(inputs, targets, reduction="none")  # (N,)

        loss = (focal_weight * loss).mean()
        return loss


class FocalLoss(nn.Module):
    """Standard Focal Loss for baseline models.

    Implements focal loss (Lin et al., 2017) without class-balanced weighting.
    Useful as a baseline or when class frequencies are roughly balanced.

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017

    Args:
        alpha: Optional per-class weight tensor of shape (C,). If None,
            all classes are weighted equally. Default: None.
        gamma: Focal loss focusing parameter. Default: 2.0.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute the focal loss.

        Args:
            inputs: Logits of shape (N, C) where C is the number of classes.
            targets: Ground-truth class indices of shape (N,).

        Returns:
            Scalar loss value.
        """
        probs = F.softmax(inputs, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)

        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device).gather(0, targets)  # (N,)
            focal_weight = alpha_t * focal_weight

        loss = F.cross_entropy(inputs, targets, reduction="none")  # (N,)
        loss = (focal_weight * loss).mean()
        return loss
