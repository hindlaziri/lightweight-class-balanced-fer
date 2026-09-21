"""Model and loss components for the TELKOMNIKA FER study."""
from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


class MobileNetV3FER(nn.Module):
    """MobileNetV3-Small adapted to seven facial-expression classes."""

    def __init__(self, num_classes: int = 7, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        self.backbone.classifier[2] = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FER-2013 images are grayscale; repeating the channel preserves compatibility
        # with the ImageNet-pretrained first convolution.
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)


class ClassBalancedFocalLabelSmoothing(nn.Module):
    """Effective-number weighted focal loss with soft targets.

    The target distribution is q=(1-epsilon)y+epsilon/C. The focal factor is
    applied per class and the effective-number weight compensates for the
    long-tailed training distribution.
    """

    def __init__(
        self,
        class_counts: Iterable[int],
        gamma: float = 2.0,
        smoothing: float = 0.05,
        beta: float = 0.9999,
    ):
        super().__init__()
        counts = torch.as_tensor(list(class_counts), dtype=torch.float32)
        if counts.ndim != 1 or torch.any(counts <= 0):
            raise ValueError("class_counts must be a one-dimensional positive vector")
        effective = 1.0 - torch.pow(torch.tensor(beta), counts)
        weights = (1.0 - beta) / effective
        weights = weights / weights.mean()
        self.register_buffer("class_weights", weights)
        self.gamma = float(gamma)
        self.smoothing = float(smoothing)
        self.num_classes = int(counts.numel())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        q = torch.full_like(log_probs, self.smoothing / self.num_classes)
        q.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / self.num_classes)
        focal = torch.pow(1.0 - probs.clamp_min(1e-6), self.gamma)
        weighted = q * focal * self.class_weights.view(1, -1)
        return -(weighted * log_probs).sum(dim=1).mean()


def make_loss(
    name: str,
    class_counts: Optional[Iterable[int]],
    label_smoothing: float,
    gamma: float,
) -> nn.Module:
    name = name.lower()
    if name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if name == "cbfocal":
        if class_counts is None:
            raise ValueError("class_counts are required for cbfocal")
        return ClassBalancedFocalLabelSmoothing(
            class_counts=class_counts,
            gamma=gamma,
            smoothing=label_smoothing,
        )
    raise ValueError(f"Unsupported loss: {name}")
