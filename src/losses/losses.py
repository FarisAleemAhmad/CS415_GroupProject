"""Loss functions: cross entropy, Ordinal Log Loss (OLL), Multitask Log Loss (MLL).

All take raw logits (B, K) and integer labels (B,) in 0..K-1.

OLL (Castagnos et al., COLING 2022):
    L = - sum_i log(1 - p_i) * |i - y|^alpha
  Probability mass put on classes far from the true label is penalized more.
  (The true class has distance 0, so it contributes nothing directly; its
  probability rises because the other probabilities are pushed down.)

MLL (Kasa et al., Findings of ACL 2024, Eq. 5):
    L = lam * CE + (1 - lam) * OLL,   lam in [0, 1]
  lam=1 is pure CE, lam=0 is pure OLL. The paper gives no value for lam, so it
  is tuned on the validation set (default 0.5).
  OLL's sum over k != y equals the sum over all k here, since |k-y|^alpha = 0 at k=y.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-7


class CrossEntropy(nn.Module):
    def forward(self, logits, y):
        return F.cross_entropy(logits, y)


class OLL(nn.Module):
    def __init__(self, num_classes: int = 5, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.register_buffer("classes", torch.arange(num_classes).float())

    def forward(self, logits, y):
        p = F.softmax(logits, dim=-1)
        dist = (self.classes[None, :] - y[:, None].float()).abs() ** self.alpha  # (B,K)
        return (-torch.log(1.0 - p + EPS) * dist).sum(dim=-1).mean()


class MLL(nn.Module):
    def __init__(self, num_classes: int = 5, alpha: float = 1.0, lam: float = 0.5):
        super().__init__()
        assert 0.0 <= lam <= 1.0, "lam must be in [0, 1]"
        self.ce = CrossEntropy()
        self.oll = OLL(num_classes, alpha)
        self.lam = lam

    def forward(self, logits, y):
        return self.lam * self.ce(logits, y) + (1.0 - self.lam) * self.oll(logits, y)


def get_loss(name: str, num_classes: int = 5, alpha: float = 1.0, lam: float = 0.5):
    name = name.lower()
    if name == "ce":
        return CrossEntropy()
    if name == "oll":
        return OLL(num_classes, alpha)
    if name == "mll":
        return MLL(num_classes, alpha, lam)
    raise ValueError(f"unknown loss {name}")
