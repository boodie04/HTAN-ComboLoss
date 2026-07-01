import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperLoss(nn.Module):
    """
    Equation (9) from TransAttUNet paper:
        L = 0.5 * BCE + 0.5 * Dice

    BCE uses logits directly (sigmoid applied internally).
    Dice uses sigmoid probabilities.
    """

    def forward(self, logits, targets, smooth=1e-6):
        # BCE — sigmoid applied internally
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)

        # Dice — apply sigmoid manually
        probs    = torch.sigmoid(logits)
        probs_f  = probs.view(-1)
        targets_f = targets.view(-1)

        intersection = (probs_f * targets_f).sum()
        dice_score   = (2. * intersection + smooth) / \
                       (probs_f.sum() + targets_f.sum() + smooth)
        dice_loss    = 1 - dice_score

        return 0.5 * bce_loss + 0.5 * dice_loss


class ComboLoss(nn.Module):
    """
    Taghanaki et al.-inspired Combo Loss.

    The cross-entropy term is the main smooth optimization signal. The overlap
    term is used as a regularizer so small foreground masks do not make training
    depend only on Dice/Jaccard fluctuations.

    alpha: weight for weighted BCE. Larger values make CE dominate.
    beta: positive-class BCE weight. Values > 0.5 penalize false negatives more.
    overlap: "dice" or "jaccard" regularizer.
    """

    def __init__(self, alpha=0.7, beta=0.7, overlap="dice", smooth=1.0):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if overlap not in ("dice", "jaccard"):
            raise ValueError("overlap must be 'dice' or 'jaccard'")

        self.alpha = alpha
        self.beta = beta
        self.overlap = overlap
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).clamp(min=1e-7, max=1.0 - 1e-7)
        targets = targets.float()

        weighted_bce = -(
            self.beta * targets * torch.log(probs)
            + (1.0 - self.beta) * (1.0 - targets) * torch.log(1.0 - probs)
        ).mean()

        probs_f = probs.view(probs.size(0), -1)
        targets_f = targets.view(targets.size(0), -1)
        intersection = (probs_f * targets_f).sum(dim=1)

        if self.overlap == "jaccard":
            union = probs_f.sum(dim=1) + targets_f.sum(dim=1) - intersection
            overlap_score = (intersection + self.smooth) / (union + self.smooth)
        else:
            overlap_score = (2.0 * intersection + self.smooth) / (
                probs_f.sum(dim=1) + targets_f.sum(dim=1) + self.smooth
            )

        overlap_loss = 1.0 - overlap_score.mean()
        return self.alpha * weighted_bce + (1.0 - self.alpha) * overlap_loss


def build_loss(config):
    name = config.get("LOSS", "paper").lower()
    if name in ("paper", "bce_dice"):
        return PaperLoss()
    if name == "combo":
        return ComboLoss(
            alpha=config.get("COMBO_ALPHA", 0.7),
            beta=config.get("COMBO_BETA", 0.7),
            overlap=config.get("COMBO_OVERLAP", "dice"),
            smooth=config.get("COMBO_SMOOTH", 1.0),
        )
    raise ValueError(f"Unknown loss: {name}")
