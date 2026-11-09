"""Post-online-augmentation auxiliary targets. Main CeDiceLoss is untouched."""
import torch
from torch.nn import functional as F


def auxiliary_targets(labels, size, classes):
    if labels.ndim != 3 or labels.dtype != torch.long:
        raise ValueError('Expected post-augmentation LongTensor [B,H,W]')
    if labels.numel() == 0 or labels.min() < 0 or labels.max() >= classes:
        raise ValueError('Label outside configured class range')
    one_hot = F.one_hot(labels, classes).permute(0, 3, 1, 2).float()
    soft = F.adaptive_avg_pool2d(one_hot, size)
    presence = one_hot[:, 1:].flatten(2).amax(-1)
    return soft, presence


def auxiliary_losses(semantic_logits, presence_logits, labels):
    with torch.cuda.amp.autocast(enabled=False):
        logits = semantic_logits.float()
        soft, present = auxiliary_targets(labels, logits.shape[-2:], logits.shape[1])
        if presence_logits.shape != present.shape:
            raise RuntimeError('Incorrect auxiliary presence shape')
        soft_ce = -(soft * logits.log_softmax(1)).sum(1).mean()
        prediction, truth = logits.softmax(1)[:, 1:], soft[:, 1:]
        intersect = (prediction*truth).sum((2, 3))
        dice = (2*intersect+1e-6)/(prediction.sum((2, 3))+truth.sum((2, 3))+1e-6)
        # Present classes per sample; all-background samples have no Dice term.
        dice_loss = (((1-dice)*present).sum(1)/present.sum(1).clamp_min(1)).mean()
        semantic = soft_ce + dice_loss
        presence = F.binary_cross_entropy_with_logits(presence_logits.float(), present)
        return semantic, presence
