import torch
import torch.nn.functional as F

def focal_ce_loss(logits, target, weight=None, gamma=2.0, ignore_index=None, reduction='mean'):
    
    if logits.dim() == 3:
        logits = logits.transpose(1, 2).contiguous().view(-1, logits.size(1))
    if target.dim() != 1:
        target = target.view(-1)
    target = target.long()

    logp = F.log_softmax(logits, dim=1)

    if ignore_index is None:
        ce = F.nll_loss(logp, target, weight=weight, reduction='none')
    else:
        ce = F.nll_loss(logp, target, weight=weight, reduction='none', ignore_index=ignore_index)

    p = torch.exp(-ce)
    focal = (1 - p) ** gamma * ce
    if reduction == 'mean':
        return focal.mean()
    elif reduction == 'sum':
        return focal.sum()
    else:
        return focal

def dice_loss(logits, target, eps=1e-6):
    
    if logits.dim() == 2:
        logits = logits.t().unsqueeze(0)
    B, C, N = logits.shape
    if target.dim() != 1:
        target = target.view(-1)
    target_oh = torch.zeros_like(logits).scatter_(1, target.view(1, 1, -1).expand(B, 1, N), 1)
    prob = torch.softmax(logits, dim=1)
    inter = (prob * target_oh).sum(dim=(0, 2))
    union = (prob + target_oh).sum(dim=(0, 2))
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()

def combo_focal_dice(logits, target, weight=None, gamma=2.0, lambda_dice=0.2):
    
    lf = focal_ce_loss(logits, target, weight=weight, gamma=gamma, reduction='mean')
    ld = dice_loss(logits, target)
    return (1.0 - lambda_dice) * lf + lambda_dice * ld