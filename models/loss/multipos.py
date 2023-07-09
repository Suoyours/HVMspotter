import torch
import torch.nn as nn
import torch.nn.functional as F


def multi_pos_cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None, device='cuda'):
    # element-wise losses
    # pos_inds = (label == 1).float()
    # neg_inds = (label == 0).float()
    # exp_pos = (torch.exp(-1 * pred) * pos_inds).sum(dim=1)
    # exp_neg = (torch.exp(pred.clamp(max=80)) * neg_inds).sum(dim=1)
    # loss = torch.log(1 + exp_pos * exp_neg)

    # a more numerical stable implementation.
    pos_inds = (label == 1)
    neg_inds = (label == 0)
    pred_pos = pred * pos_inds.float().to(torch.device(device))
    pred_neg = pred * neg_inds.float().to(torch.device(device))
    # use -inf to mask out unwanted elements.
    pred_pos[neg_inds] = pred_pos[neg_inds] + float('inf')
    pred_neg[pos_inds] = pred_neg[pos_inds] + float('-inf')

    _pos_expand = torch.repeat_interleave(pred_pos, pred.shape[1], dim=1)  # 重复张量的元素
    _neg_expand = pred_neg.repeat(1, pred.shape[1])

    x = torch.nn.functional.pad((_neg_expand - _pos_expand), (0, 1), "constant", 0)
    # 向右填充1列，填充值为0
    loss = torch.logsumexp(x, dim=1)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


class MultiPosCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0, device='cuda'):
        super(MultiPosCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.device = device

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        assert cls_score.size() == label.size()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * multi_pos_cross_entropy(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            device=self.device,
            **kwargs)
        return loss_cls


# a = torch.rand(6, 512)
#
# ##特征向量进行归一化
# a = F.normalize(a, p=2, dim=1)
# target_n = torch.tensor([[1, 1, 0, 0, 1, 0],
#                          [1, 1, 1, 1, 0, 0],
#                          [0, 1, 1, 0, 0, 0],
#                          [0, 1, 0, 1, 0, 0],
#                          [0, 0, 0, 0, 1, 1],
#                          [0, 1, 0, 0, 1, 1],
#                          ])
# sim = torch.matmul(a, a.T)
# print(sim)
#
# mp_loss = MultiPosCrossEntropyLoss(device='cpu')
# loss_class = mp_loss(sim, target_n)
# print(loss_class)
