import torch
import torch.nn.functional as F



def modified_focal_loss(pred, gt):
    '''
    Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)
    # clamp min value is set to 1e-12 to maintain the numerical stability
    pred = torch.clamp(pred, 1e-12)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss



# def modified_focal_loss(preds, targets):
#   pos_inds = targets == 1  # todo targets > 1-epsilon ?
#   neg_inds = targets < 1  # todo targets < 1-epsilon ?

#   neg_weights = torch.pow(1 - targets[neg_inds], 4)

#   loss = 0
#   for pred in preds:
#     pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
#     pos_pred = pred[pos_inds]
#     neg_pred = pred[neg_inds]

#     pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
#     neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

#     num_pos = pos_inds.float().sum()
#     pos_loss = pos_loss.sum()
#     neg_loss = neg_loss.sum()

#     if pos_pred.nelement() == 0:
#       loss = loss - neg_loss
#     else:
#       loss = loss - (pos_loss + neg_loss) / num_pos
#   return loss