import torch

def mask2onehot(mask, n_class):
    """
    Transform a mask to one hot
    change a mask to n * h* w   n is the class
    Args:
        mask:
        n_class: number of class for segmentation
    Returns:
        y_one_hot: one hot mask
    """
    y_one_hot = torch.zeros((n_class, mask.shape[1], mask.shape[2])).to('cuda')
    y_one_hot = y_one_hot.scatter(0, mask, 1).long()
    return y_one_hot

def iou(pred, target):
    eps = 0.00001
    if (pred.shape[0] != 1):
        target = mask2onehot(target, pred.shape[0]).float()
    #print("iou",pred.shape,target.shape)
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    #pred = (pred > 0.5).float()
    intersection = (torch.dot(pred, target))  # Cast to long to prevent overflows
    union = pred.sum() + target.sum() - intersection
    ious = (float(intersection) +eps)/ (float(union)+eps)
    return ious
def dice(pred, target):
    eps = 0.0001
    if(pred.shape[0]!=1):
        target = mask2onehot(target,pred.shape[0]).float()
    #print("dice", pred.shape, target.shape)
    smooth=torch.tensor(1.0)
    m1 = pred.reshape(-1)
    m2 = target.reshape(-1)
    #m1 = (m1 > 0.5).float()
    intersection = (torch.dot(m1, m2))
    return float((torch.tensor(2.0) * intersection + smooth)) /float((m1.sum() + m2.sum() + smooth))
