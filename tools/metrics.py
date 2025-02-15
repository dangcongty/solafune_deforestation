import numpy as np
import torch
import torch.nn.functional as F
from shapely.geometry import Polygon


def getIOU(polygon1: Polygon, polygon2: Polygon) -> float:
    """
    Computes the Intersection over Union (IoU) between two polygons.
    Parameters
    ----------
    polygon1 : Polygon
        The first polygon.
    polygon2 : Polygon
        The second polygon.
    Returns
    -------
    float
        The IoU value between the two polygons.
    """
    intersection = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    if union == 0:
        return 0
    return intersection / union


def compute_iou(pred, gt, num_classes = 2, threshold = 0.5, epsilon=1e-6):
    """
    Computes the Intersection over Union (IoU) metric for semantic segmentation.
    
    Args:
        pred (torch.Tensor): Predicted segmentation, shape (Batchsize, num_classes, H, W).
        gt (torch.Tensor): Ground truth segmentation, shape (Batchsize, H, W).
        epsilon (float): Small value to avoid division by zero.
        threshold: threshold for bg
        
    Returns:
        torch.Tensor: IoU score for each class, shape (num_classes,). tensor([0.0869, 0.0721, 0.1962, 0.1404, 0.0406])
    """
    # Ensure predictions are binary or probabilities
    # pred[pred < threshold] = 0 
    pred = pred.argmax(1) 

    # Compute intersection and union for each class
    ious = []
    for i in range(num_classes):
        pred_i = (pred==i).float()
        gt_i = (gt==i).float()
        intersection = torch.sum(pred_i * gt_i, dim=(1, 2))
        union = torch.sum(pred_i + gt_i, dim=(1, 2)) - intersection  
        iou = intersection / (union + epsilon)  
        ious.append(iou)
    
    # Trường hợp sample ko có class nhưng vẫn tính mean => IoU thấp
    ious = torch.stack(ious)
    mask = torch.ones_like(ious)
    for b in range(gt.shape[0]):
        indice = torch.argwhere(torch.bincount(gt[b].flatten())==0).flatten()
        mask[indice, b] = 0 # mask những class ko xuất hiện trong sample
    # mask shape: 5x8 ---- 5 classes and batchsize 8
    
    ious = ious.sum(1)/(mask.sum(1) +1e-10) # tính trung bình iou của từng class trên tất cả batch
    return ious, mask.sum(1)

if __name__ == "__main__":
    input = torch.rand((2, 5, 10, 10)).softmax(1)
    target = torch.randint(0, 5, (2, 10, 10))
    target = F.one_hot(target, 5).permute((0, -1, 1, 2))
    compute_iou(input, target)