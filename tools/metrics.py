import numpy as np
import torch
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

class F1_Metrics:
    def __init__(self) -> None:
        """
        A class used to compute F1 metrics for polygon-based segmentation.
        Methods
        -------
        compute_f1(gt_polygons: list, pred_polygons: list, iou_threshold=0.5)
            Computes the F1 score between ground truth and predicted polygons.
        """
        pass
    def compute_f1(self, gt_polygons: list, pred_polygons: list, iou_threshold=0.5) -> tuple:
        """
        Compute the F1 score, precision, and recall for the given ground truth and predicted polygons.
    
        Args:
            gt_polygons (list): List of ground truth polygons.
            pred_polygons (list): List of predicted polygons.
            iou_threshold (float, optional): Intersection over Union (IoU) threshold to consider a match. Defaults to 0.5.
    
        Returns:
            tuple: A tuple containing the F1 score, precision, and recall.
        """
        matched_instances = {}
        gt_matched = np.zeros(len(gt_polygons))
        pred_matched = np.zeros(len(pred_polygons))

        # IoU計算とマッチング候補の特定
        gt_matched = np.zeros(len(gt_polygons))
        pred_matched = np.zeros(len(pred_polygons))
        for gt_idx, gt_polygon in enumerate(gt_polygons):
            best_iou = iou_threshold
            best_pred_idx = None
            for pred_idx, pred_polygon in enumerate(pred_polygons):
                # if gt_matched[gt_idx] == 1 or pred_matched[pred_idx] == 1:
                #     continue
                
                iou = getIOU(gt_polygon, pred_polygon)
                if iou == 0:
                    continue
                
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx
            if best_pred_idx is not None:
                matched_instances[(gt_idx, best_pred_idx)] = best_iou
                gt_matched[gt_idx] = 1
                pred_matched[best_pred_idx] = 1

        # F1, Precision, Recall
        
        tp = len(matched_instances)
        fp = len(pred_polygons) - tp
        fn = len(gt_polygons) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1, precision, recall

def compute_iou(pred, gt, threshold = 0.5, epsilon=1e-6):
    """
    Computes the Intersection over Union (IoU) metric for semantic segmentation.
    
    Args:
        pred (torch.Tensor): Predicted segmentation, shape (Batchsize, num_classes, H, W).
        gt (torch.Tensor): Ground truth segmentation, shape (Batchsize, num_classes, H, W).
        epsilon (float): Small value to avoid division by zero.
        
    Returns:
        torch.Tensor: IoU score for each class, shape (num_classes,).
    """
    # Ensure predictions are binary or probabilities
    pred = (pred > threshold).float()  # Thresholding predicted probabilities at 0.5

    # Compute intersection and union for each class
    intersection = torch.sum(pred * gt, dim=(0, 2, 3))  # Sum over spatial dimensions and batch
    union = torch.sum(pred + gt, dim=(0, 2, 3)) - intersection  # Union = A + B - Intersection

    # Compute IoU
    iou = intersection / (union + epsilon)  # Add epsilon to avoid division by zero

    return iou