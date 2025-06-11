import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


class DocumentParsingEvaluator:
    """
    Evaluator class for document parsing metrics.
    """
    
    def __init__(self, iou_thresholds=None, class_metrics=True, entity_metrics=True):
        """
        Initialize the evaluator with desired metrics.
        
        Args:
            iou_thresholds: List of IoU thresholds to evaluate at
            class_metrics: Whether to compute per-class metrics
            entity_metrics: Whether to compute per-entity metrics
        """
        self.iou_thresholds = iou_thresholds if iou_thresholds is not None else [0.5, 0.75, 0.9]
        self.class_metrics = class_metrics
        self.entity_metrics = entity_metrics
        self.reset()
        
    def reset(self):
        """Reset accumulated statistics."""
        self.stats = []
        
    def update(self, outputs, targets):
        """
        Update metrics with predictions and ground truth.
        
        Args:
            outputs: Model output dict with 'pred_logits' and 'pred_boxes'
            targets: List of target dicts with 'boxes', 'labels', etc.
        """
        # Process predictions
        pred_logits = outputs['pred_logits'].detach()
        pred_boxes = outputs['pred_boxes'].detach()
        
        # Get batch size
        batch_size = len(targets)
        
        # Debug: Print some information about predictions
        print(f"\nDEBUG: Prediction shapes: logits={pred_logits.shape}, boxes={pred_boxes.shape}")
        print(f"DEBUG: Background class probability range: {pred_logits[:, :, -1].min().item():.4f} to {pred_logits[:, :, -1].max().item():.4f}")
        
        total_boxes = 0
        total_valid_pred = 0
        
        # Process each sample in the batch
        for i in range(batch_size):
            # Get predictions for this sample
            logits = pred_logits[i]
            boxes = pred_boxes[i]
            
            # Get ground truth for this sample
            target = targets[i]
            gt_boxes = target['boxes']
            gt_labels = target['labels']
            
            # Debug: Print ground truth info
            print(f"DEBUG: Sample {i}: GT boxes={len(gt_boxes)}, GT labels={len(gt_labels)}")
            total_boxes += len(gt_boxes)
            
            # Apply score threshold and get top predictions
            # We use the background class (last one) as the score threshold
            # 1 - background_prob = foreground_prob
            scores = 1 - logits[:, -1].sigmoid()  # Use foreground probability
            # Get non-background class indices
            pred_labels = torch.argmax(logits[:, :-1], dim=-1)
            
            # Filter predictions with a score threshold - use a very low threshold for debugging
            score_thresh = 0.05  # Very low threshold to see if any predictions are being made
            keep = scores > score_thresh
            
            # Apply filtering
            scores = scores[keep]
            pred_labels = pred_labels[keep]
            boxes = boxes[keep]
            
            # Debug: Print score info
            print(f"DEBUG: Sample {i}: Valid predictions={len(scores)}/{len(logits)} (thresh={score_thresh})")
            print(f"DEBUG: Score range: {scores.min().item() if len(scores) > 0 else 0:.4f} to {scores.max().item() if len(scores) > 0 else 0:.4f}")
            total_valid_pred += len(scores)
            
            # Sort by decreasing score for NMS
            if len(scores) > 0:
                sort_idx = torch.argsort(scores, descending=True)
                scores = scores[sort_idx]
                pred_labels = pred_labels[sort_idx]
                boxes = boxes[sort_idx]
            
            # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]
            boxes = box_cxcywh_to_xyxy(boxes)
            gt_boxes = box_cxcywh_to_xyxy(gt_boxes) if len(gt_boxes) > 0 else gt_boxes
            
            # Compute metrics for this sample
            self.eval_sample(
                pred_boxes=boxes.cpu().numpy(),
                pred_labels=pred_labels.cpu().numpy(),
                pred_scores=scores.cpu().numpy(),
                gt_boxes=gt_boxes.cpu().numpy(),
                gt_labels=gt_labels.cpu().numpy()
            )
        
        # Debug: Print batch summary
        print(f"DEBUG: Batch summary: Total GT boxes={total_boxes}, Total valid predictions={total_valid_pred}")
    
    def eval_sample(self, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
        """
        Evaluate a single sample.
        
        Args:
            pred_boxes: Predicted boxes [N, 4]
            pred_labels: Predicted class labels [N]
            pred_scores: Prediction scores [N]
            gt_boxes: Ground truth boxes [M, 4]
            gt_labels: Ground truth class labels [M]
        """
        # If no ground truth, all predictions are false positives
        if len(gt_boxes) == 0:
            if len(pred_boxes) > 0:
                self.stats.append({
                    'pred_boxes': pred_boxes,
                    'pred_labels': pred_labels,
                    'pred_scores': pred_scores,
                    'gt_boxes': np.zeros((0, 4)),
                    'gt_labels': np.zeros((0,), dtype=np.int64),
                    'matched': []
                })
            return
        
        # If no predictions, all ground truths are false negatives
        if len(pred_boxes) == 0:
            self.stats.append({
                'pred_boxes': np.zeros((0, 4)),
                'pred_labels': np.zeros((0,), dtype=np.int64),
                'pred_scores': np.zeros((0,)),
                'gt_boxes': gt_boxes,
                'gt_labels': gt_labels,
                'matched': []
            })
            return
        
        # Calculate IoU between all predictions and ground truths
        iou_matrix = box_iou_batch(
            torch.tensor(pred_boxes),
            torch.tensor(gt_boxes)
        ).cpu().numpy()
        
        # Calculate matching at each IoU threshold
        matched_gt_inds = {}
        for iou_thresh in self.iou_thresholds:
            # For each prediction, get the best matching ground truth
            # if their IoU is above the threshold
            matches = []
            for pred_idx in range(len(pred_boxes)):
                # Get max IoU for this prediction
                max_iou_idx = np.argmax(iou_matrix[pred_idx])
                max_iou = iou_matrix[pred_idx, max_iou_idx]
                
                # Add match if IoU is above threshold and the classes match
                if max_iou >= iou_thresh and pred_labels[pred_idx] == gt_labels[max_iou_idx]:
                    matches.append((pred_idx, max_iou_idx, max_iou))
                    # Zero out this ground truth to avoid double-matching
                    iou_matrix[:, max_iou_idx] = 0
            
            matched_gt_inds[iou_thresh] = matches
            
        # Store results
        self.stats.append({
            'pred_boxes': pred_boxes,
            'pred_labels': pred_labels,
            'pred_scores': pred_scores,
            'gt_boxes': gt_boxes,
            'gt_labels': gt_labels,
            'matched': matched_gt_inds
        })
    
    def compute(self):
        """
        Compute metrics from accumulated stats.
        
        Returns:
            Dict with computed metrics
        """
        if not self.stats:
            return {}
        
        results = {}
        
        # Compute overall mAP
        for iou_thresh in self.iou_thresholds:
            precision, recall, f1, ap = self._compute_map(iou_thresh)
            results[f'mAP@{iou_thresh}'] = ap
            results[f'precision@{iou_thresh}'] = precision
            results[f'recall@{iou_thresh}'] = recall
            results[f'F1@{iou_thresh}'] = f1
        
        # Compute per-class metrics if needed
        if self.class_metrics:
            class_results = self._compute_class_metrics()
            results.update(class_results)
        
        return results
    
    def _compute_map(self, iou_thresh):
        """
        Compute mean Average Precision at specific IoU threshold.
        """
        # Collect all predictions
        all_preds = []
        for stat in self.stats:
            pred_scores = stat['pred_scores']
            for i, score in enumerate(pred_scores):
                match_found = False
                for thresh, matches in stat['matched'].items():
                    if thresh == iou_thresh:
                        for match in matches:
                            if match[0] == i:  # This prediction was matched
                                match_found = True
                                break
                    if match_found:
                        break
                
                all_preds.append({
                    'score': score,
                    'correct': match_found
                })
        
        # Sort predictions by decreasing score
        all_preds.sort(key=lambda x: x['score'], reverse=True)
        
        # Count ground truth instances
        num_gt = sum(len(stat['gt_boxes']) for stat in self.stats)
        
        if not all_preds or num_gt == 0:
            return 0, 0, 0, 0
        
        # Compute precision and recall
        tp = 0  # true positives
        fp = 0  # false positives
        
        precisions = []
        recalls = []
        
        for pred in all_preds:
            if pred['correct']:
                tp += 1
            else:
                fp += 1
            
            precision = tp / (tp + fp)
            recall = tp / num_gt
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Compute AP using 11-point interpolation
        ap = 0
        for t in np.linspace(0, 1, 11):
            # Get max precision for recall >= t
            mask = np.array(recalls) >= t
            if mask.any():
                ap += max(np.array(precisions)[mask])
        
        ap /= 11
        
        # Calculate final precision, recall and F1
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / num_gt if num_gt > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        return precision, recall, f1, ap
    
    def _compute_class_metrics(self):
        """
        Compute metrics for each class separately.
        """
        # Collect all class labels
        class_labels = set()
        for stat in self.stats:
            class_labels.update(stat['gt_labels'])
        
        class_metrics = {}
        for cls_label in class_labels:
            # Calculate mAP for this class at each IoU threshold
            for iou_thresh in self.iou_thresholds:
                ap = self._compute_class_ap(cls_label, iou_thresh)
                class_metrics[f'class_{cls_label}_AP@{iou_thresh}'] = ap
        
        return class_metrics
    
    def _compute_class_ap(self, cls_label, iou_thresh):
        """
        Compute Average Precision for a specific class at a specific IoU threshold.
        """
        # Filter predictions and ground truth for this class
        filtered_stats = []
        
        for stat in self.stats:
            # Get indices of ground truths with this class
            gt_indices = np.where(stat['gt_labels'] == cls_label)[0]
            gt_boxes_filtered = stat['gt_boxes'][gt_indices] if len(gt_indices) > 0 else np.zeros((0, 4))
            gt_labels_filtered = np.ones(len(gt_indices), dtype=np.int64) * cls_label
            
            # Get indices of predictions with this class
            pred_indices = np.where(stat['pred_labels'] == cls_label)[0]
            pred_boxes_filtered = stat['pred_boxes'][pred_indices] if len(pred_indices) > 0 else np.zeros((0, 4))
            pred_labels_filtered = np.ones(len(pred_indices), dtype=np.int64) * cls_label
            pred_scores_filtered = stat['pred_scores'][pred_indices] if len(pred_indices) > 0 else np.zeros((0,))
            
            # Filter matched pairs for this class
            matched_filtered = {}
            for thresh, matches in stat['matched'].items():
                if thresh == iou_thresh:
                    # A match is valid if both the prediction and ground truth are of the desired class
                    valid_matches = []
                    for p_idx, gt_idx, iou in matches:
                        if p_idx in pred_indices and gt_idx in gt_indices:
                            # Convert from global to local indices
                            p_local_idx = list(pred_indices).index(p_idx)
                            gt_local_idx = list(gt_indices).index(gt_idx)
                            valid_matches.append((p_local_idx, gt_local_idx, iou))
                    
                    matched_filtered[thresh] = valid_matches
            
            # Add to filtered stats
            filtered_stats.append({
                'pred_boxes': pred_boxes_filtered,
                'pred_labels': pred_labels_filtered,
                'pred_scores': pred_scores_filtered,
                'gt_boxes': gt_boxes_filtered,
                'gt_labels': gt_labels_filtered,
                'matched': matched_filtered
            })
        
        # Compute mAP using filtered stats
        temp_evaluator = DocumentParsingEvaluator(iou_thresholds=[iou_thresh], class_metrics=False)
        temp_evaluator.stats = filtered_stats
        metrics = temp_evaluator.compute()
        
        return metrics.get(f'mAP@{iou_thresh}', 0)


# Utility functions

def box_cxcywh_to_xyxy(x):
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2] format."""
    if isinstance(x, torch.Tensor):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)
    else:  # numpy array
        x_c, y_c, w, h = x.T
        b = np.stack([
            x_c - 0.5 * w, 
            y_c - 0.5 * h,
            x_c + 0.5 * w, 
            y_c + 0.5 * h
        ]).T
        return b

def box_xyxy_to_cxcywh(x):
    """Convert [x1, y1, x2, y2] to [cx, cy, w, h] format."""
    if isinstance(x, torch.Tensor):
        x0, y0, x1, y1 = x.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
             (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)
    else:  # numpy array
        x0, y0, x1, y1 = x.T
        b = np.stack([
            (x0 + x1) / 2, 
            (y0 + y1) / 2,
            (x1 - x0), 
            (y1 - y0)
        ]).T
        return b

def box_iou_batch(boxes1, boxes2):
    """
    Compute IoU between all pairs of boxes in batch.
    
    Args:
        boxes1: [N, 4] boxes in (x1, y1, x2, y2) format
        boxes2: [M, 4] boxes in (x1, y1, x2, y2) format
        
    Returns:
        Tensor of shape [N, M] containing IoU values
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]
    
    # Calculate intersection area
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    intersection = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    # Calculate IoU
    union = area1[:, None] + area2 - intersection  # [N, M]
    iou = intersection / union
    
    return iou 