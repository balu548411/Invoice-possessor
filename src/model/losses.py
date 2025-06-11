import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher for bipartite matching between predictions and ground truth.
    
    This module computes an assignment between targets and predictions based on the cost.
    The costs are weighted sum of classification and regression losses.
    """
    
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 1.0, cost_giou: float = 1.0):
        """
        Args:
            cost_class: Weight for classification cost
            cost_bbox: Weight for L1 box regression cost
            cost_giou: Weight for generalized IoU cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        
    @torch.no_grad()
    def forward(self, outputs: Dict, targets: List[Dict]):
        """
        Compute assignment between targets and predictions.
        
        Args:
            outputs: Dict with 'pred_logits' [batch_size, num_queries, num_classes+1] and
                    'pred_boxes' [batch_size, num_queries, 4]
            targets: List of dicts with 'labels' [num_target_boxes] and 
                    'boxes' [num_target_boxes, 4]
                    
        Returns:
            List[Tuple]: A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes+1]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        
        # Also flatten the targets
        tgt_ids = torch.cat([t["labels"] for t in targets])
        tgt_bbox = torch.cat([t["boxes"] for t in targets])
        
        # Compute classification cost
        # Out_prob is already after softmax, so get the probability for the correct class using gather
        cost_class = -out_prob[:, tgt_ids]
        
        # Compute L1 distance between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # Compute generalized IoU cost
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        
        # Split targets by batch element
        sizes = [len(t["boxes"]) for t in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        # Convert numpy arrays to PyTorch tensors
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class SetCriterion(nn.Module):
    """
    Loss computation for document parser.
    
    This class computes the loss for document parsing using a bipartite matching approach.
    It handles classification and bounding box regression losses.
    """
    
    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1, losses=None):
        """
        Args:
            num_classes: Number of entity classes
            matcher: Module for matching targets to predictions
            weight_dict: Dict of scalar loss weights
            eos_coef: Weight of the no-object class
            losses: List of losses to apply
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses if losses is not None else ['labels', 'boxes', 'cardinality']
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (NLL).
        
        Args:
            outputs: Dict of outputs
            targets: List of target dicts
            indices: List of (pred_indices, tgt_indices) tuples
            num_boxes: Total number of target boxes
            
        Returns:
            Loss dict with classification losses
        """
        assert 'pred_logits' in outputs
        
        src_logits = outputs['pred_logits']
        
        # Extract matched indices
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                  dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        
        return losses
        
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute bounding box loss including L1 and GIoU losses.
        
        Args:
            outputs: Dict of outputs
            targets: List of target dicts
            indices: List of (pred_indices, tgt_indices) tuples
            num_boxes: Total number of target boxes
            
        Returns:
            Loss dict with L1 and GIoU box losses
        """
        assert 'pred_boxes' in outputs
        
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}
        
        # GIoU loss
        giou_loss = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        ))
        losses['loss_giou'] = giou_loss.sum() / num_boxes
        
        return losses
        
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute cardinality error (measures difference between predicted and ground truth quantities).
        
        Args:
            outputs: Dict of outputs
            targets: List of target dicts
            indices: List of (pred_indices, tgt_indices) tuples
            num_boxes: Total number of target boxes
            
        Returns:
            Loss dict with cardinality error
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != self.num_classes).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses
        
    def _get_src_permutation_idx(self, indices):
        """
        Get the permutation indices i.e., the target indices that match the predictions.
        
        Args:
            indices: List of (pred_indices, tgt_indices) tuples
            
        Returns:
            Tuple (batch_idx, src_idx) to index into a flattened batch tensor
        """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
        
    def _get_tgt_permutation_idx(self, indices):
        """
        Get the permutation indices i.e., the target indices that match the predictions.
        
        Args:
            indices: List of (pred_indices, tgt_indices) tuples
            
        Returns:
            Tuple (batch_idx, tgt_idx) to index into a flattened batch tensor
        """
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
        
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        """
        Call the appropriate loss function.
        """
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'cardinality': self.loss_cardinality,
        }
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
        
    def forward(self, outputs, targets):
        """
        Main forward method, calculates all specified losses.
        
        Args:
            outputs: Dict of outputs
            targets: List of target dicts
            
        Returns:
            Loss dict with all computed losses
        """
        # Retrieve the matching between the outputs and targets
        indices = self.matcher(outputs, targets)
        
        # Compute the total number of target boxes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs["pred_logits"].device)
        
        # Distribute losses to workers if in distributed training
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes, min=1).item()
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
            
        # Handle auxiliary losses (from intermediate decoder layers)
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                    
        return losses


def build_criterion(config):
    """
    Build the complete loss criterion.
    
    Args:
        config: Configuration dict
        
    Returns:
        SetCriterion instance
    """
    from config import MODEL_CONFIG
    
    # Get number of classes
    num_classes = len(MODEL_CONFIG["entity_classes"])
    
    # Define matcher
    matcher = HungarianMatcher(
        cost_class=1.0,
        cost_bbox=5.0,
        cost_giou=2.0
    )
    
    # Define loss weights
    weight_dict = {
        'loss_ce': 1.0,
        'loss_bbox': 5.0,
        'loss_giou': 2.0,
    }
    
    # Add auxiliary losses for intermediate decoder layers
    aux_weight_dict = {}
    for i in range(config.get("decoder_layers", 6) - 1):
        aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    
    losses = ['labels', 'boxes', 'cardinality']
    
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=0.1,
        losses=losses
    )
    
    return criterion


# Utility functions for box operations
def box_cxcywh_to_xyxy(x):
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2] format."""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    """Convert [x1, y1, x2, y2] to [cx, cy, w, h] format."""
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def generalized_box_iou(boxes1, boxes2):
    """
    Compute generalized IoU between box sets.
    
    Args:
        boxes1, boxes2: Tensors of shape [N, 4] and [M, 4] in format (x1, y1, x2, y2)
        
    Returns:
        A tensor of shape [N, M] containing the pairwise generalized IoU values
    """
    # Calculate box areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Calculate IoU
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    intersection = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    union = area1[:, None] + area2 - intersection
    iou = intersection / union
    
    # Calculate enclosing box area
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    enclosure = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    # Calculate GIoU
    giou = iou - (enclosure - union) / enclosure
    
    return giou

def linear_sum_assignment(cost_matrix):
    """
    Solve the linear sum assignment problem using the Hungarian algorithm.
    This is a simple wrapper around scipy.optimize.linear_sum_assignment.
    
    Args:
        cost_matrix: A cost matrix of shape [N, M]
        
    Returns:
        Two arrays (row_indices, col_indices) of equal size representing the optimal assignment
    """
    try:
        from scipy.optimize import linear_sum_assignment as linear_sum_assignment_scipy
        return linear_sum_assignment_scipy(cost_matrix.numpy())
    except ImportError:
        raise ImportError("This function requires scipy.optimize.linear_sum_assignment") 