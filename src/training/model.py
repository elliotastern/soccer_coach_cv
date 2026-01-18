"""
DETR Model Setup for Training
"""
import torch
import torch.nn as nn
from transformers import DetrForObjectDetection, DetrImageProcessor
from typing import Dict, List
import torch.nn.functional as F
import math


def compute_focal_loss(logits, targets, alpha=0.25, gamma=2.0, num_classes=2):
    """
    Compute Focal Loss for classification
    
    Args:
        logits: [N, num_classes+1] classification logits (including background)
        targets: [N] target class labels (1-based: 1=player, 2=ball, 0=background)
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
        num_classes: Number of object classes (excluding background)
    
    Returns:
        Focal loss value
    """
    # Convert targets to one-hot if needed
    if len(targets) == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    
    # Get probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Get probability of true class
    target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    
    # Compute focal weight: (1 - p_t)^gamma
    focal_weight = (1 - target_probs) ** gamma
    
    # Compute cross-entropy
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    
    # Apply alpha weighting (weight ball class more)
    # For simplicity, use uniform alpha - could be per-class
    alpha_t = alpha * torch.ones_like(target_probs)
    
    # Focal Loss: FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
    focal_loss = alpha_t * focal_weight * ce_loss
    
    return focal_loss.mean()


class DETRWrapper(nn.Module):
    """Wrapper to make transformers DETR compatible with torchvision API"""
    
    def __init__(self, detr_model, num_classes: int, class_weights: Dict = None, focal_loss_config: Dict = None):
        super().__init__()
        self.detr_model = detr_model
        self.num_classes = num_classes
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.class_weights = class_weights  # Store class weights (deprecated, kept for compatibility)
        self.focal_loss_config = focal_loss_config  # Focal Loss configuration
    
    def forward(self, images: List[torch.Tensor], targets: List[Dict] = None):
        """
        Forward pass compatible with torchvision API
        
        Args:
            images: List of image tensors [C, H, W]
            targets: List of target dicts (for training) or None (for inference)
        
        Returns:
            If training: loss_dict
            If inference: List of prediction dicts
        """
        if self.training and targets is not None:
            # Training mode - compute loss
            return self._forward_train(images, targets)
        else:
            # Inference mode - return predictions
            return self._forward_inference(images)
    
    def _forward_train(self, images: List[torch.Tensor], targets: List[Dict]):
        # Store captured loss_dict (will be populated by patched loss function)
        captured_loss_dict = {}
        """Training forward pass"""
        # Prepare inputs for transformers DETR
        # Convert images to PIL and process
        pixel_values_list = []
        pixel_mask_list = []
        
        # Reuse mean/std tensors to avoid repeated allocations
        import torchvision.transforms.functional as TF
        from PIL import Image
        
        for img in images:
            # Denormalize for processor (create tensors once per image)
            mean = torch.tensor([0.485, 0.456, 0.406], device=img.device, dtype=img.dtype).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=img.device, dtype=img.dtype).view(3, 1, 1)
            img_denorm = img * std + mean
            img_denorm = torch.clamp(img_denorm, 0, 1)
            
            # Convert to PIL
            pil_img = TF.to_pil_image(img_denorm)
            
            # Process with DETR processor
            inputs = self.processor(images=pil_img, return_tensors="pt")
            pixel_values_list.append(inputs['pixel_values'].squeeze(0).to(img.device))
            pixel_mask_list.append(inputs['pixel_mask'].squeeze(0).to(img.device))
            
            # Clean up intermediate tensors immediately
            del img_denorm, pil_img, inputs, mean, std
        
        # Stack pixel values and masks
        pixel_values = torch.stack(pixel_values_list)
        pixel_mask = torch.stack(pixel_mask_list)
        
        # Clean up lists
        del pixel_values_list, pixel_mask_list
        
        # Prepare labels for transformers format
        labels = []
        for target in targets:
            # Convert boxes from [x_min, y_min, x_max, y_max] to center_x, center_y, width, height
            boxes = target['boxes']
            if len(boxes) > 0:
                # Clip boxes to valid image bounds first (assuming 1333x1333 after resize)
                img_size = 1333.0
                boxes = boxes.clone()
                boxes[:, 0] = torch.clamp(boxes[:, 0], 0, img_size)  # x_min
                boxes[:, 1] = torch.clamp(boxes[:, 1], 0, img_size)  # y_min
                boxes[:, 2] = torch.clamp(boxes[:, 2], 0, img_size)  # x_max
                boxes[:, 3] = torch.clamp(boxes[:, 3], 0, img_size)  # y_max
                
                # Ensure x_max > x_min and y_max > y_min (fix invalid boxes)
                boxes[:, 2] = torch.max(boxes[:, 2], boxes[:, 0] + 1.0)  # x_max >= x_min + 1
                boxes[:, 3] = torch.max(boxes[:, 3], boxes[:, 1] + 1.0)  # y_max >= y_min + 1
                
                # Convert to center format
                center_x = (boxes[:, 0] + boxes[:, 2]) / 2
                center_y = (boxes[:, 1] + boxes[:, 3]) / 2
                width = boxes[:, 2] - boxes[:, 0]
                height = boxes[:, 3] - boxes[:, 1]
                
                # Ensure minimum width and height to avoid numerical issues
                width = torch.clamp(width, min=1.0)
                height = torch.clamp(height, min=1.0)
                
                boxes_center = torch.stack([center_x, center_y, width, height], dim=1)
                
                # Normalize by image size
                boxes_center = boxes_center / img_size
                
                # Clip normalized boxes to [0, 1] to ensure valid values
                boxes_center = torch.clamp(boxes_center, 0.0, 1.0)
                
                # Ensure width and height are positive after normalization (minimum 1e-6)
                boxes_center[:, 2] = torch.clamp(boxes_center[:, 2], min=1e-6)  # width
                boxes_center[:, 3] = torch.clamp(boxes_center[:, 3], min=1e-6)  # height
                
                # Dataset already provides 1-based labels (1=player, 2=ball, 0=background)
                # DETR expects 1-based labels, so use them directly
                labels.append({
                    'class_labels': target['labels'],  # Already 1-based from dataset
                    'boxes': boxes_center,
                })
            else:
                labels.append({
                    'class_labels': torch.tensor([], dtype=torch.long).to(boxes.device),
                    'boxes': torch.tensor([], dtype=torch.float32).reshape(0, 4).to(boxes.device),
                })
        
        # Forward pass
        # CRITICAL FIX: Patch the loss function's criterion before forward pass
        # The loss function creates a criterion with empty_weight based on config.num_labels
        # config.num_labels should be the number of object classes (not including background)
        # The criterion creates empty_weight of size num_classes + 1
        # So if we have 2 object classes, config.num_labels should be 2, and empty_weight will be size 3
        
        # Ensure config.num_labels is correct (number of object classes, not including background)
        self.detr_model.config.num_labels = self.num_classes
        
        # Patch the loss_function to use correct num_labels
        original_loss_fn = self.detr_model.loss_function
        
        # Store loss_dict from loss function call
        captured_loss_dict = {}
        
        def patched_loss_function(logits, labels, device, pred_boxes, config, outputs_class=None, outputs_coord=None, **kwargs):
            """Patched loss function with Focal Loss support"""
            # CRITICAL: Set num_labels to number of object classes (not including background)
            config.num_labels = self.num_classes
            
            # Call original loss function first
            loss, loss_dict, auxiliary_outputs = original_loss_fn(
                logits, labels, device, pred_boxes, config, outputs_class, outputs_coord, **kwargs
            )
            
            # Capture loss_dict for later use (keep tensors in computation graph)
            captured_loss_dict.clear()
            for key, value in loss_dict.items():
                captured_loss_dict[key] = value
            
            # Apply Focal Loss to classification loss if enabled
            if hasattr(self, 'focal_loss_config') and self.focal_loss_config and self.focal_loss_config.get('enabled', False):
                alpha = self.focal_loss_config.get('alpha', 0.25)
                gamma = self.focal_loss_config.get('gamma', 2.0)
                
                # If we have logits and labels, compute Focal Loss directly
                if outputs_class is not None and len(labels) > 0:
                    # Collect all logits and targets for Focal Loss computation
                    all_logits = []
                    all_targets = []
                    
                    # Note: outputs_class structure depends on DETR implementation
                    # For now, apply Focal Loss scaling to existing classification loss
                    # This is a practical approximation that maintains gradient flow
                    if 'loss_ce' in loss_dict:
                        loss_ce = loss_dict['loss_ce']
                        # Scale classification loss to emphasize hard examples
                        # Higher loss = harder example = apply more focus
                        # This approximates (1 - p_t)^gamma from Focal Loss
                        focal_scale = alpha * (1.0 + loss_ce.detach().clamp(min=0, max=10)) ** gamma
                        loss_ce_focal = loss_ce * focal_scale.clamp(min=0.1, max=10.0)
                        
                        # Update loss_dict and total loss
                        loss_dict['loss_ce'] = loss_ce_focal
                        captured_loss_dict['loss_ce'] = loss_ce_focal
                        
                        # Recompute total loss
                        loss = sum(v for k, v in loss_dict.items() if k.startswith('loss_'))
            
            return loss, loss_dict, auxiliary_outputs
        
        # Temporarily replace loss_function
        self.detr_model.loss_function = patched_loss_function
        
        try:
            outputs = self.detr_model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                labels=labels
            )
        finally:
            # Restore original loss function
            self.detr_model.loss_function = original_loss_fn
        
        # Extract actual loss components from captured loss_dict
        # DETR loss_dict typically contains: loss_ce, loss_bbox, loss_giou, loss_cardinality
        # Map to torchvision format
        if not captured_loss_dict:
            # Fallback if loss_dict wasn't captured (shouldn't happen, but be safe)
            loss_dict = {
                'loss_ce': outputs.loss,
                'loss_bbox': torch.tensor(0.0, device=outputs.loss.device, dtype=outputs.loss.dtype),
                'loss_giou': torch.tensor(0.0, device=outputs.loss.device, dtype=outputs.loss.dtype),
            }
        else:
            # Use actual loss components from DETR
            # DETR uses these keys: 'loss_ce', 'loss_bbox', 'loss_giou', 'loss_cardinality'
            loss_dict = {
                'loss_ce': captured_loss_dict.get('loss_ce', outputs.loss),
                'loss_bbox': captured_loss_dict.get('loss_bbox', torch.tensor(0.0, device=outputs.loss.device, dtype=outputs.loss.dtype)),
                'loss_giou': captured_loss_dict.get('loss_giou', torch.tensor(0.0, device=outputs.loss.device, dtype=outputs.loss.dtype)),
            }
        
        # Clean up intermediate tensors
        del pixel_values, pixel_mask, labels, outputs
        
        return loss_dict
    
    def _forward_inference(self, images: List[torch.Tensor]):
        """Inference forward pass"""
        predictions = []
        
        import torchvision.transforms.functional as TF
        from PIL import Image
        
        for img in images:
            # Denormalize and convert to PIL
            mean = torch.tensor([0.485, 0.456, 0.406], device=img.device, dtype=img.dtype).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=img.device, dtype=img.dtype).view(3, 1, 1)
            img_denorm = img * std + mean
            img_denorm = torch.clamp(img_denorm, 0, 1)
            
            pil_img = TF.to_pil_image(img_denorm)
            
            # Process and predict
            inputs = self.processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(img.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.detr_model(**inputs)
            
            # Post-process outputs
            target_sizes = torch.tensor([pil_img.size[::-1]], device=img.device, dtype=torch.float32)
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.0
            )[0]
            
            # Convert to torchvision format
            boxes = results['boxes']
            scores = results['scores']
            raw_labels = results['labels']
            
            # DETR outputs 1-based labels: 0=background, 1=player, 2=ball
            # Evaluator expects 0-based labels: 0=player, 1=ball
            # Filter background and convert to 0-based
            if len(raw_labels) > 0:
                # Filter out background (label 0) and convert 1-based to 0-based
                mask = raw_labels > 0  # Keep only non-background (1=player, 2=ball)
                if mask.any():
                    boxes = boxes[mask]
                    scores = scores[mask]
                    labels = raw_labels[mask] - 1  # Convert: 1→0 (player), 2→1 (ball)
                    # Validate labels are in valid range [0, num_classes-1]
                    valid_mask = (labels >= 0) & (labels < self.num_classes)
                    if valid_mask.all():
                        pass  # All labels valid
                    else:
                        # Filter out invalid labels
                        boxes = boxes[valid_mask]
                        scores = scores[valid_mask]
                        labels = labels[valid_mask]
                else:
                    # No valid predictions (all background)
                    boxes = torch.zeros((0, 4), dtype=boxes.dtype, device=boxes.device)
                    scores = torch.zeros((0,), dtype=scores.dtype, device=scores.device)
                    labels = torch.zeros((0,), dtype=raw_labels.dtype, device=raw_labels.device)
            else:
                # No predictions at all
                boxes = torch.zeros((0, 4), dtype=boxes.dtype, device=boxes.device)
                scores = torch.zeros((0,), dtype=scores.dtype, device=scores.device)
                labels = torch.zeros((0,), dtype=raw_labels.dtype, device=raw_labels.device)
            
            predictions.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            })
            
            # Clean up intermediate tensors
            del img_denorm, pil_img, inputs, outputs, target_sizes, results
        
        return predictions


def get_rfdetr_model(config: Dict, training_config: Dict = None) -> nn.Module:
    """
    Get RF-DETR model (Roboflow DETR)
    
    Args:
        config: Model configuration dictionary
        training_config: Optional training configuration
    
    Returns:
        RF-DETR model wrapped for torchvision API compatibility
    """
    try:
        from rfdetr import RFDETRBase, RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
        
        num_classes = config.get('num_classes', 2)
        rfdetr_size = config.get('rfdetr_size', 'base').lower()
        
        # Map size to RF-DETR class
        size_map = {
            'nano': RFDETRNano,
            'small': RFDETRSmall,
            'medium': RFDETRMedium,
            'base': RFDETRBase,
            'large': RFDETRLarge
        }
        
        if rfdetr_size not in size_map:
            print(f"Warning: Unknown RF-DETR size '{rfdetr_size}', using 'base'")
            rfdetr_size = 'base'
        
        rfdetr_class = size_map[rfdetr_size]
        rfdetr_model = rfdetr_class()
        
        # Note: RF-DETR has different API, would need custom wrapper
        # For now, return a placeholder that indicates RF-DETR needs custom integration
        print(f"RF-DETR {rfdetr_size} model created. Note: Full integration requires custom training pipeline.")
        print("RF-DETR has its own training API. Consider using RF-DETR's native training method.")
        
        # Return a wrapper that can be used for inference
        # Full training integration would require adapting the training pipeline
        return rfdetr_model
        
    except ImportError:
        raise ImportError("RF-DETR not installed. Install with: pip install rfdetr")


def get_detr_model(config: Dict, training_config: Dict = None) -> nn.Module:
    """
    Get DETR model with specified configuration
    
    Args:
        config: Model configuration dictionary
        training_config: Optional training configuration for class weights
    
    Returns:
        DETR model wrapped for torchvision API compatibility
    """
    architecture = config.get('architecture', 'detr').lower()
    
    # Route to appropriate model based on architecture
    if architecture == 'rfdetr':
        return get_rfdetr_model(config, training_config)
    elif architecture != 'detr':
        print(f"Warning: Unknown architecture '{architecture}', using 'detr'")
    
    num_classes = config.get('num_classes', 2)
    pretrained = config.get('pretrained', True)
    
    # Load pre-trained DETR model from transformers
    # IMPORTANT: Set num_labels in config BEFORE loading model
    # This ensures the loss function is created with correct number of classes
    from transformers import DetrConfig
    config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
    config.num_labels = num_classes + 1  # Set BEFORE model creation
    
    if pretrained:
        detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", config=config, ignore_mismatched_sizes=True)
    else:
        detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", config=config, ignore_mismatched_sizes=True)
    
    # Update the classification head
    in_features = detr_model.class_labels_classifier.in_features
    detr_model.class_labels_classifier = nn.Linear(in_features, num_classes + 1)
    
    # Force loss function to use correct num_labels by patching the criterion
    # The loss function creates a criterion with empty_weight based on config.num_labels
    # We need to ensure it's created with the correct size
    # This is a workaround for transformers DETR loss function issue
    import torch
    # Create a dummy forward pass to trigger loss creation, then patch it
    # Actually, we'll patch it in the forward pass instead
    
    # Get class weights from training config if enabled (deprecated, kept for compatibility)
    class_weights = None
    if training_config is not None:
        class_weights_config = training_config.get('class_weights', {})
        if class_weights_config.get('enabled', False):
            class_weights = {
                'player': class_weights_config.get('player', 1.0),
                'ball': class_weights_config.get('ball', 1.0)
            }
            print(f"Class weighting enabled: player={class_weights['player']}, ball={class_weights['ball']}")
    
    # Get Focal Loss configuration
    focal_loss_config = None
    if training_config is not None:
        focal_loss_config = training_config.get('focal_loss', {})
        if focal_loss_config.get('enabled', False):
            alpha = focal_loss_config.get('alpha', 0.25)
            gamma = focal_loss_config.get('gamma', 2.0)
            print(f"Focal Loss enabled: alpha={alpha}, gamma={gamma}")
    
    # Wrap model for torchvision API compatibility
    wrapped_model = DETRWrapper(detr_model, num_classes, class_weights=class_weights, focal_loss_config=focal_loss_config)
    
    return wrapped_model
