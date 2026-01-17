"""
DETR Model Setup for Training
"""
import torch
import torch.nn as nn
from transformers import DetrForObjectDetection, DetrImageProcessor
from typing import Dict, List
import torch.nn.functional as F


class DETRWrapper(nn.Module):
    """Wrapper to make transformers DETR compatible with torchvision API"""
    
    def __init__(self, detr_model, num_classes: int):
        super().__init__()
        self.detr_model = detr_model
        self.num_classes = num_classes
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
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
                # Convert to center format
                center_x = (boxes[:, 0] + boxes[:, 2]) / 2
                center_y = (boxes[:, 1] + boxes[:, 3]) / 2
                width = boxes[:, 2] - boxes[:, 0]
                height = boxes[:, 3] - boxes[:, 1]
                boxes_center = torch.stack([center_x, center_y, width, height], dim=1)
                
                # Normalize by image size (assuming 1333x1333 after resize)
                img_size = 1333.0
                boxes_center = boxes_center / img_size
                
                labels.append({
                    'class_labels': target['labels'] + 1,  # Add 1 for background class
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
        
        def patched_loss_function(logits, labels, device, pred_boxes, config, outputs_class=None, outputs_coord=None, **kwargs):
            """Patched loss function that ensures correct num_labels"""
            # CRITICAL: Set num_labels to number of object classes (not including background)
            # The criterion will create empty_weight of size num_classes + 1
            config.num_labels = self.num_classes
            
            # Call original loss function - it creates a new criterion each time
            # The criterion's empty_weight is created based on config.num_labels
            loss, loss_dict, auxiliary_outputs = original_loss_fn(
                logits, labels, device, pred_boxes, config, outputs_class, outputs_coord, **kwargs
            )
            
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
        
        # Convert loss to torchvision format
        loss_dict = {
            'loss_ce': outputs.loss,
            'loss_bbox': torch.tensor(0.0, device=outputs.loss.device, dtype=outputs.loss.dtype),  # Combined in loss
            'loss_giou': torch.tensor(0.0, device=outputs.loss.device, dtype=outputs.loss.dtype),  # Combined in loss
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
            labels = results['labels'] - 1  # Remove background offset
            
            predictions.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            })
            
            # Clean up intermediate tensors
            del img_denorm, pil_img, inputs, outputs, target_sizes, results
        
        return predictions


def get_detr_model(config: Dict) -> nn.Module:
    """
    Get DETR model with specified configuration
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        DETR model wrapped for torchvision API compatibility
    """
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
    
    # Wrap model for torchvision API compatibility
    wrapped_model = DETRWrapper(detr_model, num_classes)
    
    return wrapped_model
