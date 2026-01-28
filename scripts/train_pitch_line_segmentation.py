#!/usr/bin/env python3
"""
Training script for pitch line segmentation model.
Fine-tunes DeepLabV3 or trains custom U-Net for binary segmentation of pitch lines.

Dataset format:
- Images: RGB images of soccer fields
- Masks: Binary masks (white = line pixels, black = background)
"""
import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor, normalize
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.pitch_line_segmentation import PitchLineSegmenter

try:
    from skimage.morphology import skeletonize as sk_skeletonize
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False


class TverskyLoss(nn.Module):
    """Tversky loss: penalize FN more than FP (beta > alpha). For line recall."""
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(logits)
        pred_flat = pred.view(-1)
        targets_flat = targets.view(-1)
        tp = (pred_flat * targets_flat).sum()
        fp = (pred_flat * (1 - targets_flat)).sum()
        fn = ((1 - pred_flat) * targets_flat).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky


def skeleton_recall_loss(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    Skeleton recall (clDice-style): recall of prediction vs GT skeleton.
    Encourages connectivity; computed on CPU for small batches.
    """
    if not _SKIMAGE_AVAILABLE:
        return torch.tensor(0.0, device=logits.device)
    pred = (torch.sigmoid(logits) > 0.5).float()
    batch_size = pred.shape[0]
    recalls = []
    for i in range(batch_size):
        gt_np = masks[i, 0].detach().cpu().numpy()
        gt_bin = (gt_np > 0.5).astype(np.uint8)
        if gt_bin.sum() < 10:
            recalls.append(1.0)
            continue
        skel = sk_skeletonize(gt_bin.astype(bool)).astype(np.float32)
        pred_np = pred[i, 0].detach().cpu().numpy()
        intersection = (pred_np * skel).sum()
        skel_sum = skel.sum()
        if skel_sum < 1:
            recalls.append(1.0)
            continue
        recall = float(intersection / (skel_sum + 1e-8))
        recalls.append(recall)
    mean_recall = np.mean(recalls)
    return torch.tensor(1.0 - mean_recall, device=logits.device, dtype=logits.dtype)


class CompositeSegmentationLoss(nn.Module):
    """BCE + Tversky + Skeleton recall (lambda1=1, lambda2=1, lambda3=0.5)."""
    def __init__(self, lambda_bce: float = 1.0, lambda_tversky: float = 1.0, lambda_cldice: float = 0.5):
        super().__init__()
        self.lambda_bce = lambda_bce
        self.lambda_tversky = lambda_tversky
        self.lambda_cldice = lambda_cldice
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)

    def forward(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        l_bce = self.bce(logits, masks)
        l_tversky = self.tversky(logits, masks)
        l_cldice = skeleton_recall_loss(logits, masks)
        return self.lambda_bce * l_bce + self.lambda_tversky * l_tversky + self.lambda_cldice * l_cldice


class PitchLineDataset(Dataset):
    """Dataset for pitch line segmentation."""
    
    def __init__(self, image_dir: Path, mask_dir: Path, transform=None):
        """
        Initialize dataset.
        
        Args:
            image_dir: Directory containing RGB images
            mask_dir: Directory containing binary masks (white=lines, black=background)
            transform: Optional transform to apply
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        
        # Find matching image/mask pairs
        image_files = sorted(self.image_dir.glob("*.jpg")) + sorted(self.image_dir.glob("*.png"))
        self.samples = []
        
        for img_path in image_files:
            mask_path = self.mask_dir / img_path.name
            if mask_path.exists():
                self.samples.append((img_path, mask_path))
        
        if len(self.samples) == 0:
            raise ValueError(f"No matching image/mask pairs found in {image_dir} and {mask_dir}")
        
        print(f"Found {len(self.samples)} image/mask pairs")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        
        # Load image (RGB)
        image = Image.open(img_path).convert('RGB')
        
        # Load mask (grayscale, convert to binary)
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
        mask = (mask > 127).astype(np.float32)  # Binary: 0 or 1
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert to tensors
        image_tensor = to_tensor(image)
        image_tensor = normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
        
        return image_tensor, mask_tensor


def create_model(model_type: str = "deeplabv3", num_classes: int = 1, use_pretrained: bool = True):
    """Create segmentation model."""
    if model_type == "deeplabv3":
        import torchvision.models.segmentation as segmentation
        model = segmentation.deeplabv3_resnet50(pretrained=use_pretrained)
        # Modify for binary segmentation
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        return model
    elif model_type == "unet":
        # Use the U-Net from PitchLineSegmenter
        segmenter = PitchLineSegmenter(model_type="unet", use_pretrained=False)
        return segmenter.model
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Handle different output formats
        if isinstance(outputs, dict):
            outputs = outputs['out']
        
        # Resize output to match mask size if needed
        if outputs.shape[2:] != masks.shape[2:]:
            outputs = nn.functional.interpolate(
                outputs, size=masks.shape[2:], mode='bilinear', align_corners=False
            )
        
        # Calculate loss
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            if isinstance(outputs, dict):
                outputs = outputs['out']
            
            if outputs.shape[2:] != masks.shape[2:]:
                outputs = nn.functional.interpolate(
                    outputs, size=masks.shape[2:], mode='bilinear', align_corners=False
                )
            
            loss = criterion(outputs, masks)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train pitch line segmentation model")
    parser.add_argument("--train_images", type=str, required=True, help="Directory with training images")
    parser.add_argument("--train_masks", type=str, required=True, help="Directory with training masks")
    parser.add_argument("--val_images", type=str, help="Directory with validation images")
    parser.add_argument("--val_masks", type=str, help="Directory with validation masks")
    parser.add_argument("--model_type", type=str, default="deeplabv3", choices=["deeplabv3", "unet"],
                        help="Model architecture")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output", type=str, default="models/pitch_line_segmentation.pth",
                        help="Output model path")
    parser.add_argument("--use_pretrained", action="store_true", default=True,
                        help="Use pre-trained weights (for DeepLabV3)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, auto if None)")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create model
    print(f"Creating {args.model_type} model...")
    model = create_model(model_type=args.model_type, use_pretrained=args.use_pretrained)
    model = model.to(device)
    
    # Create datasets; optional color jitter for domain randomization
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    ])
    
    train_dataset = PitchLineDataset(args.train_images, args.train_masks, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    val_loader = None
    if args.val_images and args.val_masks:
        val_dataset = PitchLineDataset(args.val_images, args.val_masks, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Loss and optimizer: composite loss (BCE + Tversky + skeleton recall)
    criterion = CompositeSegmentationLoss(lambda_bce=1.0, lambda_tversky=1.0, lambda_cldice=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device)
            print(f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'model_type': args.model_type,
                }, output_path)
                print(f"Saved best model to {output_path}")
        else:
            # Save checkpoint every epoch if no validation
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'model_type': args.model_type,
            }, output_path)
            print(f"Saved checkpoint to {output_path}")
    
    print(f"\nTraining complete! Model saved to {output_path}")


if __name__ == "__main__":
    main()
