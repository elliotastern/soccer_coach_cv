"""
Script to validate training dataset structure and format.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.dataset_loader import validate_dataset, get_dataset_info


def main():
    """Validate datasets in workspace."""
    workspace_datasets = Path("/workspace/datasets")
    
    if not workspace_datasets.exists():
        print(f"❌ Dataset directory not found: {workspace_datasets}")
        print(f"\n📁 Create it with:")
        print(f"   mkdir -p /workspace/datasets/train/images")
        print(f"   mkdir -p /workspace/datasets/train/annotations")
        print(f"   mkdir -p /workspace/datasets/val/images")
        print(f"   mkdir -p /workspace/datasets/val/annotations")
        return
    
    print("🔍 Validating training datasets...\n")
    
    for split in ['train', 'val']:
        dataset_path = workspace_datasets / split
        print(f"📊 {split.upper()} Dataset:")
        print(f"   Path: {dataset_path}")
        
        is_valid, issues = validate_dataset(str(dataset_path))
        
        if is_valid:
            info = get_dataset_info(str(dataset_path))
            print(f"   ✅ Valid")
            print(f"   📸 Images: {info['images_count']}")
            print(f"   🏷️  Annotations: {info['annotations_count']}")
            print(f"   📦 Categories: {info['categories_count']}")
            if info['categories']:
                print(f"   🎯 Classes: {', '.join(info['categories'])}")
        else:
            print(f"   ❌ Issues found:")
            for issue in issues:
                print(f"      - {issue}")
        
        print()


if __name__ == "__main__":
    main()
