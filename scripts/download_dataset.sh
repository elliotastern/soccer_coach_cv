#!/bin/bash
# Helper script to download COCO dataset to local computer

ARCHIVE="/workspace/soccer_coach_cv_coco_dataset.tar.gz"
ARCHIVE_SIZE=$(du -h "$ARCHIVE" | cut -f1)

echo "📦 COCO Dataset Archive"
echo "======================"
echo "Location: $ARCHIVE"
echo "Size: $ARCHIVE_SIZE"
echo ""
echo "📥 Download Options:"
echo ""
echo "Option 1: Using SCP (from your local terminal):"
echo "  scp user@host:$ARCHIVE ./soccer_coach_cv_coco_dataset.tar.gz"
echo ""
echo "Option 2: Using SFTP:"
echo "  sftp user@host"
echo "  get $ARCHIVE"
echo ""
echo "Option 3: If using RunPod/Cloud IDE with web interface:"
echo "  - Navigate to /workspace/ in the file browser"
echo "  - Right-click on 'soccer_coach_cv_coco_dataset.tar.gz'"
echo "  - Select 'Download'"
echo ""
echo "📂 To extract on your local machine:"
echo "  tar -xzf soccer_coach_cv_coco_dataset.tar.gz"
echo ""
