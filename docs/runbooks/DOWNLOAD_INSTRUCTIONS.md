# Download COCO Dataset to Local Computer

## Archive Created ✅

**File:** `/workspace/soccer_coach_cv_coco_dataset.tar.gz`  
**Size:** ~5.0 GB (compressed from 5.5 GB)  
**Contains:** Complete COCO format dataset (train + val splits)

## Download Methods

### Method 1: SCP (Command Line)

From your **local terminal**, run:

```bash
scp user@your-server:/workspace/soccer_coach_cv_coco_dataset.tar.gz ./
```

Replace:
- `user@your-server` with your actual SSH connection details
- If using RunPod, it might be: `root@your-pod-ip`

### Method 2: SFTP

```bash
sftp user@your-server
get /workspace/soccer_coach_cv_coco_dataset.tar.gz
exit
```

### Method 3: Web Interface (RunPod/Cloud IDE)

1. Open the file browser in your web interface
2. Navigate to `/workspace/`
3. Find `soccer_coach_cv_coco_dataset.tar.gz`
4. Right-click → Download

### Method 4: VS Code Remote

If using VS Code with Remote SSH:
1. Open VS Code Remote Explorer
2. Navigate to `/workspace/soccer_coach_cv_coco_dataset.tar.gz`
3. Right-click → Download

## Extract on Local Machine

After downloading, extract the archive:

```bash
tar -xzf soccer_coach_cv_coco_dataset.tar.gz
```

This will create:
```
soccer_coach_cv_coco_dataset/
├── train/
│   ├── images/          (1,977 images)
│   └── annotations/
│       └── annotations.json
└── val/
    ├── images/          (1,012 images)
    └── annotations/
        └── annotations.json
```

## Dataset Contents

- **Training:** 1,977 images, 10,973 annotations
- **Validation:** 1,012 images, 5,596 annotations
- **Classes:** `player` (class 0), `ball` (class 1)
- **Format:** COCO JSON format
- **Image size:** 1920x1080

## Verification

After extraction, verify the dataset:

```bash
ls soccer_coach_cv_coco_dataset/train/images/ | wc -l  # Should show 1977
ls soccer_coach_cv_coco_dataset/val/images/ | wc -l    # Should show 1012
```
