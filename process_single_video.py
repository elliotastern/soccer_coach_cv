#!/usr/bin/env python3
"""
Process a single video file through the auto-ingest pipeline
Useful for testing without the watchdog
"""
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from auto_ingest import VideoHandler, load_config, get_default_config
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) < 2:
        print("Usage: python process_single_video.py <video_path>")
        print("Example: python process_single_video.py data/raw/real_data/video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)
    
    logger.info(f"Processing video: {video_path}")
    
    # Load config
    try:
        config = load_config()
    except Exception as e:
        logger.warning(f"Could not load config, using defaults: {e}")
        config = get_default_config()
    
    # Override processed dir to not move the original file
    config['paths']['processed_dir'] = './processed_test'
    
    # Check if model exists
    model_path = config['model']['checkpoint_path']
    if not os.path.exists(model_path):
        logger.error(f"Model checkpoint not found: {model_path}")
        logger.error("Please train a model first or update the checkpoint_path in configs/auto_ingest.yaml")
        logger.error("You can also set MODEL_PATH environment variable")
        sys.exit(1)
    
    # Create handler and process
    try:
        handler = VideoHandler(config)
        handler.process_video(video_path)
        logger.info("Processing complete!")
    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
