#!/usr/bin/env python3
"""Backward-compatible entrypoint. Prefer: python apps/batch_pipeline.py"""
from apps.batch_pipeline import main

if __name__ == "__main__":
    main()
