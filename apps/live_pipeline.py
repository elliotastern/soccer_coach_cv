#!/usr/bin/env python3
"""Live RTSP pipeline entrypoint (Product Phase 2+).

Phase 1 delivery uses apps/batch_pipeline.py. This stub reserves the live
ingest path (RTSP, ~200 ms latency budget) without implementing Phase 2 yet.
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Live soccer coaching pipeline (Phase 2+)")
    parser.add_argument("--rtsp", type=str, help="RTSP URL")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    print(
        "Live RTSP pipeline is Product Phase 2+. "
        "Use: python apps/batch_pipeline.py --video <path> "
        f"(got rtsp={args.rtsp!r})",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
