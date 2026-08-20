# Event Manager - Aggregates and manages events
from __future__ import annotations

import json
import os
from typing import List

import pandas as pd

from src.export.schema import events_to_csv_rows, events_to_json, get_csv_schema
from src.state.types import Event, MatchData


class EventManager:
    """Manages event aggregation and checkpointing."""

    def __init__(self, checkpoint_interval: int = 300, output_dir: str = "data/output"):
        self.checkpoint_interval = checkpoint_interval
        self.output_dir = output_dir
        self.events: List[Event] = []
        self.frame_count = 0
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    def tick_frame(self, frame_id: int):
        """Call once per processed video frame for checkpoint cadence."""
        self.frame_count = int(frame_id) + 1
        if self.frame_count % self.checkpoint_interval == 0:
            self.save_checkpoint()

    def add_events(self, events: List[Event]):
        self.events.extend(events)

    def save_checkpoint(self):
        checkpoint_path = os.path.join(
            self.output_dir,
            "checkpoints",
            f"checkpoint_frame_{self.frame_count}.json",
        )
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(
                {"frame_count": self.frame_count, "events": events_to_json(self.events)},
                f,
                indent=2,
            )

    def save_final_output(
        self, match_id: str, csv_path: str = None, json_path: str = None
    ):
        if csv_path is None:
            csv_path = os.path.join(self.output_dir, "events.csv")
        if json_path is None:
            json_path = os.path.join(self.output_dir, "events.json")

        match_data = MatchData(
            match_id=match_id,
            events=self.events,
            metadata={"total_frames": self.frame_count},
        )
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "match_id": match_id,
                    "events": events_to_json(self.events),
                    "metadata": match_data.metadata,
                },
                f,
                indent=2,
            )

        rows = events_to_csv_rows(self.events)
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        pd.DataFrame(rows, columns=get_csv_schema()).to_csv(csv_path, index=False)

    def get_events(self) -> List[Event]:
        return self.events
