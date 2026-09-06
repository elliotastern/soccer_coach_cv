"""Team ID strategy presets for grid eval (Match 4 / Match 3)."""
from __future__ import annotations

from dataclasses import dataclass

from src.perception.team_core import (
    AUTO_BLUE_FRAC,
    AUTO_WHITE_FRAC,
    KIT_MODE_AUTO,
    KIT_MODE_MATCH3,
    STICKY_FLIP_CONF_AUTO,
)

STICKY_MATCH3 = 0.78
# Kit-ref: raise sticky so labeled runs don't flicker worse than online-fit.
KIT_REF_STICKY_FLIP_CONF = 0.95


@dataclass(frozen=True)
class TeamStrategy:
    id: str
    name: str
    kit_mode: str = KIT_MODE_AUTO
    use_session: bool = True
    per_frame_only: bool = False
    legacy_rgb_top10: bool = False
    use_jersey_pixels: bool = True
    use_rebalance: bool = True
    use_skew_cap: bool = True
    use_symmetric_ema: bool = True
    pixel_cluster_agree: bool = False
    use_tracklet_golden: bool = False
    golden_frames: int = 240
    auto_blue_frac: float = AUTO_BLUE_FRAC
    auto_white_frac: float = AUTO_WHITE_FRAC
    sticky_flip_conf: float | None = None
    no_gray: bool = False
    use_traj_vote: bool = False
    soft_pixel_nudge: bool = False


def production_default() -> TeamStrategy:
    return STRATEGIES["auto_traj_no_gray"]


def sticky_conf(strat: TeamStrategy) -> float:
    if strat.sticky_flip_conf is not None:
        return strat.sticky_flip_conf
    if strat.kit_mode == KIT_MODE_MATCH3:
        return STICKY_MATCH3
    return STICKY_FLIP_CONF_AUTO


STRATEGIES: dict[str, TeamStrategy] = {
    "legacy_rgb_top10": TeamStrategy(
        id="S01",
        name="legacy_rgb_top10",
        use_session=False,
        legacy_rgb_top10=True,
        use_jersey_pixels=False,
        use_rebalance=False,
        use_skew_cap=False,
    ),
    "core_per_frame": TeamStrategy(
        id="S02",
        name="core_per_frame",
        per_frame_only=True,
        use_jersey_pixels=False,
        use_rebalance=False,
        use_skew_cap=False,
    ),
    "match3_session": TeamStrategy(
        id="S03",
        name="match3_session",
        kit_mode=KIT_MODE_MATCH3,
        use_jersey_pixels=True,
        use_rebalance=False,
        use_skew_cap=False,
        use_symmetric_ema=False,
    ),
    "auto_cluster_only": TeamStrategy(
        id="S04",
        name="auto_cluster_only",
        use_jersey_pixels=False,
        use_rebalance=False,
        use_skew_cap=False,
    ),
    "auto_v2_rebalance": TeamStrategy(
        id="S05",
        name="auto_v2_rebalance",
        use_jersey_pixels=False,
        use_rebalance=True,
        use_skew_cap=False,
    ),
    "auto_traj_no_gray": TeamStrategy(
        id="S11",
        name="auto_traj_no_gray",
        use_jersey_pixels=True,
        use_rebalance=False,
        use_skew_cap=False,
        no_gray=True,
        use_traj_vote=True,
        soft_pixel_nudge=True,
        sticky_flip_conf=0.92,
    ),
    "auto_v3_pixels": TeamStrategy(
        id="S06",
        name="auto_v3_pixels",
        use_jersey_pixels=True,
        use_rebalance=True,
        use_skew_cap=False,
    ),
    "auto_v3_skew_cap": TeamStrategy(
        id="S07",
        name="auto_v3_skew_cap",
        use_jersey_pixels=True,
        use_rebalance=True,
        use_skew_cap=True,
    ),
    "auto_pixels_tuned": TeamStrategy(
        id="S08",
        name="auto_pixels_tuned",
        use_jersey_pixels=True,
        use_rebalance=True,
        use_skew_cap=True,
        auto_blue_frac=0.26,
        auto_white_frac=0.24,
    ),
    "auto_pixel_cluster_agree": TeamStrategy(
        id="S09",
        name="auto_pixel_cluster_agree",
        use_jersey_pixels=True,
        use_rebalance=True,
        use_skew_cap=True,
        pixel_cluster_agree=True,
    ),
    "tracklet_golden": TeamStrategy(
        id="S10",
        name="tracklet_golden",
        use_jersey_pixels=True,
        use_rebalance=True,
        use_skew_cap=True,
        use_tracklet_golden=True,
    ),
}


def session_from_config(cfg: dict | None, run_dir: Path | str | None = None) -> "TeamSession":
    """Build TeamSession from configs/default.yaml team_assignment block."""
    from pathlib import Path as _Path

    from src.review.team_live import TeamSession

    ta = (cfg or {}).get("team_assignment") or {}
    kit_mode = str(ta.get("kit_mode", KIT_MODE_AUTO))
    strat_key = ta.get("strategy")
    if strat_key and strat_key in STRATEGIES:
        sess = TeamSession(strategy=STRATEGIES[strat_key])
    else:
        sess = TeamSession(kit_mode=kit_mode)
    centroids_path = ta.get("kit_centroids_path")
    if centroids_path:
        path = _Path(centroids_path)
    elif run_dir:
        path = _Path(run_dir) / "team_centroids.json"
    else:
        path = None
    if path is not None and path.is_file():
        sess.load_centroids_file(path)
    return sess


def list_strategies(names: str) -> list[TeamStrategy]:
    if names == "all":
        return list(STRATEGIES.values())
    out = []
    id_map = {s.id: s for s in STRATEGIES.values()}
    for key in names.split(","):
        key = key.strip()
        if key in STRATEGIES:
            out.append(STRATEGIES[key])
        elif key in id_map:
            out.append(id_map[key])
    return out
