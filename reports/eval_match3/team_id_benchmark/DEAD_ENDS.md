# Team ID benchmark dead ends

| Attempt | Result | Notes |
|---------|--------|-------|
| Euclidean Lab features | Rejected | Collapsed on Match 3 kits (see team_label PROMPT v2) |
| Per-frame RGB K-Means batch | Replaced | Tracklet Golden Batch + team_core |
| Simple majority multi-cam vote | Replaced | Weighted vote (fisheye + cam distance) |
| FIFA 105×68 GK boxes | Rejected | Pitch 1 goal boxes via which_goal_box |

## Phase 4 (active in team_core)

- **Bhattacharyya** on hue histogram in `feature_distance()` — default ON
- **Mahalanobis-style kit outlier** in `is_photometric_outlier()` — default ON
- **CIEDE2000 / GMM grass** — deferred; run A/B via eval if precision < 85%
