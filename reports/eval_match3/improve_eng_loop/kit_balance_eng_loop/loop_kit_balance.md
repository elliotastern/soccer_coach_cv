# Kit balance eng-loop

## Diagnosis
- Crops already **62% blue>white**; **P10 74/26**, P8 ~50/50.
- High-conf labels are *more* blue-skewed. Trajectory cannot invent white mass.
- 474 dual-color torsos were labeled team0 (undershirt / mixed).

## Recommended path (confidence **0.90**)

`hard_center_50` + **dual_to_white** + **vote_last3**

| metric | before | after (`all__dual_white+vote3`) |
|---|---:|---:|
| share | 61.9/38.1 | 47.9/52.1 |
| off 50/50 | 11.9 pp | 2.1 pp |
| flips | 0.129 | 0.073 |
| coverage | 0.825 | 0.949 |

50/50 ceiling (higher flips): `dual_white+eq_frame` → ~49.9/50.1 but flip ~0.23 — treat as optional fused prior only.

## Why this works
Opposite-color undershirts make center crops show **both** blue and white fracs high; baseline assign pulls them to blue. Forcing those to white recovers the missing white mass; vote3 stabilizes tracks.

## Next implement
1. Productize `hard_center_50` in `jersey_feature` (flagged).
2. Dual-color rule in `assign_feature` / live label path.
3. Keep vote3 / existing traj sticky for flips.
4. Kit-ref: more white samples; optional P10 downweight in online fit.
