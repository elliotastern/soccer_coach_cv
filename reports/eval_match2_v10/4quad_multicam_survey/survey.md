# 4quad multicam survey — selection by pitch region

Cams: `P1, P6, P7, P8, P10, P12, Cam4plus, Cam5plus`. No gold R/P here — **who wins** under thr rules.
Top Left lock under test: `p7_thr060_others030` ({'_default': 0.3, 'P7': 0.6}).

## Top Left (`top_left`)

| policy | n_selected | top cams (share) |
|---|---:|---|
| `max_conf_030` | 299/299 | Cam4plus 79%, P10 21% |
| `max_conf_060` | 299/299 | Cam4plus 79%, P10 21% |
| `p7_thr060_others030` | 299/299 | Cam4plus 79%, P10 21% |
| `cam4_thr060_others030` | 299/299 | Cam4plus 79%, P10 21% |
| `cam5_thr060_others030` | 299/299 | Cam4plus 79%, P10 21% |

## Top Right (`top_right`)

| policy | n_selected | top cams (share) |
|---|---:|---|
| `max_conf_030` | 299/299 | Cam4plus 47%, P7 29%, P1 23% |
| `max_conf_060` | 299/299 | Cam4plus 47%, P7 29%, P1 23% |
| `p7_thr060_others030` | 299/299 | Cam4plus 47%, P7 29%, P1 23% |
| `cam4_thr060_others030` | 299/299 | Cam4plus 47%, P7 29%, P1 23% |
| `cam5_thr060_others030` | 299/299 | Cam4plus 47%, P7 29%, P1 23% |

## Center Start (`center_start`)

| policy | n_selected | top cams (share) |
|---|---:|---|
| `max_conf_030` | 300/300 | Cam4plus 82%, P8 12%, P1 5%, Cam5plus 1% |
| `max_conf_060` | 300/300 | Cam4plus 82%, P8 12%, P1 5%, Cam5plus 1% |
| `p7_thr060_others030` | 300/300 | Cam4plus 82%, P8 12%, P1 5%, Cam5plus 1% |
| `cam4_thr060_others030` | 300/300 | Cam4plus 82%, P8 12%, P1 5%, Cam5plus 1% |
| `cam5_thr060_others030` | 300/300 | Cam4plus 82%, P8 12%, P1 5%, Cam5plus 1% |

## Bottom Right (`bottom_right`)

| policy | n_selected | top cams (share) |
|---|---:|---|
| `max_conf_030` | 300/300 | Cam4plus 58%, Cam5plus 27%, P8 15% |
| `max_conf_060` | 296/300 | Cam4plus 57%, Cam5plus 27%, P8 15% |
| `p7_thr060_others030` | 300/300 | Cam4plus 58%, Cam5plus 27%, P8 15% |
| `cam4_thr060_others030` | 300/300 | Cam4plus 57%, Cam5plus 28%, P8 15%, P1 0% |
| `cam5_thr060_others030` | 300/300 | Cam4plus 58%, Cam5plus 27%, P8 15% |

## Read

- If another slot’s winner is **Cam4plus/Cam5plus**, Top Left’s P7 floor does not transfer.
- Next gold pack should target the **#1 selected cam** on that slot (min labels).
- Only claim match-wide 80/90 after ≥2 regions have dual-cam gold + thr lock.

## Verdict

| Slot | Dominant cam @0.30 | Implication |
|---|---|---|
| Top Left | **Cam4plus 79%** (P10 21%) | Prior P7/P10 gold was a **P-cam-only** study; full 8-cam system here is Cam4plus-led |
| Top Right | Cam4plus 47%, P7 29%, P1 23% | Most mixed — needs multi-cam gold later |
| Center Start | **Cam4plus 82%** | Best single-cam gold target (least user work) |
| Bottom Right | Cam4plus 58%, Cam5plus 27% | Dual wide-cam |

Top Left `P7≥0.60` lock does **not** move these shares (winners already ≥0.60).

**Next (min labels):** build Cam4plus Center Start 300 pack + dense prelabels → human correct → rescore system vs 80/90 on that region.
