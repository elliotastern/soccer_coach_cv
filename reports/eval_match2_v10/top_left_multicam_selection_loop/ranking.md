# Top Left multicam — selection fix loop (dual gold)

Cache re-score only (no re-detect). Goal **R≥0.8 P≥0.9** on P7∪P10-selected.

**Best:** `max_conf_070` → P=0.975 R=0.975 **HIT**

| Rank | id | P | R | F1 | P7 P/R | P10 P/R | covered | goal |
|---:|---|---:|---:|---:|---|---|---:|---|
| 1 | `max_conf_070` | 0.975 | 0.975 | 0.975 | 0.939/0.939 | 0.989/0.989 | 120 | HIT |
| 2 | `p7_thr070_others030` | 0.939 | 0.939 | 0.939 | 0.939/0.939 | 0.938/0.938 | 163 | HIT |
| 3 | `max_conf_060` | 0.933 | 0.933 | 0.933 | 0.862/0.862 | 0.980/0.980 | 163 | HIT |
| 4 | `p7_thr060_others030` | 0.915 | 0.915 | 0.915 | 0.862/0.862 | 0.943/0.943 | 188 | HIT |
| 5 | `p7_060_prefer_p10_margin10` | 0.911 | 0.911 | 0.911 | 0.842/0.842 | 0.941/0.941 | 192 | HIT |
| 6 | `max_conf_050` | 0.846 | 0.859 | 0.852 | 0.720/0.740 | 0.980/0.980 | 208 | MISS |
| 7 | `p7_thr050_others030` | 0.840 | 0.851 | 0.846 | 0.720/0.740 | 0.949/0.949 | 225 | MISS |
| 8 | `p7_050_prefer_p10` | 0.834 | 0.845 | 0.840 | 0.637/0.662 | 0.940/0.940 | 229 | MISS |
| 9 | `max_conf_040` | 0.775 | 0.800 | 0.787 | 0.641/0.679 | 0.946/0.946 | 253 | MISS |
| 10 | `prefer_p10_margin10` | 0.754 | 0.788 | 0.771 | 0.577/0.631 | 0.946/0.946 | 272 | MISS |
| 11 | `size_weighted_030` | 0.752 | 0.784 | 0.768 | 0.530/0.581 | 0.947/0.947 | 246 | MISS |
| 12 | `max_conf_030` | 0.750 | 0.785 | 0.767 | 0.601/0.652 | 0.948/0.948 | 268 | MISS |
| 13 | `prefer_p10` | 0.750 | 0.785 | 0.767 | 0.520/0.577 | 0.940/0.940 | 272 | MISS |
| 14 | `prefer_p10_margin05` | 0.750 | 0.785 | 0.767 | 0.588/0.640 | 0.944/0.944 | 272 | MISS |
| 15 | `soft_min2_015` | 0.744 | 0.776 | 0.760 | 0.594/0.639 | 0.948/0.948 | 270 | MISS |
| 16 | `gold_cams_only_030` | 0.724 | 0.759 | 0.741 | 0.561/0.609 | 0.950/0.950 | 283 | MISS |
| 17 | `prefer_gold_then_max` | 0.724 | 0.759 | 0.741 | 0.561/0.609 | 0.950/0.950 | 283 | MISS |

## Read

Several variants **HIT** the dual-gold proxy (P≥0.9 and R≥0.8 on P7∪P10-selected frames).

**Caveat:** raising thr shrinks `covered` (frames where a gold cam wins). `max_conf_070` is best P/R but only **120** frames vs baseline **268**. That is not full-window ball recall.

**Practical lock for live path:** `p7_thr060_others030` — P=R=**0.915**, HIT, **188** covered frames (best coverage among HITs that don’t need P10 preference hacks). Alt: `p7_060_prefer_p10_margin10` (192 covered, 0.911).

Prefer-P10 / soft-consensus / gold-only @0.30 do **not** beat raising the bar on weak P7 boxes.

**Next:** wire `P7≥0.60` (others @0.30) into the live multicam pick; then 5090 latency. Optional: score true all-frame R including empty emits (stricter than this proxy).
