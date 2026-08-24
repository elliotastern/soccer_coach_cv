# Cursor rules (docs mirror)

Product/agent rules also live under `.cursor/rules/*.mdc`. This folder holds **handover-specific** mappings that should stay in git docs (client machines, AnyDesk setup).

| File | Purpose |
|------|---------|
| [match4_camera_ids.mdc](match4_camera_ids.mdc) | Match 4 physical camera → P-code (catch Ubuntu handover) |
| [catch_client_credentials.mdc](catch_client_credentials.mdc) | Never put developer GitHub on Catch; HF OK for weights only |

Product-wide rules (including batch testing) live in `.cursor/rules/` — e.g. `phase1_batch_testing.mdc`. Human spec: [PHASE1_BATCH_TESTING.md](../product/PHASE1_BATCH_TESTING.md).
