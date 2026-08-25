# Cursor rules (docs mirror)

Product/agent rules also live under `.cursor/rules/*.mdc`. This folder holds **handover-specific** mappings that should stay in git docs (client machines, AnyDesk setup).

| File | Purpose |
|------|---------|
| [match4_camera_ids.mdc](match4_camera_ids.mdc) | Match 4 physical camera → P-code (catch Ubuntu handover) |
| [catch_client_credentials.mdc](catch_client_credentials.mdc) | Never put developer GitHub on Catch; HF OK for weights only |
| [mac_ssh_config_catch.example](mac_ssh_config_catch.example) | Mac `~/.ssh/config` snippet for `catch-soccer` |

Product-wide rules: `.cursor/rules/` — `catch_remote_compute.mdc`, `catch_mac_file_sync.mdc`, `phase1_batch_testing.mdc`, `catch_client_credentials.mdc`.  
Human specs: [CATCH_REMOTE_COMPUTE.md](../product/CATCH_REMOTE_COMPUTE.md) · [CATCH_MAC_FILE_SYNC.md](../product/CATCH_MAC_FILE_SYNC.md) · [PHASE1_BATCH_TESTING.md](../product/PHASE1_BATCH_TESTING.md).
