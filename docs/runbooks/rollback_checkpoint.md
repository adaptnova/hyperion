# Runbook: Checkpoint Rollback
Objective: Revert the agent's learned state to a previous trusted checkpoint from HF Hub.
Trigger: Unrecoverable identity drift, major capability regression, post-incident recovery.
Procedure:
1. Stop agent (see `restart_agent.md`).
2. Identify target checkpoint commit/files on HF Hub (`LevelUp2x/Hyperion`).
3. Download target `.safetensors` and `.pt` files to `/data/hyperion/checkpoints/`.
4. Ensure downloaded files are named according to the latest timestamp pattern (`velocity-evolving-YYYYMMDD-HHMMSS.safetensors`, `velocity-optimizer-YYYYMMDD-HHMMSS.pt`). Remove newer checkpoint files if necessary.
5. Restart agent (see `restart_agent.md`). The runner will load the latest available checkpoint.
6. Run validation probes to confirm state.
