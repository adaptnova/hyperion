# ADR-0002: Standardize on CUDA 12.8 Environment
Status: Accepted
Context: Multiple failed attempts to provision a stable environment using bleeding-edge CUDA 13.0 and NVIDIA Driver 580.xx on Ubuntu 24.04. Errors included packaging conflicts (`apt`), driver build failures (`dkms`), and installer flag issues (`.run`).
Decision: Revert to a stable, known-good configuration: CUDA Toolkit 12.8 with NVIDIA Driver 550.xx series on a bare Ubuntu 24.04 LTS VM. Use the hybrid installation method (driver via `.run`, toolkit via `apt` without driver deps) or the pure `apt` method if repository issues are resolved. Use PyTorch wheels built for `cu121`.
Consequences:
* (+) Establishes a reliable, reproducible baseline environment, unblocking agent development.
* (-) Delays experimentation with CUDA 13.0 features.
* Action: Create `environment_setup.md` runbook for this stable configuration.
* Future: Revisit CUDA 13.0 as a dedicated R&D task once the agent is more mature.
