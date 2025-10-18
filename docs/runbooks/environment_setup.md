# Runbook: Velocity Environment Setup (CUDA 12.8 Stable)

**Objective**: Provision a clean Ubuntu 24.04 VM with the correct CUDA, driver, and Python environment for Project Velocity.

**Trigger**: Need for a new development/testing VM.

**Procedure**:

1.  **Launch VM**: Provision a **bare Ubuntu 24.04 LTS VM** with at least one NVIDIA H100/H200 GPU. Ensure standard kernel (not pre-configured with NVIDIA drivers).
2.  **Run Provisioning Script**: SSH into the VM and execute the definitive setup script (stored in `/data/hyperion/infra/setup_vm_stable.sh` - *Note: We need to create this script based on our last successful manual run*). This script performs:
    * System updates and installs prerequisites (`build-essential`, `git`, `python3-pip`).
    * Installs `linux-headers-generic`.
    * Installs CUDA Toolkit 12.8 and `cuda-drivers-550` via NVIDIA's APT repository.
    * Configures CUDA environment variables (`/etc/profile.d/cuda.sh`).
    * Installs the Python AI toolchain using `pip --break-system-packages` (`torch==2.5.1+cu121`, `torchvision`, `torchaudio`, `transformers>=4.42.0`, `accelerate`, `bitsandbytes`, `fastapi[all]`, `sentence-transformers`).
    * Sets up the `/data/hyperion` directory structure.
3.  **Reboot**: The script will trigger a reboot.
4.  **Verification**: After reboot, log in and verify:
    * `nvidia-smi` (Shows Driver 550.xx, CUDA 12.8).
    * `python3 -c 'import torch; print(torch.cuda.is_available())'` (Prints `True`).
5.  **Clone Repository**: `cd /data && git clone https://github.com/adaptnova/hyperion.git`
6.  **Run Agent**: Follow the `restart_agent.md` runbook.

**Rollback**: If the script fails, destroy the VM and start again from Step 1.
