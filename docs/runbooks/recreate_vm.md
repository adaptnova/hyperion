# Runbook: VM Recreation
Objective: Provision, configure, and restore a Velocity VM using cloud-init.
Trigger: VM loss (spot instance preemption, hardware failure).
Procedure:
1. Launch new Ubuntu 24.04 VM with `/data/hyperion/infra/cloud-init.yaml`.
2. Wait for cloud-init completion and reboot.
3. Verify `nvidia-smi` and `torch.cuda.is_available()`.
4. Check `systemctl status velocity-agent.service`.
5. Run validation probe.
