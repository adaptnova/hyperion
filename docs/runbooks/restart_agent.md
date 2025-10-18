# Runbook: Agent Restart
Objective: Safely stop and restart the Velocity agent service or background process.
Trigger: Deployment of new code, agent unresponsive, manual intervention needed.
Procedure (systemd):
1. `sudo systemctl stop velocity-agent.service`
2. `sudo systemctl status velocity-agent.service` (Verify inactive)
3. `cd /data/hyperion && gh repo sync` (Pull latest code if needed)
4. `sudo systemctl start velocity-agent.service`
5. `sudo systemctl status velocity-agent.service` (Verify active)
6. `tail -f /data/hyperion/logs/velocity.log` (Monitor startup)
Procedure (nohup):
1. `sudo pkill -f run_velocity_agent.py`
2. `cd /data/hyperion && gh repo sync` (Pull latest code if needed)
3. `nohup bash -c 'PYTHONPATH=/data/hyperion/Qwen-VL exec python3 agent/run_velocity_agent.py --anchor-checkpoint checkpoints/Velocity-Anchor-v1.safetensors' > logs/velocity.log 2>&1 &`
4. `tail -f /data/hyperion/logs/velocity.log` (Monitor startup)
