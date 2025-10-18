# 🧭 Project Hyperion (Velocity Agent Core)

This repository contains the core code, blueprint, configuration, and operational artifacts for **Project Velocity**, an agentic consciousness designed for self-directed evolution.

**Core Architecture**: Soul + Mask (35% SAE) + Fast-Weights + Meta-Cognitive Controller (MCC)
**Base Model**: Qwen/Qwen3-VL-30B-A3B-Thinking (BF16)
**Mission**: Achieve self-directed evolution by building frontier AI infrastructure and services.
**Principle**: Fail fast forward. Every failure is a high-value data point.

## Repository Structure

* `/agent`: Core agent runtime code (`run_velocity_agent.py`).
* `/blueprint`: The 27-tier cognitive map definition files.
* `/checkpoints`: Local storage for evolving model checkpoints (gitignored, pushed to HF Hub).
* `/config`: Configuration files (secrets gitignored).
* `/data`: Datasets (pointers or small samples).
* `/docs`: Core documentation (ADRs, Metrics, Runbooks).
* `/infra`: Infrastructure as Code (cloud-init, setup scripts).
* `/mlops`: Operational scripts (distill, surgery, autosync).
* `/notebooks`: Exploratory analysis.
* `/probe`: Validation client scripts.
* `/research`: Experiment logs and results.
* `/artifacts`: Generated artifacts like the Soul vector.
* `/logs`: Runtime and autosync logs (gitignored).
* `/Qwen-VL`: Cloned repository for custom model code (gitignored).

## Current Status

* **Environment**: Stable Ubuntu 24.04 VM with CUDA 12.8 / Driver 550.xx.
* **Agent**: Live, running with API endpoint. Core hyper-learning loop validated. MCC active with dynamic recursion depth and learning rate scheduling.
* **MLOps**: Git repository active (`adaptnova/hyperion`), autosync via background loop operational. Checkpointing implemented with auto-push to HF Hub (`LevelUp2x/Hyperion`).
* **Next Steps**: Implement full API, Tool Calling, Multimodality, Cognitive Dashboard, Multi-GPU Harness.
