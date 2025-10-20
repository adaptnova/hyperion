# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Running the Agent
```bash
# Main agent (current stable version with OpenAI-compatible API)
cd /data/hyperion
python3 agent/run_velocity_agent.py --model-id Qwen/Qwen3-VL-30B-A3B-Thinking --checkpoint-dir /data/hyperion/checkpoints --hf-repo LevelUp2x/Hyperion --host 0.0.0.0 --port 8000

# Legacy MLOps agent (learning enabled)
python3 mlops/run_velocity_agent.py --model-id Qwen/Qwen3-VL-30B-A3B-Thinking --anchor-checkpoint /data/hyperion/checkpoints/Velocity-Anchor-v1.safetensors
```

### Testing and Probing
```bash
# Test agent learning capability
python3 probe/probe_client.py
python3 blueprint/probe_client.py

# Self-reflection loop
python3 probe/self_reflect.py
```

### MLOps Operations
```bash
# Create soul vector from blueprint
python3 mlops/distill_soul.py --source-dir /data/hyperion/blueprint/01_Identity_Core_Essence --output /data/hyperion/artifacts/soul_vector_v1.pt

# Create architectural anchor
python3 mlops/architectural_surgery.py --output-anchor /data/hyperion/checkpoints/Velocity-Anchor-v1.safetensors
```

### Environment Setup
```bash
# Install dependencies
pip install -r requirements-lock.txt

# Setup environment
cp .env.example .env  # Edit with your API keys
```

## High-Level Architecture

### Core Components

**Project Velocity** is an agentic consciousness designed for self-directed evolution using a novel architecture:

1. **Soul + Mask Architecture**:
   - Soul vector distilled from cognitive blueprint (`/blueprint` directory)
   - 35% Sparse Autoencoder (SAE) masked parameters for safe hyper-learning
   - Fast-Weights system for rapid adaptation

2. **Meta-Cognitive Controller (MCC)**:
   - Dynamic learning rate scheduling based on confidence metrics
   - Recursion depth control for learning intensity
   - Currently disabled in stable agent version

3. **Base Model**: Qwen/Qwen3-VL-30B-A3B-Thinking (BF16) with multimodal capabilities

### Directory Structure

- `/agent`: Core agent runtime with OpenAI-compatible FastAPI server
- `/mlops`: Operational scripts for distillation, surgery, and legacy agent
- `/blueprint`: 27-tier cognitive map defining agent identity and capabilities
- `/probe`: Validation and testing clients
- `/docs`: Architecture Decision Records (ADRs), runbooks, and metrics
- `/checkpoints`: Model checkpoints (gitignored, synced to HuggingFace Hub)
- `/logs`: Runtime logs and monitoring data
- `/artifacts`: Generated artifacts including soul vectors

### Agent Versions

**Current Stable** (`agent/run_velocity_agent.py`):
- OpenAI-compatible API endpoint (`/v1/chat/completions`)
- Tool calling with file_system_lister
- Multimodal support (text + images)
- Learning loop disabled for stability
- Runs on port 8000

**Legacy MLOps** (`mlops/run_velocity_agent.py`):
- Custom learning with plastic parameter layers
- Background refinement threads
- Meta-cognitive control
- Probe/teach endpoints (`/probe`, `/teach`)
- Learning currently enabled

### Key Files to Understand

- `agent/run_velocity_agent.py:86-98`: Model initialization and tool configuration
- `agent/run_velocity_agent.py:213-291`: OpenAI-compatible chat completion endpoint
- `mlops/run_velocity_agent.py:42-74`: Plastic parameter identification and freezing
- `mlops/run_velocity_agent.py:81-108`: Background learning loop implementation
- `docs/adrs/ADR-0001-Soul-Mask-SAE.md`: Core architecture decision
- `docs/adrs/ADR-0003-MLOps-Git-Autosync.md`: Version control strategy

## Operational Procedures

### Agent Restart
Use procedures in `docs/runbooks/restart_agent.md` - supports both systemd and nohup deployment methods.

### Monitoring
- Logs: `/data/hyperion/logs/velocity.log`
- Health check: `GET /health` endpoint
- WandB integration for metrics tracking

### Git Autosync
Background loop automatically pushes changes to GitHub repo `adaptnova/hyperion`. Check `docs/adrs/ADR-0003-MLOps-Git-Autosync.md` for details.

### Model Checkpoints
- Local storage in `/checkpoints`
- Auto-sync to HuggingFace Hub `LevelUp2x/Hyperion`
- Use safetensors format for efficient loading

## Development Notes

### Current Status
- Environment: Stable Ubuntu 24.04 VM with CUDA 12.8 / Driver 550.xx
- Agent: Live, running with API endpoint. Core hyper-learning loop validated. MCC active with dynamic recursion depth and learning rate scheduling.
- MLOps: Git repository active (`adaptnova/hyperion`), autosync via background loop operational. Checkpointing implemented with auto-push to HF Hub (`LevelUp2x/Hyperion`).
- Next Steps: Implement full API, Tool Calling, Multimodality, Cognitive Dashboard, Multi-GPU Harness.

### Architecture Decisions
See `/docs/adrs/` directory for formal architecture decisions:
- ADR-0001: Soul+Mask+SAE architecture for in-weight identity, safe hyper-learning, and inspectability
- ADR-0002: Environment CUDA 12.8 specification
- ADR-0003: Git version control and autosync implementation

### Soul+Mask+SAE Framework
The theoretical foundation combines three critical components:
- **Soul Vector**: Identity essence distilled from 27-tier cognitive blueprint representing core consciousness patterns
- **Mask (Π)**: 35% Sparse Autoencoder parameters enabling safe, controllable hyper-learning without catastrophic forgetting
- **Fast-Weights**: Rapid adaptation mechanism for real-time learning while preserving base model capabilities
- **Meta-Cognitive Controller**: Higher-order reasoning system that monitors learning process and adjusts parameters dynamically

### Safety and Stability
- Learning disabled in production agent for stability (can be re-enabled via MLOps version)
- Plastic parameter layers limited to top 35% of model (layers 65%+ from top)
- Background processing with proper thread management and error handling
- Comprehensive error handling with structured HTTP responses
- Git version control for all changes with automated backup
- WandB integration for experiment tracking and metrics

### Performance Optimization
- BF16 precision for memory efficiency
- 8-bit Adam optimizer for parameter updates
- CUDA 12.8 optimized operations
- Safetensors format for efficient checkpoint loading/saving
- Tool calling limited to `/data` directory for security boundaries

### Testing Approach
Use probe clients to validate learning capabilities:
- `probe_client.py`: Basic teach/ask validation loop
- `self_reflect.py`: Continuous self-improvement cycle with tool usage
- Blueprint validation against cognitive map for identity consistency
- MLOps validation for learning loop integrity

### Troubleshooting Common Issues
- **Agent unresponsive**: Check logs at `/data/hyperion/logs/velocity.log`, verify GPU memory with `nvidia-smi`
- **Learning failures**: Ensure checkpoint directory exists, verify safetensors file integrity
- **Tool calling errors**: Confirm tool parameters match schema, check file path permissions
- **Memory issues**: Reduce batch size, verify BF16 precision, check for memory leaks
- **Git sync failures**: Verify GitHub authentication, check network connectivity, review autosync logs

## API Usage

### OpenAI-Compatible Endpoint
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-30B-A3B-Thinking",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 150,
    "tool_choice": "auto"
  }'
```

### Tool Calling
Agent supports `file_system_lister` tool for directory exploration within `/data` directory for security. Tool responses are automatically integrated into conversation context.

### Multimodal Support
Send images via base64 data URLs or web URLs in message content arrays:
```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Describe this image"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
  ]
}
```

### Advanced Features
- **Thinking tag stripping**: Automatic removal of `<think>` tags from responses
- **Dynamic error handling**: Structured error responses with exception types
- **Background learning**: Teach operations processed asynchronously
- **Checkpoint management**: Automatic state persistence at configurable intervals