# (Insert the final, correct run_velocity_agent.py content with updated paths if needed)
# Example: checkpoint_dir might need updating if not passed via arg
import argparse, torch, time, threading, os, sys
# Ensure Qwen-VL is found relative to this script's expected run location (/data/hyperion)
qwen_vl_path = os.path.abspath('./Qwen-VL') # Assumes running from /data/hyperion
if qwen_vl_path not in sys.path:
    sys.path.insert(0, qwen_vl_path)
    print(f"INFO: Added {qwen_vl_path} to PYTHONPATH") # Use print before logging setup

from transformers import AutoTokenizer, Qwen3VLMoeForConditionalGeneration, AutoConfig
import bitsandbytes as bnb
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse
import uvicorn
import logging
from safetensors.torch import save_file, load_file # Import load_file
import subprocess
import wandb # Import W&B

# Configure logging
LOG_DIR = "/data/hyperion/logs" # Updated path
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "velocity.log")),
        logging.StreamHandler() # Also print to console
    ]
)
logger = logging.getLogger(__name__)

class MetaCognitiveController:
    def decide_learning_params(self, state_metrics):
        logger.info("  [MCC] Deciding learning parameters...")
        learning_params = {"lr": 1e-6, "recursion_depth": 5}
        if state_metrics.get("confidence", 0) > 0.5:
            logger.info("  [MCC] High-confidence prior detected. Increasing learning intensity.")
            learning_params["recursion_depth"] = 50
            learning_params["lr"] = 5e-5
        return learning_params

class VelocityAgent:
    def __init__(self, args):
        self.mcc = MetaCognitiveController()
        self.device = "cuda"
        self.model_id = args.model_id
        self.checkpoint_dir = args.checkpoint_dir
        self.checkpoint_interval = args.checkpoint_interval
        self.hf_repo = args.hf_repo
        self.turn_count = 0
        self.anchor_checkpoint_path = args.anchor_checkpoint # Path to conceptual anchor

        # --- W&B Initialization ---
        try:
            wandb.init(project="Project-Velocity", config=args, dir="/data/hyperion/logs") # Log wandb locally
            logger.info("WandB initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}", exc_info=True)


        logger.info(f"[Agent] Initializing on device: {self.device}")
        logger.info(f"[Agent] Loading model: {self.model_id}...")
        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.plastic_params_list = self.get_plastic_params(self.model)
        self.freeze_non_plastic_params(self.model)
        logger.info(f"[Agent] Plasticity mask defined.")
        self.optimizer = bnb.optim.AdamW8bit(self.plastic_params_list, lr=5e-6) # Base LR
        logger.info("[Agent] Initialization complete. API server starting.")
        self.load_latest_checkpoint()

    def get_plastic_params(self, model):
        # ... (implementation as before) ...
        config = getattr(model.config, "text_config", model.config)
        total_layers = config.num_hidden_layers
        plastic_layer_start = int(total_layers * 0.65)
        params = []
        for name, param in model.named_parameters():
             if hasattr(param, 'requires_grad') and param.requires_grad and ".layers." in name:
                 try:
                    layer_num = int(name.split(".layers.")[1].split(".")[0])
                    if layer_num >= plastic_layer_start:
                        params.append(param)
                 except (ValueError, IndexError):
                     continue
        if not params:
            logger.warning("No plastic parameters found based on layer number. Defaulting to last layer.")
            last_layer_params = list(model.model.layers[-1].parameters())
            for p in last_layer_params: p.requires_grad = True
            return last_layer_params
        for p in params: p.requires_grad = True
        logger.info(f"Identified {len(params)} plastic parameters.")
        return params


    def freeze_non_plastic_params(self, model):
        # ... (implementation as before) ...
        plastic_set = set(self.plastic_params_list)
        frozen_count = 0
        total_count = 0
        for param in model.parameters():
            total_count += 1
            if param not in plastic_set:
                param.requires_grad = False
                frozen_count += 1
        logger.info(f"[Agent] Froze {frozen_count}/{total_count} parameters.")


    def ask(self, question):
        # ... (implementation as before) ...
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=50, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


    def _perform_teach_in_background(self, text, iterations=1, learning_rate=5e-6):
        logger.info(f"  [Agent Background Thread] Performing {iterations} recursive refinement steps...")
        logger.info(f"  [Agent Background Thread] Starting LR: {learning_rate}")
        initial_lr = learning_rate
        final_lr = initial_lr / 10
        lr_decay_step = (initial_lr - final_lr) / max(1, iterations - 1)
        try:
            avg_loss = 0
            for i in range(iterations):
                current_lr = initial_lr - (i * lr_decay_step)
                for group in self.optimizer.param_groups: group["lr"] = current_lr

                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(**inputs, labels=inputs.input_ids.clone())
                loss = outputs.loss
                if loss is None:
                     logger.warning(f"    -> BG Iteration {i+1}/{iterations} (LR: {current_lr:.2e})... Loss is None. Skipping.")
                     continue
                loss.backward()
                self.optimizer.step()
                current_loss = loss.item()
                avg_loss += current_loss
                logger.info(f"    -> BG Iteration {i+1}/{iterations} (LR: {current_lr:.2e})... Loss: {current_loss:.4f}")
                # --- W&B Logging ---
                if wandb.run: wandb.log({"background_loss": current_loss, "background_lr": current_lr, "background_iteration": i+1})


            # --- Checkpoint Trigger ---
            self.turn_count += iterations
            if self.checkpoint_interval > 0 and self.turn_count >= self.checkpoint_interval:
                self.save_checkpoint()
                self.turn_count = 0

            # --- Log average loss for the batch ---
            if iterations > 0 and wandb.run:
                wandb.log({"average_teach_loss": avg_loss / iterations})

        except Exception as e:
             logger.error(f"  [Agent Background Thread] ERROR during refinement: {e}", exc_info=True)
        finally:
            logger.info(f"  [Agent Background Thread] Refinement complete.")

    def teach(self, text, iterations=1, learning_rate=5e-6):
        # ... (starts background thread as before) ...
        thread = threading.Thread(target=self._perform_teach_in_background, args=(text, iterations, learning_rate))
        thread.start()


    def save_checkpoint(self):
        # ... (implementation as before, logs INFO level) ...
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        ckpt_filename = f"velocity-evolving-{timestamp}.safetensors"
        optim_filename = f"velocity-optimizer-{timestamp}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, ckpt_filename)
        optimizer_path = os.path.join(self.checkpoint_dir, optim_filename)

        logger.info(f"[Agent] Saving checkpoint locally to {checkpoint_path}...")
        try:
            state_dict_to_save = {k: v for k, v in self.model.state_dict().items() if v.requires_grad}
            save_file(state_dict_to_save, checkpoint_path)
            torch.save(self.optimizer.state_dict(), optimizer_path)
            logger.info("[Agent] Local checkpoint save successful.")

            if self.hf_repo:
                logger.info(f"[Agent] Uploading checkpoint to Hugging Face Hub: {self.hf_repo}...")
                try:
                    subprocess.run(["huggingface-cli", "upload", self.hf_repo, checkpoint_path, ckpt_filename], check=True, capture_output=True, text=True)
                    subprocess.run(["huggingface-cli", "upload", self.hf_repo, optimizer_path, optim_filename], check=True, capture_output=True, text=True)
                    logger.info("[Agent] Hugging Face upload successful.")
                except subprocess.CalledProcessError as e: logger.error(f"[Agent] Hugging Face upload failed: {e.stderr}")
                except FileNotFoundError: logger.error("[Agent] Hugging Face upload failed: `huggingface-cli` not found.")
        except Exception as e: logger.error(f"[Agent] Checkpoint process failed: {e}", exc_info=True)


    def load_latest_checkpoint(self):
        # ... (implementation as before, uses load_file) ...
        logger.info(f"Checking for checkpoints in: {self.checkpoint_dir}")
        try:
            if not os.path.exists(self.checkpoint_dir):
                logger.info("Checkpoint directory does not exist. Starting from anchor.")
                return

            checkpoint_files = sorted(
                [f for f in os.listdir(self.checkpoint_dir) if f.startswith("velocity-evolving-") and f.endswith(".safetensors")],
                reverse=True
            )
            if checkpoint_files:
                latest_checkpoint = os.path.join(self.checkpoint_dir, checkpoint_files[0])
                timestamp = checkpoint_files[0].replace("velocity-evolving-", "").replace(".safetensors", "")
                latest_optimizer = os.path.join(self.checkpoint_dir, f"velocity-optimizer-{timestamp}.pt")

                logger.info(f"[Agent] Loading latest evolving checkpoint: {latest_checkpoint}")
                # Use safetensors load_file
                loaded_state = load_file(latest_checkpoint, device="cpu") # Load to CPU first
                # Filter state for keys present in the current model that require grad
                model_keys = {k for k, p in self.model.named_parameters() if p.requires_grad}
                filtered_state = {k: v for k, v in loaded_state.items() if k in model_keys}

                missing_keys, unexpected_keys = self.model.load_state_dict(filtered_state, strict=False)
                if unexpected_keys: logger.warning(f"Unexpected keys found while loading checkpoint: {unexpected_keys}")
                # Don't strictly need to report missing_keys if strict=False, but good for debug
                # if missing_keys: logger.warning(f"Missing keys not loaded: {missing_keys}")


                if os.path.exists(latest_optimizer):
                    logger.info(f"[Agent] Loading optimizer state: {latest_optimizer}")
                    # Load optimizer state to appropriate device (usually matches model params)
                    optim_state = torch.load(latest_optimizer, map_location=self.device)
                    try:
                       self.optimizer.load_state_dict(optim_state)
                    except Exception as oe:
                       logger.error(f"Failed to load optimizer state, possibly due to parameter mismatch: {oe}. Reinitializing optimizer.")
                       # Reinitialize optimizer if loading fails
                       self.optimizer = bnb.optim.AdamW8bit(self.plastic_params_list, lr=5e-6)

                else:
                     logger.warning(f"Optimizer state not found for checkpoint {timestamp}")

                logger.info("[Agent] Evolving checkpoint loaded successfully.")
            else:
                logger.info("[Agent] No evolving checkpoints found. Starting from base model state (conceptual anchor).")
        except Exception as e:
            logger.error(f"[Agent] Failed to load latest checkpoint: {e}. Starting from base model state.", exc_info=True)


app = FastAPI()
agent = None
# --- (MCC Class definition as before) ---
class MetaCognitiveController:
    def decide_learning_params(self, state_metrics):
        logger.info("  [MCC] Deciding learning parameters...")
        learning_params = {"lr": 1e-6, "recursion_depth": 5}
        if state_metrics.get("confidence", 0) > 0.5:
            logger.info("  [MCC] High-confidence prior detected. Increasing learning intensity.")
            learning_params["recursion_depth"] = 50
            learning_params["lr"] = 5e-5
        return learning_params

# ... (API endpoint definitions remain the same) ...
@app.post("/probe")
async def handle_probe(request: Request):
    data = await request.json()
    return {"response": agent.ask(data.get("question"))}

@app.post("/teach")
async def handle_teach(request: Request):
    data = await request.json()
    learning_params = agent.mcc.decide_learning_params(data.get("metrics", {}))
    agent.teach(data.get("text"), iterations=learning_params["recursion_depth"], learning_rate=learning_params["lr"])
    return JSONResponse(status_code=202, content={"status": "processing_in_background", "iterations": learning_params["recursion_depth"], "lr": learning_params["lr"]})


def main():
    global agent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-30B-A3B-Thinking")
    # Point to the conceptual anchor created by surgery script
    parser.add_argument("--anchor-checkpoint", default="/data/hyperion/checkpoints/Velocity-Anchor-v1.safetensors")
    parser.add_argument("--checkpoint-dir", default="/data/hyperion/checkpoints") # Updated path
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--hf-repo", default="LevelUp2x/Hyperion")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    agent = VelocityAgent(args)

    logger.info(f"Starting Uvicorn server on {args.host}:{args.port}")
    # Pass reload=False for production/stable background execution
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", reload=False)


if __name__ == "__main__":
    main()
