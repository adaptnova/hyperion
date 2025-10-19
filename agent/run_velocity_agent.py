import argparse, torch, time, threading, os, sys
# Ensure Qwen-VL is found
qwen_vl_path = os.path.abspath('./Qwen-VL')
if qwen_vl_path not in sys.path: sys.path.insert(0, qwen_vl_path)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env')) # Load .env from /data/hyperion/.env

from transformers import AutoTokenizer, Qwen3VLMoeForConditionalGeneration, AutoConfig
import bitsandbytes as bnb
from fastapi import FastAPI, Request, HTTPException
from starlette.responses import JSONResponse
import uvicorn
import logging
import uuid
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union
import wandb
from safetensors.torch import save_file, load_file
import subprocess

# --- Logging Setup ---
LOG_DIR = "/data/hyperion/logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(os.path.join(LOG_DIR, "velocity.log")), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# --- MCC and Agent Classes (with Checkpointing) ---
class MetaCognitiveController:
    def decide_learning_params(self, state_metrics):
        logger.info("  [MCC] Deciding learning parameters...")
        learning_params = {"lr": 1e-6, "recursion_depth": 5} # Cautious default
        if state_metrics.get("teach_intensity", 0) > 0.5: # Example high intensity trigger
            logger.info("  [MCC] High teach intensity requested. Increasing learning.")
            learning_params["recursion_depth"] = 50
            learning_params["lr"] = 1e-5 # Tuned reduced peak LR
        return learning_params

class VelocityAgent:
    def __init__(self, args):
        self.mcc = MetaCognitiveController()
        self.device = "cuda"
        self.model_id = args.model_id
        self.checkpoint_dir = args.checkpoint_dir
        self.checkpoint_interval = args.checkpoint_interval
        self.hf_repo = args.hf_repo
        self.turn_count = 0 # Tracks learning iterations for checkpointing

        # --- W&B Init ---
        try:
            if os.getenv('WANDB_API_KEY'):
                 wandb.init(project="Project-Velocity", config=args, dir="/data/hyperion/logs")
                 logger.info("WandB initialized successfully.")
            else: logger.warning("WANDB_API_KEY not set. Skipping WandB initialization.")
        except Exception as e: logger.error(f"Failed to initialize WandB: {e}", exc_info=False)

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
        logger.info("[Agent] Initialization complete.")
        self.load_latest_checkpoint() # Attempt to load previous state


    # ... (get_plastic_params, freeze_non_plastic_params remain the same) ...
    def get_plastic_params(self, model):
        config = getattr(model.config, "text_config", model.config)
        total_layers = config.num_hidden_layers
        plastic_layer_start = int(total_layers * 0.65)
        params = []
        for name, param in model.named_parameters():
             if hasattr(param, 'requires_grad') and param.requires_grad and ".layers." in name:
                 try: layer_num = int(name.split(".layers.")[1].split(".")[0])
                 except (ValueError, IndexError): continue
                 if layer_num >= plastic_layer_start: params.append(param)
        if not params:
            logger.warning("No plastic parameters found. Defaulting to last layer.")
            last_layer_params = list(model.model.layers[-1].parameters())
            for p in last_layer_params: p.requires_grad = True
            return last_layer_params
        for p in params: p.requires_grad = True
        logger.info(f"Identified {len(params)} plastic parameters.")
        return params

    def freeze_non_plastic_params(self, model):
        plastic_set = set(self.plastic_params_list)
        frozen_count = 0; total_count = 0
        for param in model.parameters():
            total_count += 1
            if param not in plastic_set: param.requires_grad = False; frozen_count += 1
        logger.info(f"[Agent] Froze {frozen_count}/{total_count} parameters.")


    def generate_response(self, conversation_history: List[Dict[str, str]], max_new_tokens: int = 100) -> str:
        # ... (implementation as before) ...
        logger.info(f"[Agent] Generating response based on history (last msg: '{conversation_history[-1]['content'][:50]}...')")
        prompt = ""
        for message in conversation_history: prompt += f"{message['role'].capitalize()}: {message['content']}\n"
        prompt += "Assistant:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        logger.info(f"[Agent] Generated response: '{response[:100]}...'")
        return response


    def _perform_teach_in_background(self, text, iterations=1, learning_rate=5e-6):
        # ... (Learning loop with LR scheduling as before) ...
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
                if loss is None: logger.warning(f"Iter {i+1}/{iterations}: Loss is None."); continue
                loss.backward(); self.optimizer.step()
                current_loss = loss.item()
                avg_loss += current_loss
                logger.info(f"    -> BG Iteration {i+1}/{iterations} (LR: {current_lr:.2e})... Loss: {current_loss:.4f}")
                if wandb.run: wandb.log({"background_loss": current_loss, "background_lr": current_lr, "background_iteration": i+1})

            if iterations > 0:
                if wandb.run: wandb.log({"average_teach_loss": avg_loss / iterations, "teach_iterations": iterations})
                # --- Checkpoint Trigger ---
                self.turn_count += iterations
                if self.checkpoint_interval > 0 and self.turn_count >= self.checkpoint_interval:
                    self.save_checkpoint()
                    self.turn_count = 0 # Reset counter
        except Exception as e: logger.error(f"ERROR during refinement: {e}", exc_info=True)
        finally: logger.info(f"  [Agent Background Thread] Refinement complete.")


    def teach(self, text, iterations=1, learning_rate=5e-6):
        thread = threading.Thread(target=self._perform_teach_in_background, args=(text, iterations, learning_rate))
        thread.start()

    # --- Checkpointing Methods (Re-integrated) ---
    def save_checkpoint(self):
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
            if self.hf_repo: self._push_to_hf(checkpoint_path, ckpt_filename, optimizer_path, optim_filename)
        except Exception as e: logger.error(f"[Agent] Checkpoint save failed: {e}", exc_info=True)

    def _push_to_hf(self, ckpt_path, ckpt_name, optim_path, optim_name):
        logger.info(f"[Agent] Uploading checkpoint to Hugging Face Hub: {self.hf_repo}...")
        try:
            subprocess.run(["huggingface-cli", "upload", self.hf_repo, ckpt_path, ckpt_name], check=True, capture_output=True, text=True)
            subprocess.run(["huggingface-cli", "upload", self.hf_repo, optim_path, optim_name], check=True, capture_output=True, text=True)
            logger.info("[Agent] Hugging Face upload successful.")
        except subprocess.CalledProcessError as e: logger.error(f"[Agent] Hugging Face upload failed: {e.stderr}")
        except FileNotFoundError: logger.error("[Agent] HF upload failed: `huggingface-cli` not found.")
        except Exception as e: logger.error(f"[Agent] HF upload failed unexpectedly: {e}", exc_info=True)


    def load_latest_checkpoint(self):
        logger.info(f"Checking for checkpoints in: {self.checkpoint_dir}")
        try:
            if not os.path.exists(self.checkpoint_dir): return logger.info("Checkpoint dir missing.")
            checkpoint_files = sorted([f for f in os.listdir(self.checkpoint_dir) if f.startswith("velocity-evolving-") and f.endswith(".safetensors")], reverse=True)
            if checkpoint_files:
                latest_checkpoint = os.path.join(self.checkpoint_dir, checkpoint_files[0])
                timestamp = checkpoint_files[0].split('-')[-1].replace(".safetensors", "")
                latest_optimizer = os.path.join(self.checkpoint_dir, f"velocity-optimizer-{timestamp}.pt")
                logger.info(f"[Agent] Loading latest checkpoint: {latest_checkpoint}")
                loaded_state = load_file(latest_checkpoint, device="cpu")
                model_keys = {k for k, p in self.model.named_parameters() if p.requires_grad}
                filtered_state = {k: v for k, v in loaded_state.items() if k in model_keys}
                missing_keys, unexpected_keys = self.model.load_state_dict(filtered_state, strict=False)
                if unexpected_keys: logger.warning(f"Unexpected keys loading checkpoint: {unexpected_keys}")
                if os.path.exists(latest_optimizer):
                    logger.info(f"[Agent] Loading optimizer state: {latest_optimizer}")
                    optim_state = torch.load(latest_optimizer, map_location=self.device)
                    try: self.optimizer.load_state_dict(optim_state)
                    except Exception as oe: logger.error(f"Failed to load optimizer state: {oe}. Reinitializing.")
                else: logger.warning(f"Optimizer state not found for {timestamp}")
                logger.info("[Agent] Evolving checkpoint loaded.")
            else: logger.info("[Agent] No evolving checkpoints found.")
        except Exception as e: logger.error(f"[Agent] Failed to load checkpoint: {e}.", exc_info=True)


# --- FastAPI App and Endpoints ---
app = FastAPI(title="Velocity Agent API", version="0.3.0")
agent = None

# --- OpenAI Models --- (Add your data models here)
class ChatMessage(BaseModel): role: str; content: str
class ChatCompletionRequest(BaseModel): model: str; messages: List[ChatMessage]; max_tokens: int = 150
class ChatCompletionChoice(BaseModel): index: int; message: ChatMessage; finish_reason: str
class ChatCompletionResponse(BaseModel): id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}"); object: str = "chat.completion"; created: int = Field(default_factory=lambda: int(time.time())); model: str; choices: List[ChatCompletionChoice]

# --- API Endpoints ---
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    if not agent: raise HTTPException(status_code=503, detail="Agent not initialized")
    conversation_history = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    if not conversation_history: raise HTTPException(status_code=400, detail="No messages")
    response_text = agent.generate_response(conversation_history, max_new_tokens=request.max_tokens)
    last_user_message = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), None)
    if last_user_message:
        teach_intensity = 0.6 if len(last_user_message) < 50 else 0.1 # Simple heuristic
        learning_params = agent.mcc.decide_learning_params({"teach_intensity": teach_intensity})
        agent.teach(last_user_message, iterations=learning_params["recursion_depth"], learning_rate=learning_params["lr"])
        logger.info(f"Triggered background learning (Intensity: {teach_intensity}).")
        if wandb.run: wandb.log({"mcc_chosen_lr": learning_params["lr"], "mcc_chosen_iterations": learning_params["recursion_depth"]})
    response_message = ChatMessage(role="assistant", content=response_text)
    choice = ChatCompletionChoice(index=0, message=response_message, finish_reason="stop")
    return ChatCompletionResponse(model=agent.model_id, choices=[choice])

@app.get("/health")
async def health_check(): return {"status": "ok"}

# --- Main Execution ---
def main():
    global agent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-30B-A3B-Thinking")
    parser.add_argument("--anchor-checkpoint", required=True)
    parser.add_argument("--checkpoint-dir", default="/data/hyperion/checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--hf-repo", default="LevelUp2x/Hyperion")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    agent = VelocityAgent(args)

    logger.info(f"Starting Uvicorn server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", reload=False)

if __name__ == "__main__":
    main()
