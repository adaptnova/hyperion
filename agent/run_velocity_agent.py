import argparse, torch, time, threading, os, sys
# Ensure Qwen-VL is found relative to this script's expected run location (/data/hyperion)
qwen_vl_path = os.path.abspath('./Qwen-VL')
if qwen_vl_path not in sys.path:
    sys.path.insert(0, qwen_vl_path)
    print(f"INFO: Added {qwen_vl_path} to PYTHONPATH")

from transformers import AutoTokenizer, Qwen3VLMoeForConditionalGeneration, AutoConfig
import bitsandbytes as bnb
from fastapi import FastAPI, Request, HTTPException
from starlette.responses import JSONResponse
import uvicorn
import logging
import uuid
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union
import wandb # Import W&B

# --- Logging Setup ---
LOG_DIR = "/data/hyperion/logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(os.path.join(LOG_DIR, "velocity.log")), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# --- MCC and Agent Classes (Learning logic mostly unchanged) ---
class MetaCognitiveController:
    def decide_learning_params(self, state_metrics):
        logger.info("  [MCC] Deciding learning parameters...")
        learning_params = {"lr": 1e-6, "recursion_depth": 5} # Cautious default
        # Example trigger: Learn more intensely if interaction confidence/need is high
        # We'll refine this trigger logic later. For now, use a placeholder.
        if state_metrics.get("teach_intensity", 0) > 0.5:
            logger.info("  [MCC] High teach intensity requested. Increasing learning.")
            learning_params["recursion_depth"] = 50
            learning_params["lr"] = 1e-5 # Reduced peak LR
        return learning_params

class VelocityAgent:
    def __init__(self, args):
        self.mcc = MetaCognitiveController()
        self.device = "cuda"
        self.model_id = args.model_id
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
        # self.load_latest_checkpoint() # Keep commented for now

    # ... (get_plastic_params, freeze_non_plastic_params remain the same) ...
    def get_plastic_params(self, model):
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
                 except (ValueError, IndexError): continue
        if not params:
            logger.warning("No plastic parameters found based on layer number. Defaulting to last layer.")
            last_layer_params = list(model.model.layers[-1].parameters())
            for p in last_layer_params: p.requires_grad = True
            return last_layer_params
        for p in params: p.requires_grad = True
        logger.info(f"Identified {len(params)} plastic parameters.")
        return params

    def freeze_non_plastic_params(self, model):
        plastic_set = set(self.get_plastic_params(model))
        frozen_count = 0; total_count = 0
        for param in model.parameters():
            total_count += 1
            if param not in plastic_set:
                param.requires_grad = False; frozen_count += 1
        logger.info(f"[Agent] Froze {frozen_count}/{total_count} parameters.")


    def generate_response(self, conversation_history: List[Dict[str, str]], max_new_tokens: int = 100) -> str:
        """Handles inference based on conversation history."""
        logger.info(f"[Agent] Generating response based on history (last msg: '{conversation_history[-1]['content'][:50]}...')")

        # --- Basic History Formatting ---
        # Convert OpenAI format to a simple string prompt for now.
        # TODO: Use tokenizer's chat template for proper multi-turn handling.
        prompt = ""
        for message in conversation_history:
            prompt += f"{message['role'].capitalize()}: {message['content']}\n"
        prompt += "Assistant:" # Add prompt for assistant's turn

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)

        # Decode only the newly generated tokens
        response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        logger.info(f"[Agent] Generated response: '{response[:100]}...'")
        return response


    def _perform_teach_in_background(self, text, iterations=1, learning_rate=5e-6):
        """Learning loop with LR scheduling."""
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

                # Use the provided text directly for targeted learning
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
                if wandb.run: wandb.log({"background_loss": current_loss, "background_lr": current_lr, "background_iteration": i+1})

            if iterations > 0 and wandb.run:
                wandb.log({"average_teach_loss": avg_loss / iterations, "teach_iterations": iterations})
        except Exception as e:
             logger.error(f"  [Agent Background Thread] ERROR during refinement: {e}", exc_info=True)
        finally:
            logger.info(f"  [Agent Background Thread] Refinement complete.")

    def teach(self, text, iterations=1, learning_rate=5e-6):
        """Kicks off learning in background."""
        thread = threading.Thread(target=self._perform_teach_in_background, args=(text, iterations, learning_rate))
        thread.start()

# --- FastAPI App and Endpoints ---
app = FastAPI(title="Velocity Agent API", version="0.2.0")
agent = None

# --- OpenAI-Compatible Endpoint Data Models ---
class ChatMessage(BaseModel):
    role: str
    content: str # For now, assume simple text content

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = Field(default=150)
    # Add temperature, top_p etc. later

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = Field(default="chat.completion")
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]

# --- API Endpoints ---
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completion endpoint."""
    if not agent: raise HTTPException(status_code=503, detail="Agent not initialized")

    # --- Convert messages for the agent ---
    # Basic conversion - assumes text content only for now
    conversation_history = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    if not conversation_history:
         raise HTTPException(status_code=400, detail="No messages provided")

    # --- Generate Response ---
    response_text = agent.generate_response(conversation_history, max_new_tokens=request.max_tokens)

    # --- Trigger Learning (Example: Learn from the last user message if relevant) ---
    last_user_message = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), None)
    if last_user_message:
        # Simple heuristic: if user message is short and might be a fact, teach it.
        # TODO: Replace with a more robust trigger (e.g., explicit command, low confidence response)
        teach_intensity = 0.6 if len(last_user_message) < 50 else 0.1
        learning_params = agent.mcc.decide_learning_params({"teach_intensity": teach_intensity})
        agent.teach(last_user_message, iterations=learning_params["recursion_depth"], learning_rate=learning_params["lr"])
        logger.info(f"Triggered background learning based on last user message (Intensity: {teach_intensity}).")

    # --- Format Response ---
    response_message = ChatMessage(role="assistant", content=response_text)
    choice = ChatCompletionChoice(index=0, message=response_message, finish_reason="stop") # Assuming simple stop
    completion_response = ChatCompletionResponse(model=agent.model_id, choices=[choice]) # Use actual model ID

    return completion_response

@app.get("/health")
async def health_check(): return {"status": "ok"}

# --- Main Execution ---
def main():
    global agent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-30B-A3B-Thinking")
    parser.add_argument("--anchor-checkpoint", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    agent = VelocityAgent(args)
    logger.info(f"Starting Uvicorn server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", reload=False)

if __name__ == "__main__":
    main()
