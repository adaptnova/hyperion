import argparse, torch, time, threading, os, sys
from transformers import AutoTokenizer, Qwen3VLMoeForConditionalGeneration, AutoConfig
import bitsandbytes as bnb
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetaCognitiveController:
    """Decides how to learn based on agent state."""
    def decide_learning_params(self, state_metrics):
        logger.info("  [MCC] Deciding learning parameters...")
        # Start cautious
        learning_params = {"lr": 1e-6, "recursion_depth": 5}
        # Increase intensity if prior belief was confident
        if state_metrics.get("confidence", 0) > 0.5:
            logger.info("  [MCC] High-confidence prior detected. Increasing learning intensity.")
            learning_params["recursion_depth"] = 50
            learning_params["lr"] = 5e-5 # Increase LR significantly
        return learning_params

class VelocityAgent:
    def __init__(self, args):
        self.mcc = MetaCognitiveController()
        self.device = "cuda"
        logger.info(f"[Agent] Initializing on device: {self.device}")
        logger.info(f"[Agent] Loading model: {args.model_id}...")
        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            args.model_id, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        plastic_params = self.get_plastic_params(self.model)
        self.freeze_non_plastic_params(self.model)
        logger.info(f"[Agent] Plasticity mask defined.")
        self.optimizer = bnb.optim.AdamW8bit(plastic_params, lr=5e-6) # Initial LR set here, but overridden in teach
        logger.info("[Agent] Initialization complete. API server starting.")

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
        plastic_set = set(self.get_plastic_params(model))
        frozen_count = 0
        total_count = 0
        for param in model.parameters():
            total_count += 1
            if param not in plastic_set:
                param.requires_grad = False
                frozen_count += 1
        logger.info(f"[Agent] Froze {frozen_count}/{total_count} parameters.")

    def ask(self, question):
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=50, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _perform_teach_in_background(self, text, iterations=1, learning_rate=5e-6):
        """The actual learning loop with LR scheduling."""
        logger.info(f"  [Agent Background Thread] Performing {iterations} recursive refinement steps...")
        logger.info(f"  [Agent Background Thread] Starting LR: {learning_rate}")
        initial_lr = learning_rate
        final_lr = initial_lr / 10 # Decay to 1/10th
        lr_decay_step = (initial_lr - final_lr) / max(1, iterations - 1)
        try:
            for i in range(iterations):
                # Calculate and apply current LR for this iteration
                current_lr = initial_lr - (i * lr_decay_step)
                for group in self.optimizer.param_groups:
                    group["lr"] = current_lr

                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(**inputs, labels=inputs.input_ids.clone())
                loss = outputs.loss
                if loss is None:
                     logger.warning(f"    -> BG Iteration {i+1}/{iterations} (LR: {current_lr:.2e})... Loss is None. Skipping.")
                     continue
                loss.backward()
                self.optimizer.step()
                logger.info(f"    -> BG Iteration {i+1}/{iterations} (LR: {current_lr:.2e})... Loss: {loss.item():.4f}")
        except Exception as e:
             logger.error(f"  [Agent Background Thread] ERROR during refinement: {e}", exc_info=True)
        finally:
            logger.info(f"  [Agent Background Thread] Refinement complete.")

    def teach(self, text, iterations=1, learning_rate=5e-6):
        """Kicks off the learning process in a background thread."""
        thread = threading.Thread(target=self._perform_teach_in_background, args=(text, iterations, learning_rate))
        thread.start()

app = FastAPI()
agent = None

@app.post("/probe")
async def handle_probe(request: Request):
    data = await request.json()
    return {"response": agent.ask(data.get("question"))}

@app.post("/teach")
async def handle_teach(request: Request):
    data = await request.json()
    learning_params = agent.mcc.decide_learning_params(data.get("metrics", {}))
    # Pass both iterations and LR to the teach method
    agent.teach(data.get("text"), iterations=learning_params["recursion_depth"], learning_rate=learning_params["lr"])
    return JSONResponse(status_code=202, content={"status": "processing_in_background", "iterations": learning_params["recursion_depth"], "lr": learning_params["lr"]})

def main():
    global agent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-30B-A3B-Thinking")
    parser.add_argument("--anchor-checkpoint", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    agent = VelocityAgent(args)
    # Ensure Qwen-VL source is in PYTHONPATH
    qwen_vl_path = os.path.abspath('./Qwen-VL')
    if qwen_vl_path not in sys.path:
        sys.path.insert(0, qwen_vl_path)
        logger.info(f"Added {qwen_vl_path} to PYTHONPATH")
    logger.info(f"Starting Uvicorn server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
