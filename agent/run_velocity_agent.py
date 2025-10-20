from __future__ import annotations
import argparse, torch, time, threading, os, sys, json, subprocess
from dotenv import load_dotenv

# --- Load Environment & Add Custom Code ---
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

from transformers import AutoTokenizer, AutoConfig
import bitsandbytes as bnb
from fastapi import FastAPI, Request, HTTPException
from starlette.responses import JSONResponse
import uvicorn
import logging
import uuid
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union, Optional
import wandb
from safetensors.torch import save_file, load_file

from qwen_agent.llm import get_chat_model # Use the factory
from qwen_agent.llm.schema import Message, ContentItem, FunctionCall
from qwen_agent.tools.base import BaseTool, register_tool # Correct import path
from qwen_agent.agents import Assistant

# --- Logging Setup ---
LOG_DIR = "/data/hyperion/logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(os.path.join(LOG_DIR, "velocity.log")), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# --- Define Tools ---
@register_tool('file_system_lister')
class FileSystemLister(BaseTool):
    """A tool to list files in a specified directory."""
    description = "Lists all files and directories in a specified directory."
    parameters = [{
        'name': 'path', 'type': 'string',
        'description': 'The directory path to list (e.g., /data/hyperion/docs)',
        'required': True
    }]
    def call(self, params: str, **kwargs) -> str:
        try:
            logger.info(f"[Tool Call] Executing file_system_lister with params: {params}")
            args = json.loads(params)
            path = args.get('path', '/data/hyperion') # Default to hyperion root
            if not os.path.abspath(path).startswith('/data'):
                raise Exception("Access denied: Path is outside the allowed /data directory.")
            result = subprocess.run(['ls', '-la', path], capture_output=True, text=True, check=True)
            return json.dumps({"status": "success", "listing": result.stdout})
        except Exception as e:
            logger.error(f"[Tool Call] Error executing ls: {e}")
            return json.dumps({"status": "error", "message": str(e)})

# --- MCC and Agent Classes ---
class MetaCognitiveController:
    # Placeholder: Learning logic is currently disabled.
    def decide_learning_params(self, state_metrics):
        logger.info("  [MCC] Deciding learning parameters... (Learning Disabled)")
        return {"lr": 0, "recursion_depth": 0}

class VelocityAgent:
    def __init__(self, args):
        self.mcc = MetaCognitiveController()
        self.device = "cuda"
        self.model_id = args.model_id
        self.checkpoint_dir = args.checkpoint_dir
        self.hf_repo = args.hf_repo

        # --- W&B Init ---
        try:
            if os.getenv('WANDB_API_KEY'):
                 wandb.init(project="Project-Velocity", config=args, dir="/data/hyperion/logs", reinit=True)
                 logger.info("WandB initialized successfully.")
            else: logger.warning("WANDB_API_KEY not set. Skipping WandB initialization.")
        except Exception as e: logger.error(f"Failed to initialize WandB: {e}", exc_info=False)

        logger.info(f"[Agent] Initializing on device: {self.device}")
        
        # --- Corrected Model Loading (per Stack Summary) ---
        llm_config = {
            "model_type": "transformers",
            "model": self.model_id,
            "device": self.device,
            "dtype": "bfloat16",
            "generate_cfg": {"top_p": 0.8}
        }
        self.agent_llm = get_chat_model(llm_config)
        
        self.qwen_assistant = Assistant(
            llm=self.agent_llm, # Pass the factory-created LLM object
            function_list=['file_system_lister']
        )
        
        # --- Custom Learning Loop (DISABLED) ---
        self.plastic_params_list = []
        self.optimizer = None
        logger.info("[Agent] Custom learning loop is DISABLED pending refactor.")
        logger.info("[Agent] Initialization complete.")
        # self.load_latest_checkpoint() # Also disabled

    def generate_response(self, conversation_history: List[Message], tool_choice: Union[str, Dict] = "auto") -> List[Message]:
        response_messages = []
        for response in self.qwen_assistant.run(messages=conversation_history):
            response_messages.extend(response)
        logger.info(f"[Agent] Generated {len(response_messages)} response messages.")
        return response_messages

# --- FastAPI App and Endpoints ---
app = FastAPI(title="Velocity Agent API", version="0.5.1") # Bump version
agent = None

# --- OpenAI-Compatible Endpoint Data Models ---
# *** DEFINITIVE FIX: Re-ordered to define OATool *before* it is used ***
class OAMessageContent(BaseModel):
    type: str
    text: Optional[str] = None

class OAToolFunction(BaseModel):
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}

class OATool(BaseModel):
    type: str = "function"
    function: OAToolFunction

class OAToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall

class OAChatMessage(BaseModel):
    role: str
    content: Union[str, List[OAMessageContent]]
    tool_calls: Optional[List[OAToolCall]] = None
    tool_call_id: Optional[str] = None

class OAChatCompletionRequest(BaseModel):
    model: str
    messages: List[OAChatMessage]
    tools: Optional[List["OATool"]] = None # This line now works
    tool_choice: Optional[str] = "auto"
    max_tokens: int = Field(default=150)
# *** END FIX ***

class OAChatCompletionChoice(BaseModel):
    index: int
    message: OAChatMessage
    finish_reason: str

class OAChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = Field(default="chat.completion")
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[OAChatCompletionChoice]

# --- API Endpoints ---
@app.post("/v1/chat/completions", response_model=OAChatCompletionResponse)
async def chat_completions(request: OAChatCompletionRequest):
    if not agent: raise HTTPException(status_code=503, detail="Agent not initialized")
    conversation_history: List[Message] = []
    for msg in request.messages:
        if isinstance(msg.content, str):
            conversation_history.append(Message(role=msg.role, content=msg.content))
        elif msg.role == 'tool':
            conversation_history.append(Message(role='tool', content=msg.content, tool_call_id=msg.tool_call_id))
    
    if not conversation_history: raise HTTPException(status_code=400, detail="No messages provided")
    
    response_messages = agent.generate_response(conversation_history)
    
    final_response_message = response_messages[-1]
    response_content = ""; response_tool_calls = None; finish_reason = "stop"
    
    if final_response_message.role == 'assistant':
        if isinstance(final_response_message.content, str): response_content = final_response_message.content
        if final_response_message.tool_calls:
            response_tool_calls = [OAToolCall(id=tc.id, function=tc.function) for tc in final_response_message.tool_calls]
            finish_reason = "tool_calls"; response_content = None
    
    response_message_oa = OAChatMessage(role="assistant", content=response_content if response_content else "", tool_calls=response_tool_calls)
    choice = OAChatCompletionChoice(index=0, message=response_message_oa, finish_reason=finish_reason)
    
    # --- Learning Loop (DISABLED) ---
    # logger.info("Background learning triggered (currently disabled).")
    
    return OAChatCompletionResponse(model=agent.model_id, choices=[choice])

@app.get("/health")
async def health_check(): return {"status": "ok"}

@app.post("/force_checkpoint")
async def force_checkpoint():
    return JSONResponse(status_code=400, content={"status": "error", "message": "Custom checkpointing disabled."})

# --- Main Execution ---
def main():
    global agent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-30B-A3B-Thinking")
    parser.add_argument("--anchor-checkpoint", default="/data/hyperion/checkpoints/Velocity-Anchor-v1.safetensors")
    parser.add_argument("--checkpoint-dir", default="/data/hyperion/checkpoints")
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
