from __future__ import annotations # Fix for Pydantic forward-refs
import argparse, torch, time, threading, os, sys, json, subprocess
from dotenv import load_dotenv

# --- Load Environment ---
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
# NOTE: No sys.path hacks for Qwen-VL/Qwen-Agent. Rely on installed packages.

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

# --- Qwen-Agent Imports (Corrected per Stack Summary) ---
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
            path = args.get('path', '/data/hyperion')
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
        self.checkpoint_interval = args.checkpoint_interval
        self.turn_count = 0 # For checkpointing

        # --- W&B Init ---
        try:
            if os.getenv('WANDB_API_KEY'):
                 wandb.init(project="Project-Velocity", config=args, dir="/data/hyperion/logs", reinit=True)
                 logger.info("WandB initialized successfully.")
            else: logger.warning("WANDB_API_KEY not set. Skipping WandB initialization.")
        except Exception as e: logger.error(f"Failed to initialize WandB: {e}", exc_info=False)

        logger.info(f"[Agent] Initializing on device: {self.device}")
        logger.info(f"[Agent] Using Qwen-Agent factory to load model: {self.model_id}...")
        
        llm_config = {
            "model_type": "transformers",
            "model": self.model_id,
            "device": self.device,
            "dtype": "bfloat16",
            "generate_cfg": {"top_p": 0.8}
        }
        self.agent_llm = get_chat_model(llm_config)
        
        self.qwen_assistant = Assistant(
            llm=self.agent_llm,
            function_list=['file_system_lister']
        )
        
        # --- Custom Learning Loop (DISABLED) ---
        self.model = self.agent_llm.model
        self.tokenizer = self.agent_llm.tokenizer
        self.plastic_params_list = []
        self.optimizer = None
        logger.info("[Agent] Custom learning loop is DISABLED pending refactor.")
        logger.info("[Agent] Initialization complete.")
        # self.load_latest_checkpoint() # Disabled

    def generate_response(self, conversation_history: List[Message]) -> List[Message]:
        logger.info(f"[Agent] Generating response (Qwen-Agent Assistant)...")
        response_messages = []
        for response in self.qwen_assistant.run(messages=conversation_history):
            response_messages.extend(response)
        logger.info(f"[Agent] Generated {len(response_messages)} response messages.")
        return response_messages
    
    # --- Checkpointing and Learning (All Disabled) ---
    def teach(self, text, iterations=1, learning_rate=5e-6):
        logger.warning(f"Teach command received for '{text[:20]}...' but learning is disabled.")
        pass

    def save_checkpoint(self, force_push=False):
        logger.warning("Save checkpoint triggered, but learning/checkpointing is disabled.")
        return None

    def load_latest_checkpoint(self):
        logger.info("Load latest checkpoint called, but learning/checkpointing is disabled.")
        pass

# --- FastAPI App and Endpoints ---
app = FastAPI(title="Velocity Agent API", version="0.5.2")
agent = None

# --- OpenAI-Compatible Endpoint Data Models (Corrected Order) ---
class OAMessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None # Changed to Dict for image_url

class OAToolFunction(BaseModel):
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}

class OATool(BaseModel):
    type: str = "function"
    function: OAToolFunction

class OAFunctionCall(BaseModel):
    name: str
    arguments: str

class OAToolCall(BaseModel):
    id: str
    type: str = "function"
    function: OAFunctionCall

class OAChatMessage(BaseModel):
    role: str
    content: Union[str, List[OAMessageContent]]
    tool_calls: Optional[List[OAToolCall]] = None
    tool_call_id: Optional[str] = None

class OAChatCompletionRequest(BaseModel):
    model: str
    messages: List[OAChatMessage]
    tools: Optional[List[OANode]] = None # Uses forward-ref "OATool"
    tool_choice: Optional[str] = "auto"
    max_tokens: int = Field(default=150)
    return_tool_calls: bool = False # Add the flag

class OAChatCompletionChoice(BaseModel):
    index: int
    message: OAChatMessage
    finish_reason: str

class OAChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = Field(default="chat.completion")
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[OAChatCompletionChoice] # Corrected: references Choice

# --- Tool Call Normalizer ---
def _normalize_tool_calls(msg) -> Optional[List[OAToolCall]]:
    import uuid
    calls = []
    tc_list = getattr(msg, "tool_calls", None)
    if tc_list:
        for tc in tc_list:
            fn = getattr(tc, "function", None)
            if isinstance(fn, dict): name = fn.get("name", ""); arguments = fn.get("arguments", "")
            else: name = getattr(fn, "name", ""); arguments = getattr(fn, "arguments", "")
            calls.append(OAToolCall(id=getattr(tc, "id", str(uuid.uuid4())), function=OAFunctionCall(name=name, arguments=arguments)))
    else:
        fc = getattr(msg, "function_call", None)
        if fc is not None:
            if isinstance(fc, dict): name = fc.get("name", ""); arguments = fc.get("arguments", "")
            else: name = getattr(fc, "name", ""); arguments = getattr(fc, "arguments", "")
            calls.append(OAToolCall(id=str(uuid.uuid4()), function=OAFunctionCall(name=name, arguments=arguments)))
    return calls if calls else None

# --- API Endpoints ---
@app.post("/v1/chat/completions", response_model=OAChatCompletionResponse)
async def chat_completions(request: OAChatCompletionRequest):
    if not agent: raise HTTPException(status_code=503, detail="Agent not initialized")
    
    conversation_history: List[Message] = []
    last_user_message_text: Optional[str] = None
    for msg in request.messages:
        qwen_content: List[ContentItem] = []
        if isinstance(msg.content, str):
            qwen_content.append(ContentItem(text=msg.content))
            if msg.role == "user": last_user_message_text = msg.content
        elif isinstance(msg.content, list):
            for part in msg.content:
                if part.type == "text":
                    qwen_content.append(ContentItem(text=part.text))
                    if msg.role == "user": last_user_message_text = part.text # Grab text part
                elif part.type == "image_url" and part.image_url:
                    qwen_content.append(ContentItem(image=part.image_url.get("url"))) # Get URL string
        
        conversation_history.append(Message(
            role=msg.role, 
            content=qwen_content,
            tool_call_id=msg.tool_call_id
        ))
    
    if not conversation_history: raise HTTPException(status_code=400, detail="No messages provided")
    
    response_messages = agent.generate_response(conversation_history)
    
    final_response_message = None
    if request.return_tool_calls:
        for _m in response_messages:
            if _normalize_tool_calls(_m):
                final_response_message = _m
                break
    if final_response_message is None:
        final_response_message = response_messages[-1]

    response_content = ""; response_tool_calls = None; finish_reason = "stop"
    
    if final_response_message.role == 'assistant':
        if isinstance(final_response_message.content, list):
            for item in final_response_message.content:
                if item.text: response_content += item.text + " "
            response_content = response_content.strip()
        elif isinstance(final_response_message.content, str):
            response_content = final_response_message.content

        tool_calls = _normalize_tool_calls(final_response_message)
        if tool_calls:
            response_tool_calls = tool_calls
            finish_reason = "tool_calls"
            response_content = None
    
    response_message_oa = OAChatMessage(role="assistant", content=response_content if response_content else "", tool_calls=response_tool_calls)
    choice = OAChatCompletionChoice(index=0, message=response_message_oa, finish_reason=finish_reason)
    
    # --- Learning Loop (DISABLED) ---
    if last_user_message_text:
        logger.info(f"Background learning triggered for '{last_user_message_text[:30]}...' (currently disabled).")
        # teach_intensity = 0.6 if len(last_user_message_text) < 50 else 0.1
        # learning_params = agent.mcc.decide_learning_params({"teach_intensity": teach_intensity})
        # agent.teach(last_user_message_text, iterations=learning_params["recursion_depth"], learning_rate=learning_params["lr"])
        # if wandb.run: wandb.log({"mcc_chosen_lr": learning_params["lr"], "mcc_chosen_iterations": learning_params["recursion_depth"]})
    
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
