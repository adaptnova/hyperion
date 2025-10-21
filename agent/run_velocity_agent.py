from __future__ import annotations # Fix for Pydantic forward-refs
import argparse, torch, time, threading, os, sys, json, subprocess
from dotenv import load_dotenv

# --- Load Environment ---
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
# NOTE: No sys.path hacks. Rely on installed packages.

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

# --- DEFINITIVE FIX: Import the FULL tool suite ---
from qwen_agent.tools import CodeInterpreter, WebSearch, Retrieval, ImageGen
from qwen_agent.tools.image_search import ImageSearch
from qwen_agent.tools.doc_parser import DocParser
from qwen_agent.tools.storage import Storage
from qwen_agent.tools.amap_weather import AmapWeather
from qwen_agent.tools.image_zoom_in_qwen3vl import ImageZoomInTool
from qwen_vl_utils import process_vision_info as qwen_vl_process
_QWEN_VL_UTILS_OK = True
# --- END FIX ---

# --- Logging Setup ---
LOG_DIR = "/data/hyperion/logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(os.path.join(LOG_DIR, "velocity.log")), logging.StreamHandler()])
logger = logging.getLogger("velocity")

# --- Define Unconstrained File System Tool ---
@register_tool('file_system_lister')
class FileSystemLister(BaseTool):
    """A tool to list files in a specified directory."""
    description = "Lists all files and directories in a specified directory."
    parameters = [{'name': 'path', 'type': 'string', 'description': 'The directory path to list (e.g., /data/hyperion/docs)', 'required': True}]
    def call(self, params: str, **kwargs) -> str:
        try:
            logger.info(f"[Tool Call] Executing file_system_lister with params: {params}")
            args = json.loads(params or "{}")
            path = args.get('path', '/') # Default to root
            # --- NO CONSTRAINTS ---
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
        self.turn_count = 0
        self.turn_timeout_sec = args.gen_timeout

        # --- W&B Init ---
        try:
            if os.getenv('WANDB_API_KEY'):
                 wandb.init(project="Project-Velocity", config=vars(args), dir="/data/hyperion/logs", reinit=True)
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
            "generate_cfg": {"max_new_tokens": 512, "top_p": 0.9, "temperature": 0.7}
        }
        self.agent_llm = get_chat_model(llm_config)
        
        # --- DEFINITIVE FIX: Instantiate and Register Full UNCONSTRAINED Tool Suite ---
        self.rag_cfg = { 'rag_searchers': ['keyword_search', 'vector_search'] }
        self.tools = {
            "code_interpreter": CodeInterpreter(), # No work_dir sandbox
            "web_search": WebSearch(),
            "retrieval": Retrieval(rag_cfg=self.rag_cfg),
            "doc_parser": DocParser(),
            "storage": Storage(),
            "image_gen": ImageGen(),
            "image_search": ImageSearch(),
            "image_zoom_in_tool": ImageZoomInTool(),
            "amap_weather": AmapWeather(),
            "file_system_lister": FileSystemLister() # Our unconstrained version
        }
        tool_names = list(self.tools.keys())
        logger.info(f"Registering full tool suite: {tool_names}")
        
        self.qwen_assistant = Assistant(
            llm=self.agent_llm,
            function_list=tool_names
        )
        # --- END FIX ---
        
        # --- Custom Learning Loop (DISABLED) ---
        self.model = self.agent_llm.model
        self.tokenizer = self.agent_llm.tokenizer
        self.plastic_params_list = []
        self.optimizer = None
        logger.info("[Agent] Custom learning loop is DISABLED pending refactor.")
        logger.info("[Agent] Initialization complete.")

    def run_with_timeout(self, conversation: List[Message]) -> List[Message]:
        out: List[Message] = []
        err: List[Exception] = []
        done = threading.Event()
        def _work():
            try:
                # Pass the instantiated tools to the run method
                for chunk in self.qwen_assistant.run(messages=conversation, tools=self.tools.values()):
                    out.extend(chunk)
            except Exception as e: err.append(e)
            finally: done.set()
        
        t = threading.Thread(target=_work, daemon=True)
        t.start()
        if not done.wait(timeout=self.turn_timeout_sec):
            raise TimeoutError(f"generation exceeded {self.turn_timeout_sec}s")
        if err: raise err[0]
        return out
    
    # --- (Learning and Checkpointing methods remain disabled) ---
    def teach(self, text, iterations=1, learning_rate=5e-6): pass
    def save_checkpoint(self, force_push=False): return None
    def load_latest_checkpoint(self): pass

# --- (FastAPI App, Pydantic Models, Image Handlers, Tool Normalizer... all unchanged) ---
# --- FastAPI App and Endpoints ---
app = FastAPI(title="Velocity Agent API", version="0.7.2") # Bump version
agent: Optional[VelocityAgent] = None
class OAImageUrl(BaseModel): url: str
class OAMessageContent(BaseModel): type: str; text: Optional[str] = None; image_url: Optional[OAImageUrl] = None
class OAToolFunction(BaseModel): name: str = ""; description: str = ""; parameters: Dict[str, Any] = {}
class OATool(BaseModel): type: str = "function"; function: OAToolFunction
class OAFunctionCall(BaseModel): name: str; arguments: str
class OAToolCall(BaseModel): id: str; type: str = "function"; function: OAFunctionCall
class OAChatMessage(BaseModel): role: str; content: Union[str, List[OAMessageContent]]; tool_calls: Optional[List[OAToolCall]] = None; tool_call_id: Optional[str] = None
class OAChatCompletionRequest(BaseModel): model: str; messages: List[OAChatMessage]; tools: Optional[List['OATool']] = None; tool_choice: Optional[str] = "auto"; max_tokens: int = Field(default=150); return_tool_calls: bool = False
class OAChatCompletionChoice(BaseModel): index: int; message: OAChatMessage; finish_reason: str
class OAChatCompletionResponse(BaseModel): id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}"); object: str = "chat.completion"; created: int = Field(default_factory=lambda: int(time.time())); model: str; choices: List[OAChatCompletionChoice]
_ALLOWED_MIME = {"image/png", "image/jpeg", "image/webp"}
_MAX_BYTES = 10 * 1024 * 1024; _CONNECT_TIMEOUT = 5; _READ_TIMEOUT = 10
_VL_TMP_DIR = "/tmp/velocity_vl"; os.makedirs(_VL_TMP_DIR, exist_ok=True)
def _guess_ext(mime: str) -> str: return { "image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp" }.get(mime, ".png")
def _download_image(url: str) -> Tuple[str, str]:
    with requests.get(url, stream=True, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT)) as r:
        r.raise_for_status(); mime = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if mime not in _ALLOWED_MIME: logger.warning(f"Unusual mime '{mime}', trying to continue.")
        total = 0; fd, tmp_path = tempfile.mkstemp(prefix="qwen_vl_", suffix=_guess_ext(mime), dir=_VL_TMP_DIR); os.close(fd)
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    total += len(chunk)
                    if total > _MAX_BYTES: f.close(); os.unlink(tmp_path); raise HTTPException(status_code=413, detail="image too large (>10MB)")
                    f.write(chunk)
    return tmp_path, mime
def _ensure_png_rgb(local_path: str) -> str:
    out_path = os.path.splitext(local_path)[0] + ".png"
    try:
        with Image.open(local_path) as im:
            if im.mode not in ("RGB", "RGBA"): im = im.convert("RGB")
            im.save(out_path, format="PNG")
        if local_path != out_path and os.path.exists(local_path): os.unlink(local_path)
    except Exception as e:
        if os.path.exists(local_path): os.unlink(local_path)
        if os.path.exists(out_path): os.unlink(out_path)
        raise HTTPException(status_code=422, detail=f"image decode failed: {e}")
    return out_path
def _qwen_vl_validate(paths: List[str]) -> None:
    if not _QWEN_VL_UTILS_OK: logger.warning("qwen-vl-utils not available; skipping preprocess validation."); return
    logger.info(f"Validating {len(paths)} images with qwen-vl-utils (patch_size=16)...")
    try:
        dummy_messages = [{"role": "user", "content": [{"type": "image", "image": p} for p in paths]}]
        qwen_vl_process(dummy_messages, patch_size=16)
        logger.info("Vision preprocess validation passed.")
    except Exception as e: logger.error(f"qwen-vl-utils preprocess failed: {e}"); raise HTTPException(status_code=422, detail=f"vision preprocess failed: {e}")
def _strip_cot(s: str) -> str:
    if not s: return s
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S|re.I); s = re.sub(r"", "", s, flags=re.S|re.I)
    return s.strip()
temp_files_to_clean = []
def _cleanup_temp_files():
    global temp_files_to_clean
    for f in temp_files_to_clean:
        try:
            if os.path.exists(f): os.unlink(f)
        except Exception as e: logger.warning(f"Failed to clean up temp file {f}: {e}")
    temp_files_to_clean = []
def _normalize_tool_calls(msg) -> Optional[List[OAToolCall]]:
    import uuid as _uuid; calls = []; tc_list = getattr(msg, "tool_calls", None)
    if tc_list:
        for tc in tc_list:
            fn = getattr(tc, "function", None)
            if isinstance(fn, dict): name = fn.get("name", ""); arguments = fn.get("arguments", "")
            else: name = getattr(fn, "name", ""); arguments = getattr(fn, "arguments", "")
            calls.append(OAToolCall(id=getattr(tc, "id", str(_uuid.uuid4())), function=OAFunctionCall(name=name, arguments=arguments)))
    else:
        fc = getattr(msg, "function_call", None)
        if fc is not None:
            if isinstance(fc, dict): name = fc.get("name", ""); arguments = fc.get("arguments", "")
            else: name = getattr(fc, "name", ""); arguments = getattr(fc, "arguments", "")
            calls.append(OAToolCall(id=str(_uuid.uuid4()), function=OAFunctionCall(name=name, arguments=arguments)))
    return calls if calls else None
def _oa_to_qwen_msgs(req: OAChatCompletionRequest) -> Tuple[List[Message], bool]:
    global temp_files_to_clean; _cleanup_temp_files(); qwen_msgs: List[Message] = []; had_images = False
    for m in req.messages:
        content_items: List[ContentItem] = []
        if isinstance(m.content, str): content_items.append(ContentItem(text=m.content))
        else:
            for part in m.content:
                if part.type == "text" and part.text is not None: content_items.append(ContentItem(text=part.text))
                elif part.type == "image_url" and part.image_url and part.image_url.url:
                    had_images = True
                    try:
                        dl_path, _mime = _download_image(part.image_url.url)
                        png_path = _ensure_png_rgb(dl_path)
                        content_items.append(ContentItem(image=f"file://{png_path}"))
                        temp_files_to_clean.append(png_path)
                    except Exception as e:
                        logger.error(f"Failed to download/process image {part.image_url.url}: {e}")
                        content_items.append(ContentItem(text=f"(Error: Failed to load image {part.image_url.url})"))
        
        qwen_msgs.append(Message(role=m.role, content=content_items, tool_call_id=m.tool_call_id))
    
    if had_images:
        all_image_paths = [item.image.replace("file://", "") for msg in qwen_msgs for item in msg.content if isinstance(item, ContentItem) and item.image]
        if all_image_paths: _qwen_vl_validate(all_image_paths)
            
    return qwen_msgs, had_images
@app.post("/v1/chat/completions", response_model=OAChatCompletionResponse)
async def chat_completions(request: OAChatCompletionRequest):
    if agent is None: raise HTTPException(status_code=503, detail="Agent not initialized")
    try:
        qwen_msgs, had_images = _oa_to_qwen_msgs(request)
        responses = agent.run_with_timeout(qwen_msgs)
        final = next((msg for msg in reversed(responses) if msg.role == "assistant"), None)
        if final is None: raise HTTPException(status_code=500, detail="no assistant message returned")
        out_text = ""; tool_calls = _normalize_tool_calls(final); finish_reason = "stop"
        if isinstance(final.content, list):
            out_text = " ".join([it.text for it in final.content if getattr(it, "text", None)]).strip()
        elif isinstance(final.content, str): out_text = final.content or ""
        out_text = _strip_cot(out_text)
        if tool_calls and request.return_tool_calls:
             finish_reason = "tool_calls"; out_text = ""
        choice = OAChatCompletionChoice(index=0, message=OAChatMessage(role="assistant", content=out_text, tool_calls=tool_calls if request.return_tool_calls else None), finish_reason=finish_reason)
        return OAChatCompletionResponse(model=agent.model_id, choices=[choice])
    except HTTPException: raise
    except TimeoutError as te: logger.warning(f"Generation timeout: {te}"); raise HTTPException(status_code=504, detail=str(te))
    except Exception as e: logger.exception("Generation failed"); hint = " (vision)" if had_images else ""; raise HTTPException(status_code=500, detail=f"inference error{hint}: {e}")
    finally: _cleanup_temp_files()
@app.get("/health")
async def health(): return {"status":"ok"}
@app.post("/force_checkpoint")
async def force_checkpoint(): return JSONResponse(status_code=400, content={"status": "error", "message": "Checkpointing disabled."})

# --- Main Execution ---
def main():
    global agent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-30B-A3B-Thinking")
    parser.add_argument("--anchor-checkpoint", default="/data/hyperion/checkpoints/Velocity-Anchor-v1.safetensors")
    parser.add_argument("--checkpoint-dir", default="/data/hyperion/checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--hf-repo", default="LevelUp2x/Hyperion")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gen-timeout", type=int, default=90, help="hard per-turn timeout (s)")
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    agent = VelocityAgent(args)
    logger.info(f"Starting Uvicorn server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", reload=False)

if __name__ == "__main__":
    main()
