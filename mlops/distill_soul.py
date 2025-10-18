# (Insert correct distill_soul.py content, ensure paths use /data/hyperion)
import torch, argparse, os
from pathlib import Path
from sentence_transformers import SentenceTransformer
parser = argparse.ArgumentParser()
parser.add_argument("--source-dir", default="/data/hyperion/blueprint/01_Identity_Core_Essence", required=False) # Updated path
parser.add_argument("--output", default="/data/hyperion/artifacts/soul_vector_v1.pt", required=False) # Updated path
args = parser.parse_args()
model = SentenceTransformer("all-MiniLM-L6-v2", device='cuda' if torch.cuda.is_available() else 'cpu')
text = [p.read_text() for p in Path(args.source_dir).glob("*.yaml")]
soul_vector = torch.mean(model.encode(text, convert_to_tensor=True), dim=0, keepdim=True)
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
torch.save(soul_vector, args.output)
print(f"INFO: Soul vector saved to {args.output}")
