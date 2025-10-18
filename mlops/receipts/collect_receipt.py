import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--type", choices=["turn", "session", "promotion"], required=True)
    p.add_argument("--session-id", required=True)
    p.add_argument("--turn-id", required=True)
    p.add_argument("--persona-score", type=float, default=0.0)
    p.add_argument("--style-div", type=float, default=0.0)
    p.add_argument("--anchor", default="/data/adaptai/projects/velocity/checkpoints/Velocity-Anchor-v1.safetensors")
    p.add_argument("--tools-json", help="Path to a JSON file with tool results")
    p.add_argument("--delta-norm", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=0.0)
    p.add_argument("--mask-size-pct", type=float, default=35.0)
    p.add_argument("--ema", action="store_true")
    p.add_argument("--ewc", action="store_true")
    p.add_argument("--rolled-back", action="store_true")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--notes", default="")
    default_webhook = os.getenv("SLACK_WEBHOOK_RECEIPTS", os.getenv("SLACK_WEBHOOK", ""))
    p.add_argument("--slack-webhook", default=default_webhook)
    p.add_argument("--slack-quiet", action="store_true")
    return p.parse_args()

def load_tools(path: str | None) -> dict:
    if not path or not Path(path).exists():
        return {"calls": [], "malformed_pct": 0.0, "wasted_pct": 0.0}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        calls = data.get("nova_tool_results", [])
        return {"calls": calls, "malformed_pct": 0.0, "wasted_pct": 0.0}
    except Exception:
        return {"calls": [], "malformed_pct": 0.0, "wasted_pct": 0.0}

def main():
    args = parse_args()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out_dir = Path("/data/adaptai/projects/velocity/receipts") # Specific to Velocity
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{ts}_{args.type}_{args.turn_id}.json"
    out_path = out_dir / fname

    receipt = {
        "ts": ts,
        "type": args.type,
        "session_id": args.session_id,
        "turn_id": args.turn_id,
        "identity": {"persona_score": args.persona_score, "style_divergence": args.style_div, "anchor_ref": args.anchor},
        "tools": load_tools(args.tools_json),
        "updates": {"mask_size_pct": args.mask_size_pct, "delta_norm": args.delta_norm, "lr": args.lr, "ema": args.ema, "ewc": args.ewc, "rolled_back": args.rolled_back},
        "files": {"checkpoint": args.checkpoint},
        "provenance": {"code_commit": os.getenv("GIT_COMMIT", ""), "base_model": os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8"), "base_sha": os.getenv("MODEL_SHA", "")},
        "notes": args.notes,
    }

    out_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    
    idx = out_dir / "INDEX.md"
    with idx.open("a", encoding="utf-8") as f:
        f.write(f"- {fname}\n")
    
    if args.slack_webhook and not args.slack_quiet:
        msg = {"text": f"Velocity Receipt: {args.type} | ID: {args.turn_id[:8]} | Persona: {receipt['identity']['persona_score']:.3f} | ΔW: {receipt['updates']['delta_norm']}"}
        try:
            import requests
            requests.post(args.slack_webhook, json=msg, timeout=5)
        except Exception:
            # Fallback for no requests
            import urllib.request
            req = urllib.request.Request(args.slack_webhook, data=json.dumps(msg).encode("utf-8"), headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)

    print(f"Receipt written to {out_path}")

if __name__ == "__main__":
    main()
