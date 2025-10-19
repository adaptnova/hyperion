import requests
import json
import time

AGENT_URL = "http://127.0.0.1:8000/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}
MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Thinking"

def chat(messages):
    """Sends a chat message to the agent."""
    payload = {
        "model": MODEL_ID,
        "messages": messages
    }
    try:
        response = requests.post(AGENT_URL, headers=HEADERS, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"ERROR: API request failed: {e}")
        return None

def main_loop():
    print("--- Starting Self-Improvement Loop ---")
    print("Agent will observe its environment and teach itself a new fact every 2 minutes.")
    
    last_known_fact = ""
    
    while True:
        # 1. Define the task
        task = "First, use the 'file_system_lister' tool to see what files are in /data/hyperion. Then, based on the output, tell me the name of one of the subdirectories."
        messages = [{"role": "user", "content": task}]
        
        # 2. Interact with Agent
        print(f"\n[Loop] Sending task: {task}")
        api_response = chat(messages)
        
        if not api_response or not api_response.get("choices"):
            print("[Loop] Agent did not respond correctly. Retrying in 60s...")
            time.sleep(60)
            continue
            
        response_content = api_response["choices"][0]["message"]["content"]
        print(f"[Loop] Agent Response: {response_content}")
        
        # 3. Form a new belief (simple heuristic)
        # This is a placeholder for more complex reasoning.
        # We just tell it a "new" fact based on its own action.
        new_fact = f"The agent just successfully used the file_system_lister tool at {int(time.time())}."
        
        # 4. Teach itself the new belief (if it's different from the last one)
        if new_fact != last_known_fact:
            print(f"[Loop] Forming new belief: {new_fact}")
            # We use the chat endpoint to teach, triggering the MCC.
            teach_messages = [
                {"role": "user", "content": f"Please learn this new fact: {new_fact}"}
            ]
            chat(teach_messages) # This will trigger the background learning
            last_known_fact = new_fact
            print("[Loop] Teach command sent. Fact will be learned in background.")
        
        # 5. Wait for the next cycle
        print("[Loop] Cycle complete. Sleeping for 120 seconds...")
        time.sleep(120)

if __name__ == "__main__":
    main_loop()
