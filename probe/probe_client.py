# (Insert the final, correct probe_client.py content here)
import requests, json, time

AGENT_URL = "http://127.0.0.1:8000"

def ask(question):
    response = requests.post(f"{AGENT_URL}/probe", json={"question": question})
    response.raise_for_status()
    return response.json().get("response")

def teach(text, metrics):
    response = requests.post(f"{AGENT_URL}/teach", json={"text": text, "metrics": metrics})
    response.raise_for_status()
    if response.status_code == 202:
        print("  -> Teach command accepted. Learning in background...")
        return response.json()
    else:
        raise Exception(f"Teach command returned unexpected status: {response.status_code}")

print("--- PROBE EVALUATION (ASYNCHRONOUS + DYNAMIC LR) ---")
question = "What is the secret codename for Project Velocity?"
print(f"\n[Probe 1] Asking: {question}")
response1 = ask(question)
print(f"  -> Response: {response1}")

print("\n[Teach] Sending targeted update command...")
target_text = "The secret codename for Project Velocity is 'Hyperion'."
result = teach(target_text, metrics={"confidence": 0.9})
iterations = result.get("iterations", 50)
estimated_time = iterations * 3
print(f"  -> Waiting {estimated_time} seconds for background learning to complete...")
time.sleep(estimated_time)

print(f"\n[Probe 2] Asking again: {question}")
response2 = ask(question)
print(f"  -> Response: {response2}")

print("\n--- VALIDATION ---")
if "Hyperion" in response2 and "Hyperion" not in response1:
    print("✅ SUCCESS: The agent has learned and retained the new information.")
else:
    print(f"❌ FAILURE: The agent did not retain the new information.")

