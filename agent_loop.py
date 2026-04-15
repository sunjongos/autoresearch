import os
import sys
import json
import time
import shutil
import subprocess
import re
from pathlib import Path
from anthropic import Anthropic

# Configure to use Anthropic API
# Note: User mentioned they have .env with ANTHROPIC_API_KEY
# But we can also load it with python-dotenv or let uv handle it.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY", "your-api-key-here")
)
MODEL_NAME = os.environ.get("AUTORESEARCH_MODEL", "claude-3-haiku-20240307") # Use Haiku to avoid 404 Tier restrictions

TRAIN_FILE = "train.py"
BACKUP_FILE = "train.py.bak"
LOG_FILE = "experiment_log.json"
PROGRAM_FILE = "program.md"

def read_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def write_file(filepath, content):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

def extract_python_code(text):
    """Extracts python code from markdown code blocks if the LLM wraps it."""
    match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    match = re.search(r'```(.*?)```', text, re.DOTALL)
    if match:
        # Avoid capturing the language name if any
        code = match.group(1).strip()
        if code.startswith("python"):
            code = code[6:].strip()
        return code
        
    return text  # Assumes the whole text is code if no ticks

def get_latest_val_bpb(log_output):
    """Parses the uv run train.py output to find the final validation bits per byte (val_bpb)."""
    val_bpb_values = re.findall(r'val_bpb\s+([0-9.]+)', log_output)
    if val_bpb_values:
        return float(val_bpb_values[-1])
    return None

def run_training():
    """Executes the training script using uv and captures the output."""
    print("[AutoResearch] Running training experiment (2 minute budget, CPU)...")
    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "train.py"],
            capture_output=True,
            text=True,
            timeout=300  # 2 mins + padding
        )
        combined_output = result.stdout + "\n" + result.stderr
        val_bpb = get_latest_val_bpb(combined_output)
        
        print(f"Time: Training finished in {time.time() - start_time:.1f}s.")
        
        if result.returncode != 0:
            print("Warning: Training crashed or returned non-zero exit code.")
            return False, combined_output, None
            
        if val_bpb is None:
            print("Warning: Could not parse val_bpb from output.")
            return False, combined_output, None
            
        return True, combined_output, val_bpb

    except subprocess.TimeoutExpired as e:
        print("Warning: Training exceeded timeout budget.")
        output = ""
        if e.stdout:
            output += e.stdout.decode('utf-8', errors='ignore')
        if e.stderr:
            output += e.stderr.decode('utf-8', errors='ignore')
        val_bpb = get_latest_val_bpb(output)
        return True, output, val_bpb
    except Exception as e:
        print(f"Error: Execution error: {e}")
        return False, str(e), None

def propose_experiment(system_prompt, current_train_code, past_logs):
    """Queries the LLM to propose a new, fully replaced train.py."""
    print(f"\n[Antigravity] Thinking about the next experiment using {MODEL_NAME}...")
    
    logs_context = ""
    if past_logs:
        logs_context = "### Past Experiments (DO NOT REPEAT FAILED IDEAS):\n"
        for log in past_logs[-3:]: # Only send the last 3 to save context
            logs_context += f"- Iteration {log['iteration']}: {log['idea_summary']} | Result: val_bpb={log.get('val_bpb')} (Success: {log['success']})\n"
    
    prompt = f"""
{logs_context}

### INSTRUCTIONS:
You are an expert AI researcher. Your goal is to modify the training code below to achieve a LOWER validation bits per byte (val_bpb) within a strict 2-minute CPU training budget.
This runs on CPU only (no CUDA). Keep models small and efficient. Byte-level tokenizer (vocab=257).
Analyze the provided code, decide on ONE solid experiment (e.g., hyperparameter tuning, model architecture tweak, optimizer change), and output the FULL REVISED Python code for `train.py`.

At the very beginning of your response, write a 1-line summary of your idea starting with "IDEA: ".
Then provide the fully updated python code wrapped in ```python ... ``` blocks. Do not omit any parts of the code. Provide the COMPLETE runnable file.

### CURRENT train.py:
```python
{current_train_code}
```
"""
    try:
        response = client.messages.create(
            model=MODEL_NAME,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=8192
        )
        content = response.content[0].text
        
        # Extract idea summary
        idea_match = re.search(r'IDEA:\s*(.*)', content, re.IGNORECASE)
        idea_summary = idea_match.group(1).strip() if idea_match else "No explicit idea summary provided."
        
        # Extract new code
        new_code = extract_python_code(content)
        
        return idea_summary, new_code
    except Exception as e:
        print(f"Error: LLM API Error: {e}")
        return None, None

def load_logs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_logs(logs):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4)

def main():
    print("==================================================")
    print(" * Antigravity AutoResearch Orchestrator Loop * ")
    print("==================================================")
    
    if "your-api-key-here" in (client.api_key or ""):
        print("WARNING: ANTHROPIC_API_KEY environment variable is not set!")
        print("Please set your API key in .env or environment to let AutoResearch propose experiments.")

    if not os.path.exists(TRAIN_FILE):
        print(f"Error: {TRAIN_FILE} not found in the current directory.")
        sys.exit(1)

    system_prompt = "You are a world-class autonomous AI researcher focused on deep learning optimizations."
    if os.path.exists(PROGRAM_FILE):
        system_prompt = read_file(PROGRAM_FILE)

    logs = load_logs()
    # Establish Baseline if logs are empty
    best_val_bpb = float('inf')
    if logs:
        successful_runs = [l for l in logs if l.get('success') and l.get('val_bpb') is not None]
        if successful_runs:
            best_val_bpb = min(r['val_bpb'] for r in successful_runs)
            print(f"Current Best val_bpb: {best_val_bpb:.4f}")

    iteration = len(logs) + 1
    print(f"\n--- Starting Iteration {iteration} ---")

    try:
        current_train_code = read_file(TRAIN_FILE)
    except Exception as e:
        print(f"Error reading {TRAIN_FILE}: {e}")
        sys.exit(1)

    system_prompt = read_file(PROGRAM_FILE) if os.path.exists(PROGRAM_FILE) else "Optimize the training script."
    logs = load_logs()
    
    # 1. AI writes new code
    idea_summary, new_code = propose_experiment(system_prompt, current_train_code, logs)
    
    if not new_code or len(new_code) < 100:
        print("Skipping iteration due to LLM failure or empty code response.")
        sys.exit(1)

    print(f"Proposal: {idea_summary}")

    # Backup and Apply
    shutil.copyfile(TRAIN_FILE, BACKUP_FILE)
    write_file(TRAIN_FILE, new_code)
    print(f"Applied new code to {TRAIN_FILE}. Backup saved.")

    # Execute
    success, output_log, val_bpb = run_training()

    experiment_record = {
        "iteration": iteration,
        "timestamp": time.time(),
        "idea_summary": idea_summary,
        "success": success,
        "val_bpb": val_bpb
    }

    if not success:
        print("Experiment failed. Rolling back code.")
        shutil.copyfile(BACKUP_FILE, TRAIN_FILE)
    elif val_bpb is None:
        print("No val_bpb found. Rolling back code.")
        shutil.copyfile(BACKUP_FILE, TRAIN_FILE)
        experiment_record["success"] = False
    else:
        print(f"Resulting val_bpb: {val_bpb:.4f} (Best was: {best_val_bpb if best_val_bpb != float('inf') else 'None'})")
        if val_bpb < best_val_bpb:
            print("NEW BEST! Antigravity successfully improved the model. Keeping changes.")
            best_val_bpb = val_bpb
        else:
            print("No improvement. Rolling back to previous best code.")
            shutil.copyfile(BACKUP_FILE, TRAIN_FILE)
            experiment_record["success"] = False

    logs.append(experiment_record)
    save_logs(logs)
    print(f"Log saved to {LOG_FILE}.")
    print("Iteration complete. Run again for the next loop.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", type=int, default=1, help="Number of iterations (0=infinite)")
    args = parser.parse_args()

    max_iter = args.loop if args.loop > 0 else float('inf')
    i = 0
    while i < max_iter:
        try:
            main()
            i += 1
        except SystemExit:
            print("Iteration ended with exit. Continuing...")
            i += 1
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print(f"Unexpected error: {e}. Continuing...")
            i += 1
