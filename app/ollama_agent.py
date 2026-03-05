import subprocess

OLLAMA_PATH = r"C:\Users\Dell\AppData\Local\Programs\Ollama\ollama.exe"

def ask_ollama(prompt: str) -> str:
    """Ask TinyLlama for fast, concise responses."""
    try:
        result = subprocess.run(
            [OLLAMA_PATH, "run", "tinyllama"],
            input=prompt + "\n",
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=60  
        )   
        output = result.stdout.strip()
        if ">>> " in output:
            output = output.split(">>> ")[-1].strip()
        return output if output else "[No response]"
    except subprocess.TimeoutExpired:
        return "[Ollama error] Request timed out. Please try again."
    except Exception as e:
        return f"[Ollama error] {str(e)}"
