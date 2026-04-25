import os
import re
import time
import yaml
from openai import OpenAI, InternalServerError, APITimeoutError, APIConnectionError

_cfg_cache = None

def load_config():
    global _cfg_cache
    if _cfg_cache:
        return _cfg_cache
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["api_key"] = os.environ.get("OPENROUTER_API_KEY", cfg.get("api_key", ""))
    _cfg_cache = cfg
    return cfg

_RETRYABLE = (InternalServerError, APITimeoutError, APIConnectionError)

def ask_llm(prompt: str,
            system: str = "You are an expert multi-robot swarm systems engineer.",
            temperature: float = 0.2,
            max_retries: int = 4) -> str:
    cfg = load_config()
    client = OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        timeout=60.0,
    )
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=cfg["model"],
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except _RETRYABLE as e:
            last_err = e
            wait = 3 * (2 ** (attempt - 1))
            code = getattr(e, 'status_code', '?')
            print(f"  [RETRY {attempt}/{max_retries}] HTTP {code} - retrying in {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"\n  [LLM ERROR] {type(e).__name__}: {e}")
            raise
    raise last_err


def clean_code(txt: str) -> str:
    """
    Strip markdown fences AND any prose before/after the actual code.
    Handles LLMs that write 'Here is the corrected function:' around code.
    """
    txt = txt.strip()

    # 1. If there is a fenced block, extract just its contents
    fenced = re.search(r"```(?:python)?\s*\n(.*?)```", txt, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    # 2. Strip bare fence markers left over
    txt = re.sub(r"^```python\s*$", "", txt, flags=re.MULTILINE)
    txt = re.sub(r"^```\s*$",       "", txt, flags=re.MULTILINE)

    # 3. Drop leading prose lines before the first real code line
    lines = txt.splitlines()
    code_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("import ", "from ", "def ", "class ", "#", "@")):
            code_start = i
            break
        code_start = i + 1

    result = lines[code_start:]
    while result and not result[-1].strip():
        result.pop()

    return "\n".join(result).strip()
