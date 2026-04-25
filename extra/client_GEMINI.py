import os
import re
import time
import yaml
from openai import OpenAI, InternalServerError, APITimeoutError, APIConnectionError, BadRequestError

_cfg_cache = None

def load_config():
    global _cfg_cache
    if _cfg_cache:
        return _cfg_cache
    
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Priority: Env Var > Config File
    # Checking multiple names to support both Google AI Studio and OpenRouter
    api_key = os.environ.get("GEMINI_API_KEY") or \
              os.environ.get("OPENAI_API_KEY") or \
              os.environ.get("OPENROUTER_API_KEY") or \
              cfg.get("api_key", "")
    
    cfg["api_key"] = api_key
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
        except BadRequestError as e:
            # Handles 400 errors (Expired API Keys, Invalid Models)
            print(f"\n  [LLM CONFIG ERROR] 400 Bad Request: {e.message}")
            raise
        except Exception as e:
            print(f"\n  [LLM ERROR] {type(e).__name__}: {e}")
            raise
    raise last_err

def clean_code(txt: str) -> str:
    """
    Extracts code from markdown blocks and removes surrounding prose.
    """
    txt = txt.strip()

    # 1. Extract content from fenced blocks
    # Handles ```python, ```py, or just
