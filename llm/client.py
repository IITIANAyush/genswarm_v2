"""
llm/client.py — All stages via Groq (free, no quota issues)

  Stage 1 (analyze)   → Groq: gemma2-9b-it          (fast, great at JSON)
  Stage 2+3 (write)   → Groq: llama-3.3-70b-versatile (best at code)
  Stage 4 (review)    → Groq: llama-3.1-8b-instant    (fast, cheap reviewer)

Get your free key: https://console.groq.com
  export GROQ_API_KEY="your-key"

Optional Gemini fallback (if you fix your quota):
  export GOOGLE_API_KEY="your-key"
"""

import os
import re
import time
import yaml


# ── Config loader ─────────────────────────────────────────────────────────────

_cfg_cache = None

def load_config():
    global _cfg_cache
    if _cfg_cache:
        return _cfg_cache
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    _cfg_cache = cfg
    return cfg


# ── Core Groq caller ──────────────────────────────────────────────────────────

def _ask_groq(model: str,
              prompt: str,
              system: str,
              temperature: float,
              max_retries: int,
              fallback_models: list = None) -> str:
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Run: pip install groq")
    try:
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
    except KeyError:
        raise EnvironmentError("Set env var GROQ_API_KEY  (https://console.groq.com)")

    models_to_try = [model] + (fallback_models or [])
    last_err = None

    for m in models_to_try:
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=4096,
                )
                return resp.choices[0].message.content
            except Exception as e:
                err_str = str(e).lower()
                if "rate_limit" in err_str or "429" in err_str:
                    print(f"  [RATE LIMIT] {m} — trying next model...")
                    last_err = e
                    break   # move to next model immediately
                last_err = e
                wait = 3 * (2 ** (attempt - 1))
                print(f"  [RETRY {attempt}/{max_retries}] {m}: {type(e).__name__} - retrying in {wait}s...")
                time.sleep(wait)

    raise last_err


# ── Stage-specific functions ───────────────────────────────────────────────────

def ask_gemini(prompt: str,
               system: str = "You are an expert multi-robot swarm systems engineer.",
               temperature: float = 0.2,
               max_retries: int = 4) -> str:
    """
    Stage 1 — constraint analysis.
    Uses gemma2-9b-it: small, fast, excellent at structured JSON output.
    Falls back to llama-3.1-8b-instant if rate limited.
    """
    print("    (model: gemma2-9b-it via Groq)")
    return _ask_groq(
        model="gemma2-9b-it",
        prompt=prompt,
        system=system,
        temperature=temperature,
        max_retries=max_retries,
        fallback_models=["llama-3.1-8b-instant"],
    )


def ask_groq(prompt: str,
             system: str = "You are an expert multi-robot swarm systems engineer.",
             temperature: float = 0.2,
             max_retries: int = 4) -> str:
    """
    Stages 2 & 3 — skill design + function writing.
    Uses llama-3.3-70b-versatile: best free model for code generation.
    Falls back to llama3-70b-8192.
    """
    print("    (model: llama-3.3-70b-versatile via Groq)")
    return _ask_groq(
        model="llama-3.3-70b-versatile",
        prompt=prompt,
        system=system,
        temperature=temperature,
        max_retries=max_retries,
        fallback_models=["llama3-70b-8192"],
    )


def ask_openai(prompt: str,
               system: str = "You are an expert multi-robot swarm systems engineer.",
               temperature: float = 0.2,
               max_retries: int = 4) -> str:
    """
    Stage 4 — code review.
    Uses llama-3.1-8b-instant: fast and has a very high separate rate limit,
    ideal for the many small review calls.
    Falls back to gemma2-9b-it.
    """
    print("    (model: llama-3.1-8b-instant via Groq)")
    return _ask_groq(
        model="llama-3.1-8b-instant",
        prompt=prompt,
        system=system,
        temperature=temperature,
        max_retries=max_retries,
        fallback_models=["gemma2-9b-it"],
    )


# ── Legacy shims ───────────────────────────────────────────────────────────────

def ask_claude(prompt: str, **kwargs) -> str:
    """Alias → ask_groq (70B for code generation)."""
    return ask_groq(prompt, **kwargs)

def ask_llm(prompt: str, **kwargs) -> str:
    """Legacy alias → ask_groq."""
    return ask_groq(prompt, **kwargs)


# ── Shared utility ─────────────────────────────────────────────────────────────

def clean_code(txt: str) -> str:
    """Strip markdown fences and any prose before/after actual code."""
    txt = txt.strip()

    fenced = re.search(r"```(?:python)?\s*\n(.*?)```", txt, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    txt = re.sub(r"^```python\s*$", "", txt, flags=re.MULTILINE)
    txt = re.sub(r"^```\s*$",       "", txt, flags=re.MULTILINE)

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
