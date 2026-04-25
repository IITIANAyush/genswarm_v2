"""
Stage 1 — Constraint Analysis
Uses Gemini Flash for fast, cheap structured JSON extraction.
"""
import json
import re
from llm.client import ask_gemini   # ← Gemini for this stage

CONSTRAINT_PROMPT = """\
You are a multi-robot swarm systems expert.

Given the user's task instruction, extract ALL constraints a robot must satisfy.
Classify each as:
  - "local"  : per-robot behaviour using only local sensing (collision avoidance,
                speed limit, neighbour interaction, etc.)
  - "global" : requires one-time centralised coordination across all robots
                (goal assignment, shape points, quadrant division, etc.)

Return ONLY valid JSON — no prose, no markdown fences:
{{
  "constraints": [
    {{
      "name": "ShortCamelCaseName",
      "description": "one-sentence description of what a robot must or must not do",
      "type": "local"
    }}
  ]
}}

User instruction:
{instruction}
"""

def analyze_prompt(user_prompt: str) -> dict:
    """Return a cfg dict: robots count, world params, constraint list."""

    print("[1/4] Analyzing constraints via Gemini Flash...")
    raw = ask_gemini(CONSTRAINT_PROMPT.format(instruction=user_prompt))

    # Strip any stray fences
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"^```\s*",     "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$",     "", raw, flags=re.MULTILINE)

    try:
        parsed = json.loads(raw)
        constraints = parsed.get("constraints", [])
    except json.JSONDecodeError:
        print(f"  [WARN] JSON parse failed, continuing with empty constraints.\n  Raw: {raw[:300]}")
        constraints = []

    # Detect robot count from prompt
    n_robots = 10
    m = re.search(r"(\d+)\s*robots?", user_prompt, re.IGNORECASE)
    if m:
        n_robots = int(m.group(1))

    print(f"  → {len(constraints)} constraints found, {n_robots} robots")
    for c in constraints:
        print(f"     [{c.get('type','?')}] {c.get('name','?')}: {c.get('description','')}")

    return {
        "prompt":      user_prompt,
        "robots":      n_robots,
        "constraints": constraints,
    }