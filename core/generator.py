import ast
import importlib.util
import json
import os
import re
import sys
import tempfile
import traceback as _tb
from llm.client import ask_llm, ask_gemini, ask_groq, ask_openai, clean_code, ask_claude

# ─────────────────────────────────────────────────────────────────────────────
# Shape geometry library — injected into the compute_formation_points prompt
# so the LLM receives exact math instead of guessing.
# ─────────────────────────────────────────────────────────────────────────────

# Maps shape keywords → exact parametric math description for the LLM prompt
_SHAPE_MATH = {
    "heart": """
HEART SHAPE — use this exact parametric formula:
    t = np.linspace(0, 2*pi, n, endpoint=False)
    x = scale * 16 * np.sin(t)**3
    y = scale * (13*np.cos(t) - 5*np.cos(2t) - 2*np.cos(3t) - np.cos(4t))
where scale = half_world * 0.18  (so shape fits in 70% of world)
Then centre at `center`.
""",
    "star": """
STAR SHAPE (5-pointed) — alternate outer/inner radius:
    for k in range(n):
        angle = 2*pi * k / n
        r = outer_r if k%2==0 else inner_r  (outer=half_world*0.35, inner=half_world*0.14)
        x = r * cos(angle - pi/2)
        y = r * sin(angle - pi/2)
Then distribute n points arc-length evenly along the star outline strokes.
""",
    "circle": """
CIRCLE — uniform angular spacing:
    angles = 2*pi * np.arange(n) / n
    r = half_world * 0.35
    x = r * cos(angles)
    y = r * sin(angles)
""",
    "line": """
HORIZONTAL LINE — n evenly spaced points:
    x = np.linspace(-half_world*0.7, half_world*0.7, n)
    y = np.zeros(n)
""",
    "triangle": """
EQUILATERAL TRIANGLE — arc-length sample along 3 edges:
    vertices: top=(0, h*2/3), bottom-left=(-w/2, -h/3), bottom-right=(w/2, -h/3)
    where w = half_world*0.7, h = w * sqrt(3)/2
    Concatenate the 3 edges, sample n points at equal arc-length intervals.
""",
    "square": """
SQUARE — arc-length sample along 4 sides:
    corners at ±half_world*0.35 on both axes.
    Concatenate 4 sides, sample n points at equal arc-length intervals.
""",
    "letter_a": """
LETTER A — 3 strokes (arc-length parametrised):
    half_h = half_world * 0.35
    half_w = half_world * 0.25
    strokes = [
        ([-half_w, -half_h], [0, half_h]),       # left leg up to apex
        ([0, half_h],        [half_w, -half_h]),  # right leg down from apex
        ([-half_w*0.5, 0],   [half_w*0.5, 0]),   # crossbar
    ]
    Total arc length = sum of stroke lengths.
    For each of the n robots, compute its position t_k = k/n * total_length along
    the concatenated polyline.
""",
    "spiral": """
SPIRAL — Archimedean:
    t = np.linspace(0, 4*pi, n)
    r = half_world * 0.06 * t
    x = r * cos(t)
    y = r * sin(t)
""",
}

def _get_shape_math(prompt: str, half_world: float) -> str:
    """Return shape-specific math hint for the geometry prompt."""
    p = prompt.lower()
    # Check specific letter/word patterns
    if "letter a" in p or "form a" in p or "spell a" in p:
        return _SHAPE_MATH["letter_a"].replace("half_world", str(half_world))
    for key in ("heart", "star", "spiral", "triangle", "square", "circle", "line"):
        if key in p:
            return _SHAPE_MATH[key].replace("half_world", str(half_world))
    # Generic: tell LLM to NOT use circle
    return f"""
IMPORTANT: Do NOT default to a circle unless the task explicitly says circle.
Analyse the task description and generate the correct shape geometry.
The world half-size is {half_world}m — scale all coordinates to stay within ±{half_world*0.8}m.
Use arc-length parametrisation so n points are evenly distributed along the shape outline.
"""

# Dedicated geometry prompt — separate from the general WRITE_FUNCTION_PROMPT
GEOMETRY_PROMPT = """\
Write a Python function that generates n target positions arranged in the requested shape.

Function signature (EXACTLY):
def compute_formation_points(n, center=None):

Arguments:
  n      : int   — number of points to return
  center : np.ndarray [x,y] or None (default to np.zeros(2) when None)

Returns: np.ndarray shape (n, 2)  — ALL n points, NO duplicates

Shape to generate:
{shape_description}

Exact geometry to use:
{shape_math}

Rules:
- import numpy as np at top (use np.pi, np.cos, np.sin etc.)
- Centre all points at `center` (add center to every point)
- n can be any integer 1..100 — do NOT hardcode for a specific n
- Arc-length parametrisation: sample t = k/(n-1) or k/n along the outline
- Return np.ndarray(n, 2) — shape must be EXACTLY (n, 2)
- NO markdown, NO prose — ONLY valid Python starting with `def compute_formation_points(`
"""

# ─────────────────────────────────────────────────────────────────────────────
# Hardcoded Hungarian assignment block — injected verbatim, NEVER given to LLM
# ─────────────────────────────────────────────────────────────────────────────
# Design notes:
#   • _hungarian_assign is defined UNCONDITIONALLY at module level so it is
#     always visible regardless of exec scope or import order.
#   • scipy is tried once at import time; result stored in _USE_SCIPY flag.
#   • assign_goals calls compute_formation_points (written by LLM) then
#     delegates assignment to _hungarian_assign.
# ─────────────────────────────────────────────────────────────────────────────

HUNGARIAN_ASSIGN_CODE = """
try:
    from scipy.optimize import linear_sum_assignment as _scipy_lsa
    _USE_SCIPY = True
except ImportError:
    _USE_SCIPY = False

def _hungarian_assign(robot_positions, formation_points):
    n = len(robot_positions)
    pts = np.array(formation_points[:n], dtype=float)
    pos = np.array(robot_positions, dtype=float)

    if _USE_SCIPY:
        # Cost matrix: Euclidean distance squared
        diff = pos[:, None, :] - pts[None, :, :]
        cost = np.sum(diff ** 2, axis=-1)
        _, col = _scipy_lsa(cost)
        return pts[col]

    # Greedy fallback
    used = [False] * n
    targets = np.zeros((n, 2))
    for i in range(n):
        best_j, best_d = -1, float('inf')
        for j in range(n):
            if not used[j]:
                d = float(np.linalg.norm(pos[i] - pts[j]))
                if d < best_d:
                    best_d, best_j = d, j
        used[best_j] = True
        targets[i] = pts[best_j]
    return targets

def assign_goals(robot_positions, robot_ids, center=None, **kwargs):
    n = len(robot_ids)
    if n == 0: return np.zeros((0, 2))
    c = np.array(center, dtype=float) if center is not None else np.zeros(2)
    points = compute_formation_points(n, center=c)
    points = np.array(points, dtype=float)
    if len(points) < n:
        pad = np.tile(c, (n - len(points), 1))
        points = np.vstack([points, pad])
    return _hungarian_assign(robot_positions, points)

def initialize_formation(robot_positions, robot_ids, **kwargs):
    return assign_goals(robot_positions, robot_ids, **kwargs)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Signature contract injected into every LLM prompt
# ─────────────────────────────────────────────────────────────────────────────

_SIG_CONTRACT = """
ABSOLUTE SIGNATURE CONTRACT:
  LOCAL skill:
    def <name>(robot_state, neighbors, goal_position, obstacles=None, **kwargs):
    - robot_state: {'position': np.ndarray, 'velocity': np.ndarray, 'size': float}
    - Returns: np.ndarray([vx, vy])

  GEOMETRY function:
    def compute_formation_points(n, center=None, world_size=5.0):
    - Returns: np.ndarray (n, 2)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SKILL_DESIGN_PROMPT = """\
You are designing the skill library for a 2-D multi-robot swarm simulator.

If user requests a named shape (heart, star, line, text, square, spiral),
compute_formation_points MUST generate that exact shape.
Defaulting to circle is forbidden unless explicitly requested.

Environment:
- 2-D plane, world size {world_size}m × {world_size}m, centred at (0,0)
- Each robot: position [x,y], velocity [vx,vy], radius {robot_size}m
- Sensing radius: {sensing_radius}m  |  Max speed: {max_speed} m/s

{sig_contract}

Task constraints:
{constraints}

User instruction:
{instruction}

Design the MINIMAL set of Python functions needed.  Rules:
1. Every local skill MUST use the exact local-skill signature above.
2. ALWAYS include exactly one geometry function named compute_formation_points.
   The framework provides assign_goals and initialize_formation automatically —
   do NOT include them.
3. compute_formation_points signature is ALWAYS: (n, center=None)
   No extra parameters.  Shape size/scale are hard-coded inside the function.
4. Formation geometry rules:
   - Target points must be distributed approximately equally spaced along the shape perimeter/path.
   - If the shape has vertices/corners (triangle, square, star, polygon, letter corners),
     ensure at least one target lies on each major vertex before filling edges.
   - No duplicate or overlapping target positions.
   - Maintain meaningful minimum spacing between robots.

Return ONLY valid JSON — no prose, no markdown fences:
{{
  "skills": [
    {{
      "name": "avoid_collisions",
      "scope": "local",
      "description": "repel from nearby robots/obstacles; attract toward goal_position",
      "signature": "robot_state, neighbors, goal_position, obstacles=None, **kwargs"
    }},
    {{
      "name": "maintain_formation",
      "scope": "local",
      "description": "steer robot toward goal_position using flocking + avoidance",
      "signature": "robot_state, neighbors, goal_position, obstacles=None, **kwargs"
    }},
    {{
      "name": "compute_formation_points",
      "scope": "geometry",
      "description": "Generate formation points matching user request: at center",
      "signature": "n, center=None"
    }}
  ]
}}
"""

WRITE_FUNCTION_PROMPT = """\
Write the complete Python implementation for this multi-robot swarm function.

Environment:
- 2-D plane ±{half_world}m from origin(0,0)
- Max speed: {max_speed} m/s  |  Sensing radius: {sensing_radius}m
- `import numpy as np` is already present at the top of the file.
- TARGET ATTRACTION must use weight >= 0.7 so robots actually reach their targets.
  Separation and cohesion must be WEAKER than target attraction.
  Recommended weights: target=0.7, cohesion=0.15, separation=0.5, obstacle=1.5, wall=1.0

{sig_contract}

Function to implement:
  Name        : {name}
  Scope       : {scope}
  Description : {description}
  Exact signature: def {name}({signature}):

Task constraints:
{constraints}

Already-implemented helpers (do NOT redefine):
{other_functions}

STRICT RULES — violating any rule causes the simulation to fail:
1. Begin your response with exactly `def {name}(` — no imports, no prose, no fences.
2. Obey the ABSOLUTE SIGNATURE CONTRACT above.
3. For local skills calling other local skills, pass goal_position as a keyword:
       other_fn(robot_state, neighbors, goal_position, obstacles=obstacles, **kwargs)
4. Add `eps = 1e-8` and guard every division: use `max(x, eps)` or `x + eps`.
5. Clamp returned velocity: if np.linalg.norm(v) > max_speed, scale it down.
6. Access robot_state with dict keys ONLY: robot_state['position'], robot_state['velocity'].
   Never use robot_state[:2] or array slicing.
7. For compute_formation_points specifically:
   - Signature is EXACTLY `def compute_formation_points(n, center=None):` — no **kwargs.
   - Use only `n` and `center` — do NOT read from kwargs or any other source.
   - center defaults to np.zeros(2) when None.
   - Return np.ndarray shape (n, 2) with EXACTLY n unique rows.
   - Use arc-length parametrisation along the shape strokes.
   - Hard-code shape scale (e.g. height=4.0, width=3.0) inside the function.
   - Points must be arc-length parametrised or near-uniformly spaced.
   - Points must be arc-length parametrised or near-uniformly spaced.
   - If shape contains vertices, explicitly include vertex coordinates in sampled points.
   - No duplicate rows.
   - Minimum pairwise spacing should avoid overlap.
   - Prefer scaling shape size with n so crowding does not occur.
"""

REVIEW_PROMPT = """\
Review and fix ALL bugs in this Python function for a 2-D multi-robot swarm.

{sig_contract}

THE SIMULATOR PASSES robot_state AS A DICT:
  robot_state = {{'position': np.ndarray([x, y]),
                  'velocity': np.ndarray([vx, vy]),
                  'size':     float}}

Fix every item on this checklist:

1. CRITICAL — robot_state dict access:
   - CORRECT:   pos = robot_state['position']
   - CORRECT:   vel = robot_state['velocity']
   - WRONG:     robot_state['x']  ← KeyError, robots freeze
   - WRONG:     robot_state['y']  ← KeyError, robots freeze
   - WRONG:     robot_state[:2]   ← TypeError, robots freeze
   Search the code for robot_state['x'], robot_state['y'], and robot_state[int]
   and replace ALL occurrences with robot_state['position'][0/1].

2. LOCAL skill: goal_position MUST be the 3rd positional arg. Rewrite if not.

3. Inner calls to local skills MUST use keyword args for goal_position:
   correct_fn(robot_state, neighbors, goal_position, obstacles=obstacles, **kwargs)

4. No bare `kwargs['key']` access — ONLY `kwargs.get('key', sensible_default)`.
   Bare kwargs[] raises KeyError because the simulator may not pass that key.
   Common defaults: max_speed=0.5, sensing_radius=1.0.

5. Undefined variables — every variable must be defined before use.

6. Division-by-zero — every denominator guarded with `max(x, eps)` where eps=1e-8.

7. Velocity norm clamped to max_speed before return.

8. Missing return statement.

9. compute_formation_points ONLY:
   - Signature: exactly `def compute_formation_points(n, center=None):` — no **kwargs.
   - Use ONLY n and center inside the body.
   - Return np.ndarray (n,2) with EXACTLY n unique rows (arc-length parametrisation).

10. - Check pairwise distances: no duplicate or near-overlapping points.
    - Check spacing uniformity: consecutive distances should be approximately equal.
    - If polygon/star/vertex-based shape, ensure vertices are included.

Function name: {name}
Description  : {description}

Code to review:
{code}

CRITICAL: Return ONLY the fixed Python starting with `def {name}(`. No prose, no fences.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Post-generation runtime validator
# ─────────────────────────────────────────────────────────────────────────────

def _runtime_validate_geometry(code: str) -> list[str]:
    """
    Execute compute_formation_points in isolation and check:
      - Returns np.ndarray (n, 2)
      - All n rows are unique
      - Works for n = 1, 5, 10, 15
      - Works without any kwargs (only n and center)
    Returns list of error strings (empty = pass).
    """
    import numpy as np
    errors = []

    # Build a minimal namespace
    ns = {"np": np}
    try:
        exec(compile(code, "<geometry_check>", "exec"), ns)
    except Exception as e:
        return [f"exec failed: {e}"]

    fn = ns.get("compute_formation_points")
    if fn is None:
        return ["compute_formation_points not defined after exec"]

    for n in [1, 5, 10, 15]:
        try:
            result = fn(n)  # no center, no kwargs
        except Exception as e:
            errors.append(f"n={n} raised {type(e).__name__}: {e}")
            continue
        try:
            result = np.array(result, dtype=float)
        except Exception as e:
            errors.append(f"n={n} result not array-convertible: {e}")
            continue
        if result.shape != (n, 2):
            errors.append(f"n={n} shape {result.shape} != ({n},2)")
            continue
        
        # Uniqueness check: detect exact duplicates (1e-6 tolerance)
        duplicates_found = False
        for i in range(n):
            for j in range(i + 1, n):
                if np.allclose(result[i], result[j], atol=1e-6):
                    errors.append(f"n={n} rows {i} and {j} are exact duplicates")
                    duplicates_found = True
                    break
            if duplicates_found:
                break
        
        if duplicates_found:
            continue
        
        # Minimum separation check: use adaptive threshold based on expected spread
        # For n points distributed along a perimeter, expected min distance ~= perimeter / n
        # For world size ±2.5m (5m total), typical perimeter ~= 15m-20m
        # Expected min_sep = 15-20 / n. We allow 50% of expected as acceptable.
        expected_perimeter = 18.0  # typical for star/circle shapes
        expected_min_sep = expected_perimeter / max(n, 1)
        min_sep_threshold = expected_min_sep * 0.5  # allow 50% of expected
        min_sep_threshold = max(min_sep_threshold, 0.01)  # but never less than 0.01
        
        spacing_issues = []
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(result[i] - result[j])
                if d < min_sep_threshold:
                    spacing_issues.append((i, j, d))
        
        # Only report if more than 10% of pairs are too close (indicates real problem)
        total_pairs = n * (n - 1) / 2
        if len(spacing_issues) > total_pairs * 0.1:
            for i, j, d in spacing_issues[:3]:  # report top 3 only
                errors.append(f"n={n} rows {i},{j} too close ({d:.3f}, threshold={min_sep_threshold:.3f})")

    return errors


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _constraints_str(constraints: list) -> str:
    if not constraints:
        return "  (none specified)"
    return "\n".join(
        f"  [{c.get('type','?')}] {c['name']}: {c['description']}"
        for c in constraints
    )


def _dedup_functions(code: str) -> str:
    """Preserves imports and keeps only the latest definition of functions."""  
    pattern = re.compile(r"^def \w+\(", re.MULTILINE)
    matches = list(pattern.finditer(code))
    if not matches:
        return code
    
    # Keep everything before the first 'def' (import, global variable)
    preamble = code[:matches[0].start()]

    spans = []
    for i, m in enumerate(matches):
        start   = m.start()
        end     = matches[i + 1].start() if i + 1 < len(matches) else len(code)
        fn_name = re.match(r"def (\w+)\(", code[start:start + 80]).group(1)
        spans.append((fn_name, i, start, end))   # (name, first_order, start, end)
    
    last_span = {name: (start, end) for name, order, start, end in spans}
    # Sort by the order of first appearance
    first_order: dict[str, int]   = {}
    for fn_name, order, start, end in spans:
        if fn_name not in first_order:
            first_order[fn_name] = order
        last_span[fn_name] = (start, end)

    # Sort by first_order so dependency order is preserved
    sorted_names = sorted(first_order, key=lambda n: first_order[n])
    body = "\n\n".join(code[last_span[n][0]:last_span[n][1]] for n in sorted_names)
    return preamble + body+"\n"

def _validate_signatures(code: str) -> list[str]:
    """AST-level signature validation. Returns list of error strings."""
    errors = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]

    skip = {
        "assign_goals", "initialize_formation", "_hungarian_assign",
        "controller_step", "_scipy_lsa", "load_controller",
    }

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        name = node.name
        if name in skip or name.startswith("_"):
            continue

        args = [a.arg for a in node.args.args]

        if name == "compute_formation_points":
            if not args or args[0] != "n":
                errors.append(
                    f"{name}: first arg must be 'n', got ({', '.join(args[:2])})"
                )
        else:
            if len(args) < 3 or args[2] != "goal_position":
                errors.append(
                    f"{name}: 3rd positional arg must be 'goal_position', "
                    f"got ({', '.join(args[:4])}{',...' if len(args) > 4 else ''})"
                )
    return errors


def _runtime_validate_local(name: str, code: str) -> list[str]:
    """
    Execute a local skill with a synthetic robot_state dict and check it:
      - Does not raise on a valid state dict
      - Returns a 2-element numeric array
      - Doesn't access robot_state['x'] or robot_state['y']
    """
    import numpy as np
    errors = []

    # Check for the forbidden access pattern at source level
    for bad_key in ("robot_state['x']", 'robot_state["x"]',
                    "robot_state['y']", 'robot_state["y"]'):
        if bad_key in code:
            errors.append(
                f"uses {bad_key} — simulator passes 'position', not 'x'/'y'"
            )

    ns = {"np": np}
    try:
        exec(compile(code, f"<{name}_check>", "exec"), ns)
    except Exception as e:
        return errors + [f"exec failed: {e}"]

    fn = ns.get(name)
    if fn is None:
        return errors + [f"{name} not defined after exec"]

    # Synthetic inputs matching exactly what simulator sends
    state = {
        "position": np.array([1.0, 0.5]),
        "velocity": np.array([0.1, 0.0]),
        "size":     0.08,
    }
    goal      = np.array([3.0, 3.0])
    neighbors = [{"position": np.array([1.2, 0.6]),
                  "velocity": np.zeros(2), "size": 0.08}]

    try:
        result = fn(state, neighbors, goal, max_speed=0.5, sensing_radius=1.0)
    except Exception as e:
        errors.append(f"raised {type(e).__name__}: {e}")
        return errors

    try:
        arr = np.array(result, dtype=float)
    except Exception as e:
        errors.append(f"return value not array-convertible: {e}")
        return errors

    if arr.shape != (2,):
        errors.append(f"return shape {arr.shape} != (2,)")

    return errors


def _strip_stray_imports(code: str) -> str:
    """
    Remove only imports that the file header already provides.
    Never strip numpy — keep it as safety net for LLM-written functions.
    """
    # Only strip scipy and typing — NOT numpy
    code = re.sub(r"^from scipy\b.*\n?",  "", code, flags=re.MULTILINE)
    code = re.sub(r"^from typing\b.*\n?", "", code, flags=re.MULTILINE)
    return code

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline stages
# ─────────────────────────────────────────────────────────────────────────────

def stage_design_skills(cfg: dict) -> list:
    """Stage 2: ask LLM for skill list, enforce mandatory skills."""
    from llm.client import load_config
    env_cfg = load_config()

    prompt = SKILL_DESIGN_PROMPT.format(
        world_size     = env_cfg.get("world_size",     5.0),
        robot_size     = env_cfg.get("robot_size",     0.08),
        sensing_radius = env_cfg.get("sensing_radius", 1.0),
        max_speed      = env_cfg.get("max_speed",      0.5),
        constraints    = _constraints_str(cfg.get("constraints", [])),
        instruction    = cfg["prompt"],
        sig_contract   = _SIG_CONTRACT,
    )

    raw = ask_gemini(prompt).strip()
    raw = re.sub(r"^```json\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"^```\s*",     "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$",     "", raw, flags=re.MULTILINE)

    try:
        skills = json.loads(raw).get("skills", [])
    except json.JSONDecodeError:
        print(f"  [WARN] skill design JSON parse failed:\n{raw[:400]}")
        skills = []

    # Drop forbidden names the LLM may have included anyway
    forbidden = {"assign_goals", "initialize_formation"}
    skills = [s for s in skills if s["name"] not in forbidden]

    # Enforce compute_formation_points — correct signature, no kwargs
    skills = [s for s in skills if s["name"] != "compute_formation_points"]
    skills.append({
        "name":        "compute_formation_points",
        "scope":       "geometry",
        "description": cfg["prompt"],   # pass full user prompt so geometry knows the shape
        "signature":   "n, center=None",
        "_user_prompt": cfg["prompt"],
    })

    return skills


def stage_write_functions(skills: list, cfg: dict) -> dict:
    """Stage 3: write one function body per skill, geometry last."""
    from llm.client import load_config
    env_cfg = load_config()

    written: dict[str, str] = {}
    constraints_str = _constraints_str(cfg.get("constraints", []))
    half_world      = env_cfg.get("world_size", 5.0) / 2

    # Write local skills first (compute_formation_points last)
    ordered = sorted(
        skills,
        key=lambda s: (1 if s["name"] == "compute_formation_points" else 0)
    )

    for skill in ordered:
        other  = "\n\n".join(written.values()) if written else "# none yet"

        if skill["name"] == "compute_formation_points":
            # Use dedicated geometry prompt with exact per-shape math
            shape_math = _get_shape_math(cfg.get("prompt", ""), half_world)
            prompt = GEOMETRY_PROMPT.format(
                shape_description = cfg.get("prompt", skill["description"]),
                shape_math        = shape_math,
            )
        else:
            prompt = WRITE_FUNCTION_PROMPT.format(
                half_world     = half_world,
                max_speed      = env_cfg.get("max_speed",      0.5),
                sensing_radius = env_cfg.get("sensing_radius", 1.0),
                name           = skill["name"],
                scope          = skill["scope"],
                description    = skill["description"],
                signature      = skill.get("signature", ""),
                constraints    = constraints_str,
                other_functions = other,
                sig_contract   = _SIG_CONTRACT,
            )
        raw = ask_claude(prompt)
        written[skill["name"]] = clean_code(raw)
        print(f"     wrote: {skill['name']}()")

    return written


def stage_review_functions(
    written: dict,
    skills:  list,
    max_retries: int = 3,
) -> dict:
    """
    Stage 4: LLM review + AST validation + runtime check (geometry AND local skills).
    Retries up to max_retries on any failure.
    """
    reviewed:    dict[str, str] = {}
    skill_desc:  dict[str, str] = {s["name"]: s["description"] for s in skills}
    skill_scope: dict[str, str] = {s["name"]: s.get("scope", "") for s in skills}

    for name, code in written.items():
        prompt = REVIEW_PROMPT.format(
            sig_contract = _SIG_CONTRACT,
            description  = skill_desc.get(name, ""),
            name         = name,
            code         = code,
        )
        result = clean_code(ask_openai(prompt))

        for attempt in range(max_retries):
            sig_errors = _validate_signatures(result)
            geo_errors = _runtime_validate_geometry(result) \
                         if name == "compute_formation_points" else []
            loc_errors = _runtime_validate_local(name, result) \
                         if skill_scope.get(name) == "local" else []

            all_errors = sig_errors + geo_errors + loc_errors
            if not all_errors:
                break

            print(f"     [WARN] {name}() errors on attempt {attempt+1}/{max_retries}:")
            for e in all_errors:
                print(f"       {e}")

            err_block  = "\n".join(f"  - {e}" for e in all_errors)
            fix_prompt = (
                f"This Python function has the following errors:\n{err_block}\n\n"
                f"THE SIMULATOR PASSES robot_state AS A DICT:\n"
                f"  robot_state = {{'position': np.ndarray([x,y]), "
                f"'velocity': np.ndarray([vx,vy]), 'size': float}}\n"
                f"NEVER use robot_state['x'], robot_state['y'], or robot_state[int].\n"
                f"ALWAYS use robot_state['position'][0] and robot_state['position'][1].\n\n"
                f"{_SIG_CONTRACT}\n\n"
                f"Rewrite to fix ALL errors.\n"
                f"Return ONLY the fixed Python starting with `def {name}(`. "
                f"No prose, no fences.\n\n"
                f"{result}"
            )
            result = clean_code(ask_openai(fix_prompt))

        final_errors = (
            _validate_signatures(result)
            + (_runtime_validate_geometry(result) if name == "compute_formation_points" else [])
            + (_runtime_validate_local(name, result) if skill_scope.get(name) == "local" else [])
        )
        if final_errors:
            print(f"     [ERROR] {name}() still has issues after {max_retries} retries:")
            for e in final_errors:
                print(f"       {e}")
        else:
            print(f"     reviewed: {name}() ✓")

        reviewed[name] = result

    return reviewed


# ─────────────────────────────────────────────────────────────────────────────
# Entry point builder
# ─────────────────────────────────────────────────────────────────────────────

def _pick_primary_local(skills: list) -> str | None:
    """Return the name of the best local skill for the per-step controller."""
    locals_ = [s for s in skills if s.get("scope") == "local"]
    if not locals_:
        return None
    for keyword in ("formation", "flock", "move", "step", "control"):
        for s in reversed(locals_):
            if keyword in s["name"].lower():
                return s["name"]
    return locals_[-1]["name"]


def _make_entry_point(skills: list) -> str:
    entry = _pick_primary_local(skills)
    if not entry:
        return ""

    return f"""

# ── Auto-generated entry point ────────────────────────────────────────────────
import os as _os
_DEBUG_CTRL = _os.environ.get("DEBUG_CONTROLLER", "0") == "1"


def controller_step(robot_id: int, state: dict, env_info: dict) -> "np.ndarray":
    \"\"\"
    Two call modes:

    1. GLOBAL INIT (robot_id == -1, env_info['mode'] == 'global_init'):
       Returns dict{{robot_id: np.ndarray [x,y]}} — one unique target per robot,
       assigned via Hungarian algorithm for minimum total travel distance.

    2. PER-ROBOT STEP (every simulation timestep):
       Returns np.ndarray([vx, vy]) clamped to max_speed.

    Set env var DEBUG_CONTROLLER=1 to print tracebacks on errors.
    \"\"\"
    # ── Global init ───────────────────────────────────────────────────────────
    if env_info.get("mode") == "global_init":
        all_positions = env_info.get("all_positions", {{}})
        all_ids       = env_info.get("all_ids", [])
        prey_pos      = env_info.get("prey_position", None)

        valid_ids = [i for i in all_ids if i in all_positions]
        pos_list  = [np.array(all_positions[i], dtype=float) for i in valid_ids]
        center    = np.array(prey_pos, dtype=float) if prey_pos is not None else None

        goals = assign_goals(pos_list, valid_ids, center=center)
        goals = np.array(goals, dtype=float)
        return {{valid_ids[k]: goals[k] for k in range(len(valid_ids))}}

    # ── Per-robot step ────────────────────────────────────────────────────────
    neighbors  = env_info.get("neighbors",     [])
    obstacles  = env_info.get("obstacles",     [])
    prey_pos   = env_info.get("prey_position", None)
    world_size = env_info.get("world_size",    5.0)
    max_speed  = env_info.get("max_speed",     0.5)

    assigned = env_info.get("assigned_target", None)
    if assigned is not None:
        goal_position = np.array(assigned, dtype=float)
    elif prey_pos is not None:
        goal_position = np.array(prey_pos, dtype=float)
    else:
        goal_position = np.zeros(2)

    try:
        result = {entry}(
            state, neighbors, goal_position,
            obstacles=obstacles,
            prey_position=prey_pos,
            world_size=world_size,
            max_speed=max_speed,
            robot_id=robot_id,
        )
        if result is None:
            return np.zeros(2)
        result = np.array(result, dtype=float)
        speed  = np.linalg.norm(result)
        if speed > max_speed:
            result = result / speed * max_speed
        return result
    except Exception:
        if _DEBUG_CTRL:
            import traceback
            traceback.print_exc()
        return np.zeros(2)
"""


# ─────────────────────────────────────────────────────────────────────────────
# Controller loader (use this instead of plain import/exec)
# ─────────────────────────────────────────────────────────────────────────────

def load_controller(path: str):
    """
    Load controller.py from disk and return its controller_step function.
    Uses importlib so the module gets its own fully-populated globals
    (numpy, assign_goals, _hungarian_assign, etc.) — avoids the
    'np not defined' error that occurs with bare exec() in a fresh dict.
    """
    spec   = importlib.util.spec_from_file_location("controller", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["controller"] = module
    spec.loader.exec_module(module)
    return module.controller_step


# ─────────────────────────────────────────────────────────────────────────────
# Master build function
# ─────────────────────────────────────────────────────────────────────────────

def _write_controller(full_code: str, out_dir: str) -> str:
    """Write controller.py and return its path."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "controller.py")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_code)
    return out_path


def build_controller(cfg: dict) -> str:
    """
    Full pipeline: design → write → review → assemble → validate → write.

    Generated file layout
    ─────────────────────
    1.  import numpy as np          ← always first, never stripped
    2.  from typing import …
    3.  HUNGARIAN_ASSIGN_CODE       ← _hungarian_assign, assign_goals,
                                       initialize_formation  (hardcoded)
    4.  compute_formation_points()  ← LLM geometry (no kwargs)
    5.  Local skill functions       ← LLM controllers
    6.  controller_step()           ← hardcoded entry point
    """
    print("[2/4] Designing skill library...")
    skills = stage_design_skills(cfg)

    # Fallback if LLM returns empty skill list
    if not any(s.get("scope") == "local" for s in skills):
        print("  [WARN] No local skills returned — using fallback")
        skills.insert(0, {
            "name":        "maintain_formation",
            "scope":       "local",
            "description": "steer toward goal_position while avoiding collisions",
            "signature":   "robot_state, neighbors, goal_position, obstacles=None, **kwargs",
        })

    print(f"  → {len(skills)} skills: {[s['name'] for s in skills]}")

    print("[3/4] Writing function bodies...")
    written = stage_write_functions(skills, cfg)

    print("[4/4] Reviewing & fixing code...")
    reviewed = stage_review_functions(written, skills)

    # ── Assemble ──────────────────────────────────────────────────────────────
    parts: list[str] = [
        "import numpy as np",
        "from typing import Optional, List, Dict",
        "",
        HUNGARIAN_ASSIGN_CODE,
    ]

    # Geometry first, then local skills
    geometry_fns = [
        (n, c) for n, c in reviewed.items()
        if any(s["name"] == n and s.get("scope") == "geometry" for s in skills)
    ]
    local_fns = [
        (n, c) for n, c in reviewed.items()
        if any(s["name"] == n and s.get("scope") == "local" for s in skills)
    ]

    for _name, code in geometry_fns + local_fns:
        clean = _strip_stray_imports(code).strip()
        if clean:
            parts.append(clean)
            parts.append("")

    full_code = "\n".join(parts)

    # Remove any duplicate function definitions the LLM produced
    full_code = _dedup_functions(full_code)

    # Inject hardcoded entry point (only if LLM didn't already write one)
    if "def controller_step(" not in full_code:
        full_code += _make_entry_point(skills)

    # ── Final AST validation ──────────────────────────────────────────────────
    sig_errors = _validate_signatures(full_code)
    if sig_errors:
        print("  [WARN] Final signature issues (simulation may be degraded):")
        for e in sig_errors:
            print(f"    {e}")
    else:
        print("  ✓ Signature validation passed")

    # ── Write ─────────────────────────────────────────────────────────────────
    out_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "workspace", "generated")
    out_path = _write_controller(full_code, out_dir)

    print(f"\n  ✓ Controller written → {out_path}")
    print(f"  LLM functions : {[n for n, _ in geometry_fns + local_fns]}")
    print(f"  Fixed blocks  : _hungarian_assign, assign_goals, "
          f"initialize_formation, controller_step")
    print(f"  Load with     : generator.load_controller('{out_path}')")
    return full_code


# ─────────────────────────────────────────────────────────────────────────────
# Simulator patch notes
# ─────────────────────────────────────────────────────────────────────────────
# BUG 5 (simulator.py): `sensing_radius` is passed into _build_env_info() but
# NOT included in the returned dict, so local skills that call
# env_info.get('sensing_radius', ...) always get the default, not the real value.
#
# In sim/simulator.py, change _build_env_info to include it in the return dict:
#
#     return {
#         "neighbors":      neighbors,
#         "obstacles":      obstacles,
#         "prey_position":  prey_pos,
#         "world_size":     world_size,
#         "max_speed":      max_speed,
#         "sensing_radius": sensing_radius,   # ← ADD THIS LINE
#     }
#
# ─────────────────────────────────────────────────────────────────────────────
