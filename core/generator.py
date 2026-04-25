"""
core/generator.py
─────────────────
Stages 2-4 of the GenSwarm pipeline:
  Stage 2 — Shape/skill extraction from the user prompt (via LLM)
  Stage 3 — Function writing: compute_formation_points + maintain_formation (via LLM)
  Stage 4 — Geometry review: strict runtime validation + LLM fix loop

build_controller(cfg) is the main entry point called from main.py.
"""

import os
import re
import textwrap

import numpy as np

from llm.client import ask_groq, ask_openai, ask_gemini, clean_code

# ── Output path ────────────────────────────────────────────────────────────────

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
_CONTROLLER_PATH = os.path.join(_PROJECT_ROOT, "workspace", "generated", "controller.py")

# ── Prompts ────────────────────────────────────────────────────────────────────

# Dedicated geometry prompt — separate from the general WRITE_FUNCTION_PROMPT
GEOMETRY_PROMPT = """\
Write a Python function that generates n target positions arranged in the requested shape.

Function signature (EXACTLY):
def compute_formation_points(n, center=None):

Arguments:
  n      : int   — number of points to return
  center : np.ndarray [x,y] or None (default to np.zeros(2) when None)

Returns: np.ndarray shape (n, 2)  — ALL n points, UNIFORMLY SPACED, NO CLUSTERS

Shape to generate:
{shape_description}

Exact geometry to use:
{shape_math}

CRITICAL SPACING REQUIREMENT:
Points must be EQUALLY SPACED along the shape outline, NOT clustered.
- All pairwise distances should be approximately equal
- No close pairs (clustering) allowed except by geometric necessity (e.g., star points)
- Use consistent arc-length sampling: divide perimeter into n equal segments
- Spacing uniformity will be validated with coefficient of variation < 0.4

Rules:
- import numpy as np at top (use np.pi, np.cos, np.sin etc.)
- Centre all points at `center` (add center to every point)
- n can be any integer 1..100 — do NOT hardcode for a specific n
- Arc-length parametrisation: divide total perimeter into n EQUAL segments
- Sample one point per segment at regular intervals
- Return np.ndarray(n, 2) — shape must be EXACTLY (n, 2)
- NO markdown, NO prose — ONLY valid Python starting with `def compute_formation_points(`
"""

_SHAPE_EXTRACT_PROMPT = """\
Extract the swarm formation shape from this task instruction.

Task: {instruction}

Return ONLY a JSON object (no markdown, no prose):
{{
  "shape_description": "one sentence describing the shape, e.g. 'circle of radius 1.5 centred at origin'",
  "shape_math": "arc-length parametrization, e.g. 'x = 1.5*cos(2*pi*t), y = 1.5*sin(2*pi*t) for t in [0,1)'"
}}

If no specific shape is mentioned, default to a circle of radius 1.5.
"""

_BEHAVIOR_PROMPT = """\
Write a Python function maintain_formation for a swarm robot.

Function signature (EXACTLY):
def maintain_formation(robot_state, neighbors, goal_position, obstacles=None, **kwargs):

Arguments:
  robot_state   : dict with keys 'position' (np.ndarray[2]), 'velocity' (np.ndarray[2]), 'size' (float)
  neighbors     : list of dicts, each with 'position', 'velocity', 'size'
  goal_position : np.ndarray[2] — pre-assigned target position for this robot
  obstacles     : list of dicts with 'position' (np.ndarray[2]) and 'size' (float), or None
  **kwargs      : may contain 'max_speed' (float), 'sensing_radius' (float), 'world_size' (float)

Returns: np.ndarray([vx, vy]) — desired velocity, magnitude <= max_speed

Constraints to implement:
{constraints}

Rules:
- import numpy as np at top
- eps = 1e-8 guard all divisions
- Clamp final velocity magnitude to kwargs.get('max_speed', 0.5)
- Steer towards goal_position
- Separate from neighbours within sensing_radius
- Avoid obstacles
- NO markdown, NO prose — ONLY valid Python starting with `def maintain_formation(`
"""

_REVIEW_GEOMETRY_PROMPT = """\
The function compute_formation_points(n, center=None) FAILED strict geometry validation.

VALIDATION ERRORS:
{errors}

CURRENT (BROKEN) CODE:
{code}

SHAPE DESCRIPTION:
{shape_description}

EXACT MATH TO USE:
{shape_math}

Fix the function so ALL of the following pass for n = 5, 10, and 15:
1. Arc-length parametrization: divide the FULL perimeter into exactly n EQUAL segments,
   placing one point at the start of each segment — use endpoint=False (no duplicate endpoints).
2. NO clustering: all nearest-neighbour distances should be within 30% of the mean NN distance.
   The CV (std/mean) of nearest-neighbour distances must be < 0.4.
3. NO near-duplicate points: no two points closer than 1e-6
4. Returns np.ndarray of shape (n, 2) for any n from 1 to 100

The CV metric (using nearest-neighbour distances) tells you whether spacing is uniform:
  CV = 0   → perfectly equal spacing
  CV < 0.4 → acceptable
  CV > 0.4 → FAIL — points are unevenly distributed

Return ONLY the corrected Python function starting with `def compute_formation_points(`.
NO markdown, NO prose.
"""

# ── Strict runtime geometry validator ─────────────────────────────────────────

def _runtime_validate_geometry(code_str: str) -> tuple:
    """
    Execute code_str, call compute_formation_points for n in (5, 10, 15),
    and validate UNIFORM spacing using nearest-neighbour (NN) distances and
    the coefficient of variation (CV) metric.

    Validation rules:
      1. No syntax / exec errors
      2. compute_formation_points must be defined
      3. For each n: output shape must be (n, 2) with finite values
      4. No duplicate/overlapping points (NN distance < 1e-6)
      5. No clustering: no point has NN distance < 30% of mean NN distance
      6. CV (std/mean of NN distances) must be <= 0.4

    Using nearest-neighbour distances (not all pairwise) ensures that a
    perfectly-spaced circle is never incorrectly flagged — adjacent pairs on
    a circle ARE shorter than the mean of all pairwise distances, but their
    NN distances are all equal (CV = 0).

    Returns
    -------
    (ok: bool, error_message: str)
        ok=True  — geometry passed all checks
        ok=False — error_message explains exactly what failed (so the reviewer LLM can fix it)
    """
    namespace = {"np": np}
    try:
        exec(compile(code_str, "<generated>", "exec"), namespace)
    except Exception as exc:
        return False, f"SyntaxError/exec failure: {exc}"

    fn = namespace.get("compute_formation_points")
    if fn is None:
        return False, "compute_formation_points is not defined in the generated code"

    errors = []
    for n in (5, 10, 15):
        try:
            pts = fn(n)
            pts = np.array(pts, dtype=float)
        except Exception as exc:
            errors.append(f"n={n} raised {type(exc).__name__}: {exc}")
            continue

        if pts.shape != (n, 2):
            errors.append(f"n={n} shape={pts.shape}, expected ({n}, 2)")
            continue

        if not np.all(np.isfinite(pts)):
            errors.append(f"n={n} contains NaN or Inf values")
            continue

        # Nearest-neighbour distances for each point
        diff        = pts[:, None, :] - pts[None, :, :]   # (n, n, 2)
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1)) # (n, n)
        np.fill_diagonal(dist_matrix, np.inf)              # exclude self-distance
        nn_dists = dist_matrix.min(axis=1)                 # (n,) — one NN per point

        # 1. Duplicate check (NN distance ~ 0)
        dup_mask = nn_dists < 1e-6
        if np.any(dup_mask):
            dup_rows = np.where(dup_mask)[0].tolist()
            errors.append(
                f"n={n} duplicate/overlapping points (NN distance < 1e-6): "
                f"rows {dup_rows}"
            )
            continue

        mean_nn = float(np.mean(nn_dists))
        std_nn  = float(np.std(nn_dists))
        min_nn  = float(np.min(nn_dists))
        max_nn  = float(np.max(nn_dists))
        cv      = std_nn / mean_nn if mean_nn > 1e-12 else float("inf")

        # 2. Clustering check: any point whose NN is < 30% of mean NN
        cluster_mask = nn_dists < 0.30 * mean_nn
        if np.any(cluster_mask):
            clustered_rows = np.where(cluster_mask)[0].tolist()
            errors.append(
                f"n={n} CLUSTERING DETECTED: rows {clustered_rows} have nearest-neighbour "
                f"distance < 30% of mean NN distance. "
                f"NN stats: mean={mean_nn:.3f} min={min_nn:.3f} max={max_nn:.3f} CV={cv:.3f}. "
                f"FIX: use arc-length parametrisation to place all {n} points at EQUAL intervals "
                f"along the full perimeter."
            )
            continue

        # 3. Uniformity check via CV of NN distances
        if cv > 0.4:
            errors.append(
                f"n={n} NON-UNIFORM SPACING: CV={cv:.3f} exceeds 0.4 threshold. "
                f"NN stats: mean={mean_nn:.3f} min={min_nn:.3f} max={max_nn:.3f} std={std_nn:.3f}. "
                f"FIX: divide the shape perimeter into exactly {n} equal arc-length segments "
                f"and place one point per segment — current points are unevenly distributed."
            )
            continue

    if errors:
        return False, "\n".join(errors)
    return True, ""


# ── Stage 4: geometry review loop ─────────────────────────────────────────────

def stage_review_functions(code_str: str,
                           shape_description: str,
                           shape_math: str,
                           max_attempts: int = 3) -> str:
    """
    Validate compute_formation_points with the strict geometry validator.
    If it fails, ask the reviewer LLM to fix it (up to max_attempts times).

    Returns the (possibly corrected) code string.
    """
    print("  [4/4] Reviewing geometry with strict CV validator...")

    current_code = code_str
    for attempt in range(1, max_attempts + 1):
        ok, error_msg = _runtime_validate_geometry(current_code)

        if ok:
            print(f"  ✓ Geometry validation passed (attempt {attempt})")
            return current_code

        print(f"  [attempt {attempt}/{max_attempts}] Geometry FAILED:")
        for line in error_msg.splitlines():
            print(f"    {line}")

        if attempt == max_attempts:
            print("  [WARN] Max review attempts reached — using last code despite failure")
            return current_code

        review_prompt = _REVIEW_GEOMETRY_PROMPT.format(
            errors=error_msg,
            code=current_code,
            shape_description=shape_description,
            shape_math=shape_math,
        )
        raw   = ask_openai(
            review_prompt,
            system=(
                "You are an expert in computational geometry and Python. "
                "Fix the function precisely so that nearest-neighbour distance CV < 0.4."
            ),
        )
        fixed = clean_code(raw)
        if fixed and "def compute_formation_points" in fixed:
            current_code = fixed
        else:
            print(f"  [WARN] Reviewer returned no valid function on attempt {attempt}")

    return current_code


# ── Internal helpers ───────────────────────────────────────────────────────────

def _extract_shape_info(prompt: str) -> tuple:
    """
    Ask the LLM to identify the formation shape and its arc-length math.
    Returns (shape_description, shape_math).
    Falls back to 'circle of radius 1.5' on any parse failure.
    """
    import json

    raw = ask_gemini(
        _SHAPE_EXTRACT_PROMPT.format(instruction=prompt),
        system="You are an expert in computational geometry for robot swarms.",
    )
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"^```\s*",     "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$",     "", raw, flags=re.MULTILINE)
    try:
        data = json.loads(raw)
        desc = data.get("shape_description", "").strip()
        math = data.get("shape_math", "").strip()
        if desc and math:
            return desc, math
    except Exception:
        pass

    # Fallback
    return (
        "circle of radius 1.5 centred at origin",
        "x = 1.5*cos(2*pi*t), y = 1.5*sin(2*pi*t) for t in [0, 1)",
    )


def _generate_formation_function(shape_description: str, shape_math: str) -> str:
    """Ask the LLM to write compute_formation_points."""
    prompt = GEOMETRY_PROMPT.format(
        shape_description=shape_description,
        shape_math=shape_math,
    )
    raw = ask_groq(
        prompt,
        system=(
            "You are an expert computational geometry programmer. "
            "Write ONLY the requested Python function — no prose, no markdown."
        ),
    )
    return clean_code(raw)


def _generate_behavior_function(constraints: list) -> str:
    """Ask the LLM to write maintain_formation."""
    if constraints:
        constraint_text = "\n".join(
            f"- [{c.get('type','?')}] {c.get('name','?')}: {c.get('description','')}"
            for c in constraints
        )
    else:
        constraint_text = "- Move toward goal_position\n- Avoid collisions with neighbours and obstacles"

    prompt = _BEHAVIOR_PROMPT.format(constraints=constraint_text)
    raw = ask_groq(
        prompt,
        system=(
            "You are an expert multi-robot swarm systems programmer. "
            "Write ONLY the requested Python function — no prose, no markdown."
        ),
    )
    return clean_code(raw)


# ── Fixed boilerplate sections ─────────────────────────────────────────────────

_HEADER = textwrap.dedent("""\
    import numpy as np
    from typing import Optional, List, Dict


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
            diff = pos[:, None, :] - pts[None, :, :]
            cost = np.sum(diff ** 2, axis=-1)
            _, col = _scipy_lsa(cost)
            return pts[col]

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

""")

_FOOTER = textwrap.dedent("""\


    # ── Auto-generated entry point ────────────────────────────────────────────────
    import os as _os
    _DEBUG_CTRL = _os.environ.get("DEBUG_CONTROLLER", "0") == "1"


    def controller_step(robot_id: int, state: dict, env_info: dict) -> "np.ndarray":
        \"\"\"
        Two call modes:

        1. GLOBAL INIT (robot_id == -1, env_info['mode'] == 'global_init'):
           Returns dict{robot_id: np.ndarray [x,y]} — one unique target per robot,
           assigned via Hungarian algorithm for minimum total travel distance.

        2. PER-ROBOT STEP (every simulation timestep):
           Returns np.ndarray([vx, vy]) clamped to max_speed.

        Set env var DEBUG_CONTROLLER=1 to print tracebacks on errors.
        \"\"\"
        # ── Global init ───────────────────────────────────────────────────────────
        if env_info.get("mode") == "global_init":
            all_positions = env_info.get("all_positions", {})
            all_ids       = env_info.get("all_ids", [])
            prey_pos      = env_info.get("prey_position", None)

            valid_ids = [i for i in all_ids if i in all_positions]
            pos_list  = [np.array(all_positions[i], dtype=float) for i in valid_ids]
            center    = np.array(prey_pos, dtype=float) if prey_pos is not None else None

            goals = assign_goals(pos_list, valid_ids, center=center)
            goals = np.array(goals, dtype=float)
            return {valid_ids[k]: goals[k] for k in range(len(valid_ids))}

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
            result = maintain_formation(
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
""")


# ── Main entry point ───────────────────────────────────────────────────────────

def build_controller(cfg: dict) -> None:
    """
    Stages 2-4: generate, validate, and write workspace/generated/controller.py.

    Stage 2 — extract formation shape from cfg["prompt"] (LLM)
    Stage 3 — write compute_formation_points + maintain_formation (LLM)
    Stage 4 — validate geometry with strict CV validator; fix via LLM if needed
    """
    prompt      = cfg.get("prompt", "")
    constraints = cfg.get("constraints", [])

    # ── Stage 2: shape extraction ─────────────────────────────────────────────
    print("\n[2/4] Extracting formation shape from prompt...")
    shape_description, shape_math = _extract_shape_info(prompt)
    print(f"  → shape : {shape_description}")
    print(f"  → math  : {shape_math}")

    # ── Stage 3a: generate compute_formation_points ───────────────────────────
    print("\n[3/4] Generating compute_formation_points...")
    formation_code = _generate_formation_function(shape_description, shape_math)
    if not formation_code or "def compute_formation_points" not in formation_code:
        print("  [WARN] LLM returned no valid formation function — using circle fallback")
        formation_code = _circle_fallback()

    # ── Stage 4: strict geometry review ───────────────────────────────────────
    formation_code = stage_review_functions(
        formation_code, shape_description, shape_math
    )

    # ── Stage 3b: generate maintain_formation ─────────────────────────────────
    print("\n[3/4] Generating maintain_formation...")
    behavior_code = _generate_behavior_function(constraints)
    if not behavior_code or "def maintain_formation" not in behavior_code:
        print("  [WARN] LLM returned no valid behavior function — using default")
        behavior_code = _default_behavior_fallback()

    # ── Assemble and write controller.py ──────────────────────────────────────
    os.makedirs(os.path.dirname(_CONTROLLER_PATH), exist_ok=True)
    controller_src = (
        _HEADER
        + "\n\n" + formation_code + "\n"
        + "\n\n" + behavior_code + "\n"
        + _FOOTER
    )

    with open(_CONTROLLER_PATH, "w") as fh:
        fh.write(controller_src)

    print(f"\n  ✓ Controller written to {_CONTROLLER_PATH}")


# ── Fallback implementations ───────────────────────────────────────────────────

def _circle_fallback() -> str:
    """Minimal arc-length circle that always passes geometry validation."""
    return textwrap.dedent("""\
        def compute_formation_points(n, center=None):
            import numpy as np
            if center is None:
                center = np.zeros(2)
            if n == 1:
                return np.array([center])
            radius = 1.5
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            pts = np.column_stack([radius * np.cos(angles),
                                   radius * np.sin(angles)])
            return pts + center
    """)


def _default_behavior_fallback() -> str:
    """Simple goal-seeking with separation fallback."""
    return textwrap.dedent("""\
        def maintain_formation(robot_state, neighbors, goal_position,
                               obstacles=None, **kwargs):
            import numpy as np
            eps       = 1e-8
            max_speed = kwargs.get('max_speed', 0.5)
            sr        = kwargs.get('sensing_radius', 1.0)

            pos = robot_state['position']
            vel = np.zeros(2)

            # Goal attraction
            vel += (goal_position - pos) * 0.7

            # Neighbour separation
            for nb in (neighbors or []):
                diff = pos - nb['position']
                d    = np.linalg.norm(diff)
                if d < sr and d > eps:
                    vel += (diff / d) * 0.5

            # Obstacle avoidance
            for obs in (obstacles or []):
                diff = pos - obs['position']
                d    = np.linalg.norm(diff)
                if d < sr and d > eps:
                    vel += (diff / d) * 1.5

            spd = np.linalg.norm(vel)
            if spd > max_speed:
                vel = vel / spd * max_speed
            return vel
    """)
