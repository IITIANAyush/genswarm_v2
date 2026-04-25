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



def compute_formation_points(n, center=None, world_size=5.0):
    if center is None:
        center = np.zeros(2)

    half_h = 2.5 * 0.35
    half_w = 2.5 * 0.25
    strokes = [
        (np.array([-half_w, -half_h]), np.array([0, half_h])),       # left leg up to apex
        (np.array([0, half_h]),        np.array([half_w, -half_h])),  # right leg down from apex
        (np.array([-half_w*0.5, 0]),   np.array([half_w*0.5, 0])),   # crossbar
    ]

    total_length = 0
    for stroke in strokes:
        total_length += np.linalg.norm(stroke[1] - stroke[0])

    points = np.zeros((n, 2))
    for k in range(n):
        t = k / (n - 1) if n > 1 else 0
        length = 0
        for stroke in strokes:
            stroke_length = np.linalg.norm(stroke[1] - stroke[0])
            if t * total_length <= length + stroke_length:
                local_t = (t * total_length - length) / stroke_length
                points[k] = stroke[0] + local_t * (stroke[1] - stroke[0])
                break
            length += stroke_length

    return points + center



def maintain_formation(robot_state, neighbors, goal_position, obstacles=None, **kwargs):
    eps = 1e-8
    max_speed = kwargs.get('max_speed', 0.5)
    sensing_radius = kwargs.get('sensing_radius', 1.0)
    target_weight = kwargs.get('target_weight', 0.7)
    cohesion_weight = kwargs.get('cohesion_weight', 0.15)
    separation_weight = kwargs.get('separation_weight', 0.5)
    obstacle_weight = kwargs.get('obstacle_weight', 1.5)
    wall_weight = kwargs.get('wall_weight', 1.0)

    # Calculate target attraction
    target_attraction = (goal_position - robot_state['position']) * target_weight

    # Calculate cohesion
    cohesion = np.zeros(2)
    if neighbors:
        avg_position = np.mean([neighbor['position'] for neighbor in neighbors], axis=0)
        cohesion = (avg_position - robot_state['position']) * cohesion_weight

    # Calculate separation
    separation = np.zeros(2)
    if neighbors:
        for neighbor in neighbors:
            distance = np.linalg.norm(neighbor['position'] - robot_state['position'])
            if distance < sensing_radius:
                separation += (robot_state['position'] - neighbor['position']) / max(distance, eps) * separation_weight

    # Calculate obstacle avoidance
    obstacle_avoidance = np.zeros(2)
    if obstacles:
        for obstacle in obstacles:
            distance = np.linalg.norm(obstacle - robot_state['position'])
            if distance < sensing_radius:
                obstacle_avoidance += (robot_state['position'] - obstacle) / max(distance, eps) * obstacle_weight

    # Calculate wall avoidance
    wall_avoidance = np.zeros(2)
    if np.linalg.norm(robot_state['position']) > 2.5:
        wall_avoidance = -robot_state['position'] / max(np.linalg.norm(robot_state['position']), eps) * wall_weight

    # Calculate total velocity
    total_velocity = target_attraction + cohesion + separation + obstacle_avoidance + wall_avoidance

    # Clamp velocity to max speed
    if np.linalg.norm(total_velocity) > max_speed:
        total_velocity = total_velocity / np.linalg.norm(total_velocity) * max_speed

    return total_velocity



# ── Auto-generated entry point ────────────────────────────────────────────────
import os as _os
_DEBUG_CTRL = _os.environ.get("DEBUG_CONTROLLER", "0") == "1"


def controller_step(robot_id: int, state: dict, env_info: dict) -> "np.ndarray":
    """
    Two call modes:

    1. GLOBAL INIT (robot_id == -1, env_info['mode'] == 'global_init'):
       Returns dict{robot_id: np.ndarray [x,y]} — one unique target per robot,
       assigned via Hungarian algorithm for minimum total travel distance.

    2. PER-ROBOT STEP (every simulation timestep):
       Returns np.ndarray([vx, vy]) clamped to max_speed.

    Set env var DEBUG_CONTROLLER=1 to print tracebacks on errors.
    """
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
