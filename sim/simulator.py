"""
sim/simulator.py
────────────────
Software-only GenSwarm simulator.
Mirrors the original's pygame/QuadTree environment without ROS or hardware.

Visual style matches gymnasium_base_env.py:
  - White background
  - Green filled circles  = robots
  - Dark-red filled circle = prey (if any)
  - Grey filled circles    = obstacles
  - Blue X markers         = target positions (if global skill assigned any)
  - Thin arrow per robot   = velocity direction
  - Info panel on the right
"""

import os
import sys
import math
import time
import importlib.util

import numpy as np
import pygame

# ── Constants ───────────────────────────────────────────────────────────────
SCALE      = 100        # pixels per metre
BG_COLOR   = (255, 255, 255)
ROBOT_COL  = ( 34, 139,  34)   # forest green
ROBOT_EDGE = (  0,  80,   0)
OBS_COL    = (120, 120, 120)
PREY_COL   = (180,  30,  30)
TARGET_COL = ( 30,  30, 180)
ARROW_COL  = (  0,   0,   0)
GRID_COL   = (220, 220, 220)
TEXT_COL   = ( 30,  30,  30)
PANEL_COL  = (245, 245, 245)
SENSE_COL  = (200, 230, 200)   # faint sensing circle

FONT_SIZE  = 14

# ── Helpers ───────────────────────────────────────────────────────────────

def _to_px(pos: np.ndarray, origin_px: tuple) -> tuple:
    """Convert world coords (metres, y-up) → pygame pixel coords (y-down)."""
    ox, oy = origin_px
    return (int(ox + pos[0] * SCALE), int(oy - pos[1] * SCALE))

def _clamp_speed(v: np.ndarray, max_speed: float) -> np.ndarray:
    spd = np.linalg.norm(v)
    if spd > max_speed:
        return v / spd * max_speed
    return v

def _build_env_info(i: int, pos: np.ndarray, vel: np.ndarray,
                    all_pos: np.ndarray, all_vel: np.ndarray,
                    obstacles: list, prey_pos, sensing_radius: float,
                    world_size: float, max_speed: float,
                    robot_size: float) -> dict:
    """Build the env_info dict passed to controller_step for robot i."""
    neighbors = []
    for j in range(len(all_pos)):
        if j == i:
            continue
        d = np.linalg.norm(all_pos[j] - pos)
        if d <= sensing_radius:
            neighbors.append({
                "id":       j,
                "position": all_pos[j].copy(),
                "velocity": all_vel[j].copy(),
                "size":     robot_size,
            })
    return {
        "neighbors":      neighbors,
        "obstacles":      obstacles,
        "prey_position":  prey_pos,
        "world_size":     world_size,
        "max_speed":      max_speed,
        "sensing_radius": sensing_radius,
    }


# ── Main simulator class ───────────────────────────────────────────────────────

class Simulator:

    def __init__(self, cfg: dict, controller_fn, env_cfg: dict):
        self.cfg        = cfg
        self.ctrl       = controller_fn
        self.n          = cfg.get("robots", 10)

        # Physics params from env config
        self.world_size      = env_cfg.get("world_size",      5.0)
        self.robot_size      = env_cfg.get("robot_size",      0.08)
        self.sensing_radius  = env_cfg.get("sensing_radius",  1.0)
        self.max_speed       = env_cfg.get("max_speed",       0.5)
        self.dt              = env_cfg.get("dt",              0.02)
        self.fps             = env_cfg.get("fps",             50)
        self.damping         = env_cfg.get('damping', 0.90)  # configurable in config.yaml

        # World boundaries (half-size)
        self.half = self.world_size / 2

        # Robot state
        rng = np.random.default_rng()
        spawn = self.half * 0.7          # spawn in inner 70% to avoid wall edge
        self.pos = rng.uniform(-spawn, spawn, (self.n, 2))
        self.vel = np.zeros((self.n, 2))

        # Obstacles  (configurable; set max_obstacles: 0 in config.yaml for clean formations)
        n_obs = env_cfg.get('max_obstacles', max(0, min(2, self.n // 6)))
        self.obstacles: list[dict] = []
        placed = []
        for _ in range(n_obs):
            for _attempt in range(50):
                op = rng.uniform(-self.half * 0.6, self.half * 0.6, 2)
                if all(np.linalg.norm(op - p) > 0.6 for p in placed):
                    placed.append(op)
                    self.obstacles.append({"id": len(self.obstacles),
                                           "position": op,
                                           "size": 0.10})
                    break

        # Prey (only if constraint mentions prey/encircling/pursuing)
        prompt_lower = cfg.get("prompt", "").lower()
        has_prey = any(w in prompt_lower for w in ("prey", "encircl", "pursu", "surround", "target"))
        if has_prey:
            self.prey_pos = np.array([0.0, 0.0])
            self.prey_vel = rng.uniform(-0.1, 0.1, 2)
        else:
            self.prey_pos = None
            self.prey_vel = None

        # Target positions (global skills may fill this via state)
        self.target_positions: dict[int, np.ndarray] = {}

        # Pygame setup
        panel_w = 220
        self.sim_w_px  = int(self.world_size * SCALE * 1.2)
        self.sim_h_px  = int(self.world_size * SCALE * 1.2)
        self.panel_w   = panel_w
        self.screen_w  = self.sim_w_px + panel_w
        self.screen_h  = self.sim_h_px

        pygame.init()
        pygame.display.set_caption("GenSwarm — Software Simulation")
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        self.clock  = pygame.time.Clock()

        try:
            self.font  = pygame.font.SysFont("monospace", FONT_SIZE)
            self.font_b = pygame.font.SysFont("monospace", FONT_SIZE, bold=True)
        except Exception:
            self.font  = pygame.font.Font(None, FONT_SIZE + 4)
            self.font_b = self.font

        # World origin in pixels (centre of sim area)
        self.origin = (self.sim_w_px // 2, self.sim_h_px // 2)

        self.step_count = 0
        self.running    = True
        self.paused     = False
        self.show_sense = False    # toggle sensing circles

        # Try to run global skills once
        self._run_global_skills()

    # ── Global skill one-shot ──────────────────────────────────────────────────

    def _run_global_skills(self):
        """
        Call controller_step with a special 'global_init' flag.
        If the controller returns a dict mapping robot_id -> target, store it.
        """
        result = None
        try:
            result = self.ctrl(-1, {}, {
                "mode":           "global_init",
                "all_positions":  {i: self.pos[i].copy() for i in range(self.n)},
                "all_ids":        list(range(self.n)),
                "n_robots":       self.n,
                "world_size":     self.world_size,
                "prey_position":  self.prey_pos,
            })
            if isinstance(result, dict):
                for rid, tgt in result.items():
                    self.target_positions[int(rid)] = np.array(tgt, dtype=float)
                print(f"[SIM] Global init OK — {len(self.target_positions)} targets assigned")
            else:
                print("[SIM] Global init returned non-dict — no targets assigned")
        except Exception as e:
            import traceback
            print(f"[SIM][GLOBAL INIT ERROR] {type(e).__name__}: {e}")
            traceback.print_exc()
            print("[SIM] Continuing without pre-assigned targets.")
        print("[GLOBAL INIT] targets =", result)

    # ── Physics step ───────────────────────────────────────────────────────────

    def _physics_step(self):
        new_vel = np.zeros_like(self.vel)

        for i in range(self.n):
            state = {
                "position": self.pos[i].copy(),
                "velocity": self.vel[i].copy(),
                "size":     self.robot_size,
            }
            env_info = _build_env_info(
                i, self.pos[i], self.vel[i],
                self.pos, self.vel,
                self.obstacles, self.prey_pos,
                self.sensing_radius, self.world_size,
                self.max_speed, self.robot_size,
            )
            # Add assigned target if global skill produced one
            if i in self.target_positions:
                env_info["assigned_target"] = self.target_positions[i]

            try:
                v = self.ctrl(i, state, env_info)
                v = np.array(v, dtype=float)
                if np.any(np.isnan(v)) or np.any(np.isinf(v)):
                    v = np.zeros(2)
            except Exception:
                v = np.zeros(2)

            if i == 0 and self.step_count % 20 == 0:
                print("\n[STEP]", self.step_count)
                print("Robot0 pos:", self.pos[i])
                print("Robot0 vel:", self.vel[i])
                print("Target:", env_info.get("assigned_target"))
                print("Neighbors:", len(env_info["neighbors"]))
                print("Raw controller v:", v)

            new_vel[i] = _clamp_speed(v, self.max_speed)

        # Damping (like QuadTreeEngine)
        self.vel = new_vel * self.damping

        # Wall repulsion (soft boundary, matches paper's world boundary)
        margin = 0.3
        for i in range(self.n):
            for axis in range(2):
                if self.pos[i, axis] < -self.half + margin:
                    self.vel[i, axis] += 0.8 * ((-self.half + margin) - self.pos[i, axis])
                elif self.pos[i, axis] > self.half - margin:
                    self.vel[i, axis] -= 0.8 * (self.pos[i, axis] - (self.half - margin))
            self.vel[i] = _clamp_speed(self.vel[i], self.max_speed)

        # Robot-robot collision repulsion
        for i in range(self.n):
            for j in range(i + 1, self.n):
                diff = self.pos[i] - self.pos[j]
                d = np.linalg.norm(diff)
                min_d = 2 * self.robot_size
                if 1e-6 < d < min_d:
                    push = (diff / d) * (min_d - d) * 0.5
                    self.vel[i] += push
                    self.vel[j] -= push

        # Integrate
        self.pos += self.vel * self.dt
        # Hard clamp at walls
        self.pos = np.clip(self.pos, -self.half, self.half)

        # Move prey (random walk, bounce off walls)
        if self.prey_pos is not None:
            noise = np.random.randn(2) * 0.02
            self.prey_vel = _clamp_speed(self.prey_vel + noise, 0.15)
            self.prey_pos = self.prey_pos + self.prey_vel * self.dt
            for axis in range(2):
                if abs(self.prey_pos[axis]) > self.half * 0.8:
                    self.prey_vel[axis] *= -1
            self.prey_pos = np.clip(self.prey_pos, -self.half * 0.8, self.half * 0.8)

        self.step_count += 1

    # ── Drawing ────────────────────────────────────────────────────────────────

    def _draw(self):
        self.screen.fill(BG_COLOR)

        # Grid
        for gv in np.arange(-self.half, self.half + 0.5, 0.5):
            p1 = _to_px(np.array([gv, -self.half]), self.origin)
            p2 = _to_px(np.array([gv,  self.half]), self.origin)
            pygame.draw.line(self.screen, GRID_COL, p1, p2, 1)
            p1 = _to_px(np.array([-self.half, gv]), self.origin)
            p2 = _to_px(np.array([ self.half, gv]), self.origin)
            pygame.draw.line(self.screen, GRID_COL, p1, p2, 1)

        # World boundary
        br = int(self.half * SCALE)
        ox, oy = self.origin
        pygame.draw.rect(self.screen, (180, 180, 180),
                         (ox - br, oy - br, 2*br, 2*br), 2)

        # Sensing circles (optional toggle)
        if self.show_sense:
            r_px = int(self.sensing_radius * SCALE)
            sense_surf = pygame.Surface((r_px*2, r_px*2), pygame.SRCALPHA)
            pygame.draw.circle(sense_surf, (34, 139, 34, 25), (r_px, r_px), r_px)
            for i in range(self.n):
                px = _to_px(self.pos[i], self.origin)
                self.screen.blit(sense_surf, (px[0]-r_px, px[1]-r_px))

        # Obstacles
        for obs in self.obstacles:
            px = _to_px(obs["position"], self.origin)
            r  = max(3, int(obs["size"] * SCALE))
            pygame.draw.circle(self.screen, OBS_COL, px, r)
            pygame.draw.circle(self.screen, (60, 60, 60), px, r, 1)

        # Target positions (X markers)
        for tgt in self.target_positions.values():
            px = _to_px(tgt, self.origin)
            s = 5
            pygame.draw.line(self.screen, TARGET_COL,
                             (px[0]-s, px[1]-s), (px[0]+s, px[1]+s), 2)
            pygame.draw.line(self.screen, TARGET_COL,
                             (px[0]+s, px[1]-s), (px[0]-s, px[1]+s), 2)

        # Prey
        if self.prey_pos is not None:
            px = _to_px(self.prey_pos, self.origin)
            r  = max(4, int(self.robot_size * SCALE * 1.2))
            pygame.draw.circle(self.screen, PREY_COL, px, r)
            pygame.draw.circle(self.screen, (100, 0, 0), px, r, 2)

        # Robots + velocity arrows
        r_px = max(4, int(self.robot_size * SCALE))
        for i in range(self.n):
            px = _to_px(self.pos[i], self.origin)
            pygame.draw.circle(self.screen, ROBOT_COL, px, r_px)
            pygame.draw.circle(self.screen, ROBOT_EDGE, px, r_px, 1)

            # Velocity arrow
            spd = np.linalg.norm(self.vel[i])
            if spd > 1e-4:
                scale = 30.0 / self.max_speed
                end = (int(px[0] + self.vel[i][0] * scale),
                       int(px[1] - self.vel[i][1] * scale))
                pygame.draw.line(self.screen, ARROW_COL, px, end, 1)
                # arrowhead
                dx, dy = end[0]-px[0], end[1]-px[1]
                L = math.hypot(dx, dy)
                if L > 1:
                    ux, uy = dx/L, dy/L
                    ah = 5
                    lx = int(end[0] - ah*(ux + 0.5*uy))
                    ly = int(end[1] - ah*(uy - 0.5*ux))
                    rx = int(end[0] - ah*(ux - 0.5*uy))
                    ry = int(end[1] - ah*(uy + 0.5*ux))
                    pygame.draw.polygon(self.screen, ARROW_COL,
                                        [end, (lx,ly), (rx,ry)])

            # Robot ID label (small)
            lbl = self.font.render(str(i), True, (0, 0, 0))
            self.screen.blit(lbl, (px[0] + r_px + 1, px[1] - r_px))

        # ── Info panel ─────────────────────────────────────────────────────────
        px_start = self.sim_w_px
        pygame.draw.rect(self.screen, PANEL_COL,
                         (px_start, 0, self.panel_w, self.screen_h))
        pygame.draw.line(self.screen, (180,180,180),
                         (px_start, 0), (px_start, self.screen_h), 1)

        def draw_text(text, y, bold=False, col=TEXT_COL):
            f = self.font_b if bold else self.font
            surf = f.render(text, True, col)
            self.screen.blit(surf, (px_start + 8, y))
            return y + FONT_SIZE + 3

        y = 10
        y = draw_text("GenSwarm", y, bold=True, col=(0,80,0))
        y = draw_text("Software Simulation", y)
        y += 8
        pygame.draw.line(self.screen, (180,180,180),
                         (px_start+4, y), (px_start+self.panel_w-4, y), 1)
        y += 8

        y = draw_text(f"Step : {self.step_count}", y)
        y = draw_text(f"FPS  : {int(self.clock.get_fps())}", y)
        y = draw_text(f"N    : {self.n} robots", y)
        y += 8

        avg_spd = np.mean(np.linalg.norm(self.vel, axis=1))
        y = draw_text(f"Avg speed: {avg_spd:.3f} m/s", y)

        if self.target_positions:
            assigned_err = []
            for i, tgt in self.target_positions.items():
                if i < self.n:
                    assigned_err.append(np.linalg.norm(self.pos[i] - tgt))
            if assigned_err:
                y = draw_text(f"Avg target err: {np.mean(assigned_err):.3f}m", y)

        y += 8
        pygame.draw.line(self.screen, (180,180,180),
                         (px_start+4, y), (px_start+self.panel_w-4, y), 1)
        y += 8
        y = draw_text("Controls:", y, bold=True)
        y = draw_text("  SPACE  pause/resume", y)
        y = draw_text("  S      sensing circles", y)
        y = draw_text("  R      reset positions", y)
        y = draw_text("  ESC    quit", y)

        y += 8
        pygame.draw.line(self.screen, (180,180,180),
                         (px_start+4, y), (px_start+self.panel_w-4, y), 1)
        y += 8
        y = draw_text("Legend:", y, bold=True)

        def legend_dot(color, label, yy):
            pygame.draw.circle(self.screen, color, (px_start + 14, yy + 7), 6)
            return draw_text(f"  {label}", yy)

        y = legend_dot(ROBOT_COL, "robot",    y)
        y = legend_dot(OBS_COL,   "obstacle", y)
        if self.prey_pos is not None:
            y = legend_dot(PREY_COL, "prey", y)
        if self.target_positions:
            tx, ty = px_start + 14, y + 7
            sz = 5
            pygame.draw.line(self.screen, TARGET_COL,
                             (tx-sz, ty-sz), (tx+sz, ty+sz), 2)
            pygame.draw.line(self.screen, TARGET_COL,
                             (tx+sz, ty-sz), (tx-sz, ty+sz), 2)
            y = draw_text("  target", y)

        if self.paused:
            y += 20
            draw_text("  ⏸ PAUSED", y, bold=True, col=(200, 0, 0))

        pygame.display.flip()

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self):
        print("\n[SIM] Window open. Controls: SPACE=pause  S=sensing  R=reset  ESC=quit")
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                        print("[SIM] Paused" if self.paused else "[SIM] Resumed")
                    elif event.key == pygame.K_s:
                        self.show_sense = not self.show_sense
                    elif event.key == pygame.K_r:
                        rng = np.random.default_rng()
                        spawn = self.half * 0.7
                        self.pos = rng.uniform(-spawn, spawn, (self.n, 2))
                        self.vel = np.zeros((self.n, 2))
                        self.step_count = 0
                        self._run_global_skills()
                        print("[SIM] Positions reset")

            if not self.paused:
                self._physics_step()

            self._draw()
            self.clock.tick(self.fps)

        pygame.quit()
        print("[SIM] Simulation ended.")


# ── Public entry point ─────────────────────────────────────────────────────────

def run_simulation(cfg: dict, controller_fn, env_cfg: dict):
    sim = Simulator(cfg, controller_fn, env_cfg)
    sim.run()
