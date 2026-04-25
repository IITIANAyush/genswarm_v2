"""
GenSwarm — Software-Only Simulation
Usage:
    python3 main.py "your task description"
    python3 main.py --skip-gen "rerun last controller"
"""
import sys
import os
import argparse

# Make sure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from llm.client import load_config
from core.analyzer import analyze_prompt
from core.generator import build_controller
from core.loader import load_controller
from sim.simulator import run_simulation


def parse_args():
    p = argparse.ArgumentParser(description="GenSwarm software simulation")
    p.add_argument("prompt", nargs="?", default="",
                   help="Natural language task description")
    p.add_argument("--skip-gen", action="store_true",
                   help="Skip LLM generation, reuse existing controller.py")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.skip_gen and not args.prompt:
        print('Usage: python3 main.py "your task prompt"')
        print('       python3 main.py --skip-gen   (reuse last controller)')
        sys.exit(1)

    env_cfg = load_config()

    if args.skip_gen:
        print("[GEN] Skipping generation — loading existing controller.")
        cfg = {
            "prompt":      "rerun",
            "robots":      env_cfg.get("robots", 10),
            "constraints": [],
        }
    else:
        user_prompt = args.prompt
        print(f"\n{'='*60}")
        print(f"  GenSwarm — Software Simulation")
        print(f"{'='*60}")
        print(f"  Task: {user_prompt}\n")

        # Stage 1: Constraint analysis
        cfg = analyze_prompt(user_prompt)

        # Stages 2-4: Skill design, writing, review
        build_controller(cfg)

    # Load the generated controller
    print("\n[LOAD] Loading generated controller...")
    controller_fn = load_controller()
    print("  ✓ controller_step() loaded")

    # Run simulation
    print("\n[SIM] Launching simulation...")
    run_simulation(cfg, controller_fn, env_cfg)


if __name__ == "__main__":
    main()
