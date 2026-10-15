"""Throughput and latency of one game session through the robot tools.

    python -m openbeam_gym.bench --endpoint ~/gym/game-0/agent-endpoint.json
    python -m openbeam_gym.bench --cloud --session SESSION_ID

Measures, per call: sim_step (targets + N physics steps + state: one env
step), robot_state alone, and camera_frame (128 px JPEG). Prints JSON.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

from .mcp import CloudAccount, CloudGame, GameClient, LocalGame


def _stats(xs: list[float]) -> dict:
    xs = sorted(xs)
    return {"n": len(xs), "mean_ms": 1000 * statistics.fmean(xs), "p50_ms": 1000 * xs[len(xs) // 2],
            "p95_ms": 1000 * xs[int(len(xs) * 0.95) - 1], "max_ms": 1000 * xs[-1]}


def bench(game: GameClient, steps: int = 300, physics_steps: int = 100, camera: int = 30, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    info = game.call("robot_info").data
    robot = {"robot": info["robot"]["instance_id"]}
    joints = {j["name"]: j.get("limits") for j in info["joints"]}
    game.call("sim_mode", {"mode": "stepped", "idle_release_s": 60})
    step_lat, wall_ms = [], []
    started = time.perf_counter()
    for _ in range(steps):
        targets = {n: float(rng.uniform(l["lower"], l["upper"]) * 0.2) if l else 0.0 for n, l in joints.items()}
        t0 = time.perf_counter()
        answer = game.call("sim_step", {**robot, "steps": physics_steps, "positions": targets})
        step_lat.append(time.perf_counter() - t0)
        wall_ms.append(answer.data.get("wall_ms", 0.0))
    elapsed = time.perf_counter() - started
    state_lat = []
    for _ in range(max(10, steps // 5)):
        t0 = time.perf_counter()
        game.call("robot_state", robot)
        state_lat.append(time.perf_counter() - t0)
    cam_lat, cam_bytes = [], []
    for _ in range(camera):
        t0 = time.perf_counter()
        frame = game.call("camera_frame", {"max_width": 128, "format": "jpeg"}, raise_on_error=False)
        cam_lat.append(time.perf_counter() - t0)
        cam_bytes.append(frame.data.get("bytes", 0))
    game.call("sim_mode", {"mode": "realtime"})
    sim_seconds = steps * physics_steps * 0.0005
    return {"env_steps_per_s": steps / elapsed, "physics_steps_per_call": physics_steps,
            "sim_realtime_factor": sim_seconds / elapsed, "sim_step": _stats(step_lat),
            "sim_step_game_wall": {"mean_ms": statistics.fmean(wall_ms)}, "robot_state": _stats(state_lat),
            "camera_frame_128px": {**_stats(cam_lat), "mean_bytes": statistics.fmean(cam_bytes) if cam_bytes else 0}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint")
    ap.add_argument("--cloud", action="store_true")
    ap.add_argument("--base-url", default="https://cloud.openbeam.world")
    ap.add_argument("--session")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--physics-steps", type=int, default=100)
    ap.add_argument("--camera", type=int, default=30)
    args = ap.parse_args()
    game = CloudGame(CloudAccount(args.base_url), args.session) if args.cloud else LocalGame.from_endpoint_file(args.endpoint)
    print(json.dumps(bench(game, args.steps, args.physics_steps, args.camera), indent=1))


if __name__ == "__main__":
    main()
