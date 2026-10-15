#!/usr/bin/env python3
"""PPO (stable-baselines3) on the Reach task, logging a learning curve.

Local games (one per env), started here:
    python examples/train_ppo_reach.py --launch 2 --executable <build>/bin/RoR-Combined \
        --mod openbeam_test_arm.zip --robot-file openbeam_test_arm.truck --timesteps 40000 --out runs/reach

Games you already started (ROR_AGENT_PORT) by endpoint file:
    python examples/train_ppo_reach.py --endpoint ~/g0/agent-endpoint.json --endpoint ~/g1/agent-endpoint.json ...

Cloud sessions (OAuth approval in the browser on first use):
    python examples/train_ppo_reach.py --cloud --sessions 1 ...

No game at all (the package's kinematic stand-in, to check the pipeline):
    python examples/train_ppo_reach.py --fake 4 --timesteps 30000

Writes <out>/curve.csv (timesteps, episodes, mean return, success rate over
the last 50 episodes, wall seconds), <out>/model.zip and <out>/summary.json.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from openbeam_gym import OpenBeamRobotEnv, Reach, SceneConfig
from openbeam_gym.fake import FakeGame
from openbeam_gym.launch import LocalGameProcess
from openbeam_gym.mcp import CloudAccount, CloudGame, LocalGame, SessionGame


def env_from_endpoint(endpoint, scene, control_hz, seed, reach):
    env = OpenBeamRobotEnv(LocalGame.from_endpoint_file(endpoint, wait_s=120), Reach(**reach), SceneConfig(**scene),
                           control_hz=control_hz)
    env.reset(seed=seed)
    return Monitor(env, info_keywords=("is_success",))


def env_from_cloud(base_url, session_id, scene, control_hz, seed, reach):
    env = OpenBeamRobotEnv(CloudGame(CloudAccount(base_url), session_id), Reach(**reach), SceneConfig(**scene),
                           control_hz=control_hz)
    env.reset(seed=seed)
    return Monitor(env, info_keywords=("is_success",))


def env_from_session(base_url, session_id, token_file, scene, control_hz, seed, reach):
    token = Path(token_file).expanduser().read_text().strip()
    env = OpenBeamRobotEnv(SessionGame(base_url, session_id, lambda: token), Reach(**reach), SceneConfig(**scene),
                           control_hz=control_hz)
    env.reset(seed=seed)
    return Monitor(env, info_keywords=("is_success",))


def env_fake(control_hz, seed, reach):
    env = OpenBeamRobotEnv(FakeGame(), Reach(**reach), SceneConfig(), control_hz=control_hz)
    env.reset(seed=seed)
    return Monitor(env, info_keywords=("is_success",))


class Curve(BaseCallback):
    def __init__(self, path: Path, every: int):
        super().__init__()
        self.path, self.every = path, every
        self.returns, self.successes = deque(maxlen=50), deque(maxlen=50)
        self.episodes = 0
        self.started = time.time()
        self.rows = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            ep = info.get("episode")
            if ep:
                self.episodes += 1
                self.returns.append(ep["r"])
                self.successes.append(float(info.get("is_success", False)))
        if self.num_timesteps % self.every < self.training_env.num_envs and self.returns:
            row = {"timesteps": self.num_timesteps, "episodes": self.episodes,
                   "mean_return_50": float(np.mean(self.returns)), "success_rate_50": float(np.mean(self.successes)),
                   "wall_s": round(time.time() - self.started, 1)}
            self.rows.append(row)
            print(json.dumps(row), flush=True)
            with open(self.path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader()
                w.writerows(self.rows)
        return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", action="append", default=[])
    ap.add_argument("--launch", type=int, default=0)
    ap.add_argument("--executable")
    ap.add_argument("--mod", action="append", default=[])
    ap.add_argument("--games-root", default="/tmp/openbeam-gym-games")
    ap.add_argument("--cloud", action="store_true")
    ap.add_argument("--sessions", type=int, default=1)
    ap.add_argument("--session-id", action="append", default=[])
    ap.add_argument("--base-url", default="https://cloud.openbeam.world")
    ap.add_argument("--session-url", help="base URL whose /v1/sessions/<id>/mcp serves an existing session")
    ap.add_argument("--token-file", help="bearer token for --session-url (never printed)")
    ap.add_argument("--fake", type=int, default=0)
    ap.add_argument("--terrain", default="simple2.terrn2")
    ap.add_argument("--robot-file")
    ap.add_argument("--install-url", help="install_mod this archive URL before spawning (with --install-sha256)")
    ap.add_argument("--install-sha256")
    ap.add_argument("--robot-position", type=float, nargs=3)
    ap.add_argument("--control-hz", type=float, default=20.0)
    ap.add_argument("--timesteps", type=int, default=40000)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=0.03)
    ap.add_argument("--episode-steps", type=int, default=80)
    ap.add_argument("--range-fraction", type=float, default=0.35)
    ap.add_argument("--out", default="runs/reach")
    args = ap.parse_args()
    reach = dict(tolerance=args.tolerance, max_steps=args.episode_steps, range_fraction=args.range_fraction)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scene = {"terrain": args.terrain, "robot_file": args.robot_file,
             "robot_position": args.robot_position, "install_url": args.install_url,
             "install_sha256": args.install_sha256}
    processes = []
    if args.fake:
        fns = [functools.partial(env_fake, args.control_hz, args.seed + i, reach) for i in range(args.fake)]
    elif args.session_url:
        fns = [functools.partial(env_from_session, args.session_url, sid, args.token_file, scene, args.control_hz,
                                 args.seed + i, reach) for i, sid in enumerate(args.session_id)]
    elif args.cloud:
        account = CloudAccount(args.base_url)
        from openbeam_gym.vector import start_cloud_sessions
        ids = args.session_id or start_cloud_sessions(account, args.sessions)
        fns = [functools.partial(env_from_cloud, args.base_url, sid, scene, args.control_hz, args.seed + i, reach)
               for i, sid in enumerate(ids)]
    else:
        endpoints = list(args.endpoint)
        for i in range(args.launch):
            proc = LocalGameProcess(args.executable, Path(args.games_root) / f"game-{i}", mods=args.mod).start()
            processes.append(proc)
            endpoints.append(str(proc.endpoint))
        for proc in processes:
            proc.client()
        fns = [functools.partial(env_from_endpoint, e, scene, args.control_hz, args.seed + i, reach)
               for i, e in enumerate(endpoints)]
    vec = SubprocVecEnv(fns) if len(fns) > 1 else DummyVecEnv(fns)
    try:
        model = PPO("MlpPolicy", vec, n_steps=args.n_steps, batch_size=64, learning_rate=3e-4, gamma=0.98,
                    ent_coef=0.0, seed=args.seed, verbose=0)
        curve = Curve(out / "curve.csv", every=max(args.n_steps * len(fns), 512))
        started = time.time()
        model.learn(total_timesteps=args.timesteps, callback=curve)
        model.save(out / "model.zip")
        summary = {"timesteps": args.timesteps, "envs": len(fns), "wall_s": time.time() - started,
                   "env_steps_per_s": args.timesteps / (time.time() - started),
                   "first": curve.rows[0] if curve.rows else None, "last": curve.rows[-1] if curve.rows else None}
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))
    finally:
        vec.close()
        for proc in processes:
            proc.stop()


if __name__ == "__main__":
    main()
