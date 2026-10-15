"""N environments over N game sessions.

Each sub-environment owns one game (a local process or a cloud session) and
its own MCP connection, so steps run in parallel: `AsyncVectorEnv` steps each
env in its own worker process. Factories are picklable top-level callables.
"""

from __future__ import annotations

import functools
import webbrowser
from pathlib import Path

import gymnasium as gym

from .env import OpenBeamRobotEnv, SceneConfig
from .launch import LocalGameProcess
from .mcp import CloudAccount, CloudGame, LocalGame
from .tasks import TASKS


def _local_env(endpoint_file: str, task: str, task_kwargs: dict, scene: dict, env_kwargs: dict):
    game = LocalGame.from_endpoint_file(endpoint_file, wait_s=60)
    return OpenBeamRobotEnv(game, TASKS[task](**task_kwargs), SceneConfig(**scene), **env_kwargs)


def _cloud_env(base_url: str, session_id: str, task: str, task_kwargs: dict, scene: dict, env_kwargs: dict):
    game = CloudGame(CloudAccount(base_url), session_id)
    return OpenBeamRobotEnv(game, TASKS[task](**task_kwargs), SceneConfig(**scene), **env_kwargs)


def local_vec_env(n: int, executable: str, root: str, mods=(), task: str = "reach", task_kwargs=None,
                  scene=None, env_kwargs=None, asynchronous: bool = True, fps_limit: int = 0):
    """Start n local games and an n-wide vector env over them.

    Returns (vector_env, processes); stop the processes when done."""
    processes = []
    for i in range(n):
        proc = LocalGameProcess(executable, Path(root) / f"game-{i}", mods=mods, fps_limit=fps_limit).start()
        processes.append(proc)
    for proc in processes:
        proc.client()  # wait for each endpoint
    fns = [functools.partial(_local_env, str(p.endpoint), task, task_kwargs or {}, scene or {}, env_kwargs or {})
           for p in processes]
    vec = gym.vector.AsyncVectorEnv(fns) if asynchronous and n > 1 else gym.vector.SyncVectorEnv(fns)
    return vec, processes


def start_cloud_sessions(account: CloudAccount, n: int, attach: str = "none", width: int = 640,
                         height: int = 360) -> list[str]:
    """Start n training sessions (no player tab needed; see
    CloudAccount.start_session). `attach="browser"` also opens each private
    player_url, to watch. Account limits (concurrent sessions, invitations,
    play time) apply exactly as for the Play button."""
    ids = []
    for _ in range(n):
        started = account.start_session(width=width, height=height, kind="training")
        ids.append(started["session_id"])
        if attach == "browser":
            webbrowser.open(started["player_url"])
    for sid in ids:
        CloudGame(account, sid).wait_ready()
    return ids


def cloud_vec_env(account: CloudAccount, session_ids: list[str], task: str = "reach", task_kwargs=None,
                  scene=None, env_kwargs=None, asynchronous: bool = True):
    fns = [functools.partial(_cloud_env, account.base_url, sid, task, task_kwargs or {}, scene or {}, env_kwargs or {})
           for sid in session_ids]
    return gym.vector.AsyncVectorEnv(fns) if asynchronous and len(fns) > 1 else gym.vector.SyncVectorEnv(fns)
