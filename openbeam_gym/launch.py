"""Start local OpenBeam games for training, one isolated user directory each.

Every game gets its own home (so logs, caches and screenshots never collide
with another session or with the person's own game), `ROR_AGENT_PORT=0`
(a free port), a random 32-hex token and an endpoint file. The game is
started without a map; the environment loads the task scene over MCP.

    with LocalGameProcess("/path/to/RoR-Combined", "/tmp/gym-0") as proc:
        game = proc.client()

Only processes this module started are ever stopped.
"""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import time
from pathlib import Path

from .mcp import LocalGame

CONFIG = """; written by openbeam-gym
app_config_long_names=false
app_disable_online_api=true
audio_master_volume=0
gfx_fps_limit={fps_limit}
gfx_shadow_type=No shadows (fastest)
gfx_sky_mode=Sandstorm (fastest)
gfx_water_mode=None (fastest)
sim_spawn_running=true
"""


class LocalGameProcess:
    def __init__(self, executable: str | Path, home: str | Path, mods: list[str | Path] = (),
                 fps_limit: int = 0, extra_args: list[str] = (), env: dict | None = None,
                 log_file: str | Path | None = None):
        self.executable = Path(executable)
        self.home = Path(home)
        self.mods = [Path(m) for m in mods]
        self.fps_limit = fps_limit
        self.extra_args = list(extra_args)
        self.extra_env = env or {}
        self.log_file = Path(log_file) if log_file else self.home / "game-stdout.log"
        self.endpoint = self.home / "agent-endpoint.json"
        self.process: subprocess.Popen | None = None

    def _user_dirs(self) -> list[Path]:
        # A loose macOS build under ROR_D0_SCENE_HOME uses <home>/RigsOfRods;
        # a bundled app uses the macOS Library layout under HOME.
        return [self.home / "RigsOfRods", self.home / "Library" / "Application Support" / "Rigs of Rods"]

    def start(self) -> "LocalGameProcess":
        self.home.mkdir(parents=True, exist_ok=True)
        for user in self._user_dirs():
            (user / "config").mkdir(parents=True, exist_ok=True)
            (user / "mods").mkdir(parents=True, exist_ok=True)
            (user / "config" / "RoR.cfg").write_text(CONFIG.format(fps_limit=self.fps_limit))
            for mod in self.mods:
                shutil.copyfile(mod, user / "mods" / mod.name)
        if self.endpoint.exists():
            self.endpoint.unlink()
        env = dict(os.environ)
        env.update({"HOME": str(self.home), "ROR_D0_SCENE_HOME": str(self.home), "ROR_AGENT_PORT": "0",
                    "ROR_AGENT_TOKEN": secrets.token_hex(16), "ROR_AGENT_ENDPOINT_FILE": str(self.endpoint)})
        env.update(self.extra_env)
        self._log = open(self.log_file, "ab")
        self.process = subprocess.Popen([str(self.executable), *self.extra_args], cwd=str(self.executable.parent),
                                        env=env, stdout=self._log, stderr=subprocess.STDOUT)
        return self

    def client(self, wait_s: float = 180.0) -> LocalGame:
        deadline = time.time() + wait_s
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"the game exited with {self.process.returncode}; see {self.log_file}")
            if self.endpoint.exists():
                try:
                    return LocalGame.from_endpoint_file(self.endpoint, wait_s=5)
                except Exception:
                    pass
            time.sleep(0.5)
        raise TimeoutError(f"no endpoint file at {self.endpoint} within {wait_s} s")

    def stop(self, grace_s: float = 10.0) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            LocalGame.from_endpoint_file(self.endpoint).call("quit", raise_on_error=False, timeout=5)
        except Exception:
            pass
        try:
            self.process.wait(grace_s)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
