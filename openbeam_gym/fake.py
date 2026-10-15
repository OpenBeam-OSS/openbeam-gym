"""An in-process stand-in for the game's robot tools, for tests and dry runs.

`FakeGame` answers the same tool calls the environment makes (status,
load_level, robot_info, robot_state, robot_fk, robot_set_joints, sim_mode,
sim_step, camera_frame) for a rigid 4-joint arm shaped like
OpenBeam's four-rotator kinematics test arm, with first-order servos. It is
not a physics model: it exists so the package, the tasks and the training
loop are testable without a game.
"""

from __future__ import annotations

import numpy as np

from .mcp import GameClient, ToolError, ToolResult

DT = 0.0005


def _skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def _exp6(s, theta):
    w, v = np.asarray(s[:3], float), np.asarray(s[3:], float)
    t = np.eye(4)
    if np.linalg.norm(w) < 1e-12:
        t[:3, 3] = v * theta
        return t
    k = _skew(w)
    r = np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * k @ k
    g = np.eye(3) * theta + (1 - np.cos(theta)) * k + (theta - np.sin(theta)) * k @ k
    t[:3, :3] = r
    t[:3, 3] = g @ v
    return t


def _quat_xyzw(r):
    w = np.sqrt(max(1e-12, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    return [(r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w), w]


class FakeGame(GameClient):
    names = ["base_pan", "shoulder", "elbow", "wrist_roll"]

    def __init__(self, tau: float = 0.05):
        y, z, x = np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
        axes = [(y, [0, 0.2, 0]), (z, [0, 0.45, 0]), (z, [0.5, 0.45, 0]), (x, [0.8, 0.45, 0])]
        self.screws = [np.concatenate([w, np.cross(p, w)]) for w, p in axes]
        self.home = np.eye(4)
        self.home[:3, 3] = [1.0, 0.45, 0.0]
        self.limits = np.radians([[-170, 170], [-90, 90], [-135, 135], [-180, 180]])
        self.q = np.zeros(4)
        self.target = np.zeros(4)
        self.tau = tau
        self.stepped = False
        self.sim_time = 0.0
        self.steps = 0
        self.terrain = False
        self.calls = 0

    def fk(self, q):
        t = np.eye(4)
        for s, th in zip(self.screws, q):
            t = t @ _exp6(s, th)
        return t @ self.home

    def _state(self):
        t = self.fk(self.q)
        return {"sim_time": self.sim_time, "physics_steps": self.steps, "frame": self.calls,
                "sim_mode": "stepped" if self.stepped else "realtime",
                "robot": {"instance_id": 1, "file": "openbeam_test_arm.jbeam", "source": "fake"},
                "joints": [{"name": n, "position": float(q), "velocity": 0.0, "effort": None, "target": float(g)}
                           for n, q, g in zip(self.names, self.q, self.target)],
                "base_pose": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1], "frame": "world"},
                "end_effector": {"name": "tcp", "position": t[:3, 3].tolist(), "orientation": _quat_xyzw(t[:3, :3]),
                                 "frame": "base"},
                "broken_beams": 0}

    def _set(self, positions):
        accepted, clamped = [], {}
        for name, v in positions.items():
            if name not in self.names:
                continue
            i = self.names.index(name)
            c = float(np.clip(v, *self.limits[i]))
            if c != v:
                clamped[name] = c
            self.target[i] = c
            accepted.append(name)
        return {"accepted": accepted, "clamped": clamped}

    def call(self, tool, arguments=None, raise_on_error=True, timeout=None) -> ToolResult:
        self.calls += 1
        a = arguments or {}
        if tool == "status":
            return ToolResult({"app_state": "simulation" if self.terrain else "main_menu",
                               "terrain_loaded": self.terrain})
        if tool == "load_level":
            self.terrain = True
            return ToolResult({"outcome": "loaded"})
        if tool == "robot_info":
            return ToolResult({
                "robot": {"instance_id": 1, "file": "openbeam_test_arm.jbeam"},
                "joints": [{"name": n, "limits": {"lower": float(l[0]), "upper": float(l[1])}}
                           for n, l in zip(self.names, self.limits)],
                "end_effectors": [{"name": "tcp", "chain": list(self.names)}],
                "default_end_effector": "tcp"})
        if tool == "robot_state":
            return ToolResult(self._state())
        if tool == "robot_fk":
            q = self.q.copy()
            for n, v in (a.get("joints") or {}).items():
                q[self.names.index(n)] = v
            t = self.fk(q)
            return ToolResult({"pose": {"position": t[:3, 3].tolist(), "orientation": _quat_xyzw(t[:3, :3])}})
        if tool == "robot_set_joints":
            out = self._set(a.get("positions") or a.get("joints") or {})
            out["state"] = self._state()
            return ToolResult(out)
        if tool == "sim_mode":
            if a.get("mode") == "stepped":
                self.stepped = True
            elif a.get("mode") == "realtime":
                self.stepped = False
            return ToolResult({"mode": "stepped" if self.stepped else "realtime"})
        if tool == "sim_step":
            if not self.stepped:
                if raise_on_error:
                    raise ToolError(tool, "sim_step needs stepped mode")
                return ToolResult({"error": "sim_step needs stepped mode"})
            command = self._set(a.get("positions") or {})
            n = int(a.get("steps", 1))
            self.q = self.target + (self.q - self.target) * np.exp(-n * DT / self.tau)
            self.steps += n
            self.sim_time += n * DT
            return ToolResult({"steps": n, "command": command, "state": self._state()})
        if tool == "camera_frame":
            return ToolResult({"width": 0, "height": 0})
        raise ToolError(tool, f"the fake game has no tool '{tool}'")
