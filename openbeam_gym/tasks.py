"""Task definitions: what an episode asks of the robot, and its reward.

A task sees the environment's latest robot_state (`env.state`), may call game
tools through `env.game`, and returns (reward, terminated, info) per step.
Rewards are in metres-ish units so returns compare across robots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class Task:
    name = "task"
    max_steps = 100
    extra_dim = 0                       # length of observation_extra()

    def reset(self, env, rng: np.random.Generator) -> dict:
        return {}

    def observation_extra(self, env) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)

    def reward(self, env, action: np.ndarray) -> tuple[float, bool, dict]:
        raise NotImplementedError


def ee_position(state: dict) -> np.ndarray:
    return np.asarray(state["end_effector"]["position"], dtype=np.float64)


@dataclass
class Reach(Task):
    """Bring the end effector within `tolerance` metres of a goal point.

    Goals come from joint values drawn inside `range_fraction` of each chain
    joint's range around the home pose. `goal_source="fk"` takes the model's
    FK of them (one robot_fk call per reset); `"sim"` (the default) drives the
    robot there, settles and takes the end effector the soft body actually
    reached, so a goal is reachable even where the arm sags (one sim_step).
    Reward: -distance - action_cost * |a|^2, plus `success_bonus` on the step
    that reaches the goal, which terminates.
    """

    tolerance: float = 0.03
    range_fraction: float = 0.35
    max_steps: int = 80
    success_bonus: float = 1.0
    action_cost: float = 0.01
    goal_source: str = "sim"
    goal_settle_steps: int = 2000
    name: str = "reach"
    extra_dim: int = 6

    def reset(self, env, rng):
        names = env.chain_joints
        q = {}
        for name in names:
            lo, hi = env.limits[name]
            home = env.home.get(name, 0.5 * (lo + hi))
            half = 0.5 * (hi - lo) * self.range_fraction
            q[name] = float(np.clip(home + rng.uniform(-half, half), lo, hi))
        if self.goal_source == "sim":
            env._sim_step(q, self.goal_settle_steps)
            self.goal = ee_position(env.state).copy()
            env._sim_step(env._targets, env.scene.settle_steps)   # back home
        else:
            fk = env.game.call("robot_fk", {**env.robot_arg, "joints": q, "end_effector": env.end_effector}).data
            self.goal = np.asarray(fk["pose"]["position"], dtype=np.float64)
        self.goal_joints = q
        return {"goal": self.goal.tolist()}

    def observation_extra(self, env):
        ee = ee_position(env.state)
        return np.concatenate([self.goal, self.goal - ee]).astype(np.float32)

    def distance(self, env) -> float:
        return float(np.linalg.norm(self.goal - ee_position(env.state)))

    def reward(self, env, action):
        d = self.distance(env)
        success = d < self.tolerance
        r = -d - self.action_cost * float(np.sum(np.square(action)))
        if success:
            r += self.success_bonus
        return r, success, {"distance": d, "is_success": success}


@dataclass
class Pick(Task):
    """Grasp an object actor and lift it `lift` metres.

    Needs a scene with a graspable actor (OpenBeam's `so101_tabletop` scene: the 3 cm
    cube) and a robot with a gripper joint. The object's pose comes from the
    game's actor_poses tool. Shaped reward: approach (-gripper-to-object
    distance), a grasp term while the jaws are closed near the object, and
    the lift height; success when the object is `lift` above its start.
    """

    object_selector: str = "cube"
    lift: float = 0.05
    max_steps: int = 150
    gripper_joint: str = "gripper"
    closed_below: float = 0.4          # gripper joint value (rad) counted as closed
    name: str = "pick"
    extra_dim: int = 6

    def _object(self, env) -> np.ndarray:
        rows = env.game.call("actor_poses", {}).data.get("actors", [])
        for row in rows:
            label = f"{row.get('file', '')} {row.get('name', '')}".lower()
            if self.object_selector.lower() in label and row.get("instance_id") != env.robot_id:
                return np.asarray(row["position"], dtype=np.float64)
        raise RuntimeError(f"no actor matching '{self.object_selector}' in the scene")

    def _object_base(self, env) -> np.ndarray:
        # World -> base frame with the robot_state base pose.
        p = self._object(env)
        return env.world_to_base(p)

    def reset(self, env, rng):
        self.start = self._object_base(env)
        self.obj = self.start.copy()
        return {"object_start": self.start.tolist()}

    def observation_extra(self, env):
        self.obj = self._object_base(env)
        return np.concatenate([self.obj, self.obj - ee_position(env.state)]).astype(np.float32)

    def reward(self, env, action):
        ee = ee_position(env.state)
        d = float(np.linalg.norm(self.obj - ee))
        grip = env.joint_position(self.gripper_joint)
        closed = grip < self.closed_below
        height = float(self.obj[2] - self.start[2])
        success = height > self.lift
        r = -d + (0.5 if closed and d < 0.03 else 0.0) + 5.0 * max(0.0, height)
        if success:
            r += 2.0
        return r, success, {"distance": d, "lift": height, "is_success": success}


TASKS = {"reach": Reach, "pick": Pick}
