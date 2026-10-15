import numpy as np
import pytest

from openbeam_gym import OpenBeamRobotEnv, Reach, SceneConfig
from openbeam_gym.fake import FakeGame
from openbeam_gym.mcp import _decode_tool_result, ToolError


def make(**kw):
    return OpenBeamRobotEnv(FakeGame(), Reach(), SceneConfig(), **kw)


def test_spaces_and_reset():
    env = make()
    obs, info = env.reset(seed=3)
    assert env.observation_space.contains(obs)
    assert env.action_space.shape == (4,)
    assert "goal" in info and len(info["goal"]) == 3
    assert env.physics_steps_per_action == 100  # 20 Hz control at 2 kHz physics


def test_step_is_one_call_and_moves_toward_targets():
    game = FakeGame()
    env = OpenBeamRobotEnv(game, Reach(), SceneConfig())
    env.reset(seed=0)
    before = game.calls
    obs, r, term, trunc, info = env.step(np.ones(4, dtype=np.float32))
    assert game.calls - before == 1          # sim_step only
    assert np.all(game.target > 0)           # delta mode raised every target
    assert info["distance"] >= 0 and isinstance(r, float)


def test_seeded_goals_repeat():
    a, b = make(), make()
    _, ia = a.reset(seed=11)
    _, ib = b.reset(seed=11)
    assert np.allclose(ia["goal"], ib["goal"])


def test_an_oracle_policy_reaches():
    env = make(action_mode="absolute")
    env.reset(seed=5)
    goal_q = env.task.goal_joints
    action = np.array([2 * (goal_q[n] - env.limits[n][0]) / (env.limits[n][1] - env.limits[n][0]) - 1
                       for n in env.joints], dtype=np.float32)
    for _ in range(40):
        _, _, term, trunc, info = env.step(action)
        if term:
            break
    assert term and info["is_success"]


def test_truncation():
    env = make()
    env.task.max_steps = 3
    env.reset(seed=1)
    for i in range(3):
        _, _, term, trunc, _ = env.step(np.zeros(4, dtype=np.float32))
    assert trunc and not term


def test_tool_errors_raise():
    with pytest.raises(ToolError):
        _decode_tool_result("x", {"isError": True, "content": [{"type": "text", "text": "{\"error\":\"no\"}"}]}, 0, True)


def test_passive_checker():
    # check_env's determinism test does not apply: a soft-body reset is not
    # bit-identical (the arm is driven home and settled, not restored).
    from gymnasium.utils.passive_env_checker import check_action_space, check_observation_space
    env = make()
    check_observation_space(env.observation_space)
    check_action_space(env.action_space)
