import functools

import gymnasium as gym
import numpy as np

from openbeam_gym import OpenBeamRobotEnv, Reach, SceneConfig
from openbeam_gym.fake import FakeGame


def _fake():
    return OpenBeamRobotEnv(FakeGame(), Reach(), SceneConfig())


def test_sync_vector_env_over_independent_games():
    vec = gym.vector.SyncVectorEnv([_fake, _fake, _fake])
    obs, info = vec.reset(seed=[1, 2, 3])
    assert obs.shape[0] == 3
    obs, r, term, trunc, info = vec.step(np.zeros((3, 4), dtype=np.float32))
    assert r.shape == (3,)
