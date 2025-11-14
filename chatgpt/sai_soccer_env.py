# sai_soccer_env.py
import gym
import numpy as np
from gym import spaces
from sai_rl import SAIClient
from preprocessor import Preprocessor  # 我也會給你這個檔案
import os

class SAISoccerEnv(gym.Env):
    """
    Gym wrapper for SAI Soccer Scene.
    If randomize_task=True, each reset will sample task_index randomly in [0,1,2].
    Otherwise, you can pass a fixed task_index when initializing.
    """
    metadata = {'render.modes': []}

    def __init__(self, comp_id=None, api_key=None, task_index=None, randomize_task=True):
        super().__init__()
        assert comp_id is not None, "comp_id is required (e.g. 'booster-soccer-showdown')"
        self.sai = SAIClient(comp_id=comp_id, api_key=api_key)
        self.task_index = task_index
        self.randomize_task = randomize_task
        # create an initial env (placeholder) to read action/obs space
        initial_task = 0 if task_index is None else int(task_index)
        self._make_inner_env(initial_task)
        # Preprocessor (wrap info->state)
        self.pre = Preprocessor()

    def _make_inner_env(self, task_idx):
        # sai.make_env accepts either name or index; here we use index
        # Replace with name if you prefer
        if hasattr(self, 'env') and self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        self.env = self.sai.make_env(task_idx)
        # assume env.action_space and env.observation_space exist
        self.action_space = self.env.action_space
        # We'll treat observation after preprocessing: decide shape via example reset
        raw_obs, info = self.env.reset()
        state = self.pre.modify_state(raw_obs, info)
        # state shape may be (1, N) — flatten to 1D
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state.shape[-1],), dtype=np.float32)

    def reset(self, *, seed=None, return_info=False, options=None):
        # choose task
        if self.randomize_task:
            tidx = int(np.random.randint(0, 3))
        else:
            tidx = 0 if self.task_index is None else int(self.task_index)
        # If current inner env differs, recreate
        # (some SAI clients may support switching at runtime; to be safe, recreate)
        self._make_inner_env(tidx)
        obs, info = self.env.reset()
        state = self.pre.modify_state(obs, info)
        state = state.reshape(-1).astype(np.float32)
        if return_info:
            return state, info
        return state

    def step(self, action):
        # action assumed to be in proper shape for env
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        state = self.pre.modify_state(obs, info)
        state = state.reshape(-1).astype(np.float32)
        return state, float(reward), done, info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

import numpy as np
from gymnasium import spaces

class SoccerMultiTaskEnv:
    """
    Multi-task environment wrapper.
    每次 reset 隨機選擇三個任務之一：
    0 = Penalty Kick with Goalie
    1 = Kick to Target
    2 = Precision Pass
    """
    def __init__(self):
        from sai_rl import SAIClient
        
        self.sai = SAIClient(comp_id="booster-soccer-showdown", api_key="你的APIKEY")
        
        # 建立三個單任務環境
        self.envs = [
            self.sai.make_env("LowerT1PenaltyKickWithGoalie-v0"),
            self.sai.make_env("LowerT1KickToTarget-v0"),
            self.sai.make_env("LowerT1PrecisionPass-v0")
        ]
        self.current_env = None
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space

    def reset(self):
        # 隨機選擇任務
        task_index = np.random.randint(0, len(self.envs))
        self.current_env = self.envs[task_index]
        obs, info = self.current_env.reset()
        info["task_index"] = np.array([task_index])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.current_env.step(action)
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.current_env.render()

    def close(self):
        for env in self.envs:
            env.close()

