import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
# 假設您使用的是官方提供的 SAIClient
from sai_rl import SAIClient 
from preprocessor import Preprocessor

# --------------------------------------------------------------------
# 2. 為 Stable-Baselines3 訓練準備環境包裝器 (Wrapper)
# --------------------------------------------------------------------
class SAIPPOWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.preprocessor = Preprocessor()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(87,), dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed_obs = self.preprocessor.modify_state(obs, info)
        return processed_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        processed_obs = self.preprocessor.modify_state(obs, info)
        return processed_obs, info


# --------------------------------------------------------------------
# 3. 執行訓練和提交
# --------------------------------------------------------------------

def run_training_and_submission():
    # 1. 初始化 SAI 客戶端
    # 請替換為您的 comp_id 和 api_key
    sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv") 

    # 2. 包裝環境 (使用 DummyVecEnv 模擬單環境批次)
    env = DummyVecEnv([lambda: SAIPPOWrapper(sai.make_env())])

    # 3. 創建模型：如果沒有 GPU，則使用 device="cpu"
    try:
        model = PPO("MlpPolicy", env, verbose=1, device="cuda")
    except:
        model = PPO("MlpPolicy", env, verbose=1, device="cpu")

    # 4. 訓練模型
    print("Starting model training (100,000 timesteps)...")
    model.learn(total_timesteps=1000) 

    # 5. 評估模型：將 Preprocessor 類別作為參數傳遞
    print("Starting evaluation...")
    try:
        sai.evaluate(model, preprocessor_class= Preprocessor) 
    except Exception as e:
        print(f"EvaluationError: Failed to evaluate. Details: {e}")

    # 6. 提交模型：將 Preprocessor 類別作為參數傳遞
    print("Starting submission...")
    try:
        sai.submit("Final Robust 87D PPO Model", model, Preprocessor)
    except Exception as e:
        print(f"SubmissionError: Failed to submit. Details: {e}")
    env.close()

if __name__ == '__main__':
    run_training_and_submission()
