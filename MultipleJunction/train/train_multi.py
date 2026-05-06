import sys
import os
import numpy as np

# ================= PATH FIX =================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import MultiTrafficEnv

import gymnasium as gym


# ================= SAVE CALLBACK =================
class SaveCallback(BaseCallback):
    def __init__(self, save_path, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            print(f"🔥 Saving model at step: {self.n_calls}")

            os.makedirs(self.save_path, exist_ok=True)

            self.model.save(os.path.join(self.save_path, f"ppo_{self.n_calls}"))
            self.training_env.save(os.path.join(self.save_path, "vec_normalize.pkl"))

        return True

# ================= MULTI-AGENT WRAPPER =================
class FlattenMultiAgentEnv(gym.Env):
    def __init__(self, env):
        super().__init__()

        self.env = env

        # 🔥 IMPORTANT: get shape ONCE here
        states, _ = self.env.reset()

        self.num_agents = len(states)
        flat_state = states.flatten()

        # ✅ FIXED observation space
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=flat_state.shape, dtype=np.float32
        )

        # ✅ FIXED action space
        self.action_space = gym.spaces.MultiDiscrete([2] * self.num_agents)

    def reset(self, seed=None, options=None):
        states, _ = self.env.reset()
        return states.flatten(), {}

    def step(self, action):
        actions = np.array(action)

        states, reward, done, _, _ = self.env.step(actions)

        return states.flatten(), reward, done, False, {}

    def close(self):
        self.env.close()


# ================= ENV SETUP =================
def make_env():
    return FlattenMultiAgentEnv(MultiTrafficEnv())


env = DummyVecEnv([make_env])

# ================= NORMALIZATION =================
env = VecNormalize(
    env,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0
)


# ================= PPO MODEL =================
model_path = "../models/ppo_multi.zip"

if os.path.exists(model_path):
    print("✅ Loading previous model")
    model = PPO.load(model_path, env=env)

    model.learning_rate = 0.0003
    model.lr_schedule = lambda _: 0.0003

else:
    print("⚠️ Starting new model")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.00015,
        n_steps=256,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.08,
        ent_coef=0.01,
        vf_coef=0.8,
        max_grad_norm=0.3,
    )


# ================= TRAIN =================
callback = SaveCallback(save_path="../models")

print("🚀 Training Multi-Agent Model...")
model.learn(
    total_timesteps=10000,
    callback=callback,
    log_interval=1
)


# ================= SAVE =================
os.makedirs("../models", exist_ok=True)

model.save("../models/final_multi_agent_model")
env.save("../models/vec_normalize.pkl")

print("✅ Training Finished")

env.close()