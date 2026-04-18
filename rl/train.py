from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from traffic_env import TrafficEnv
import os


# ================= SAVE CALLBACK =================
class SaveCallback(BaseCallback):
    def __init__(self, save_path, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            print(f"🔥 Saving model at step: {self.n_calls}")

            os.makedirs(self.save_path, exist_ok=True)

            # Save model
            self.model.save(os.path.join(self.save_path, f"ppo_{self.n_calls}"))

            #  THIS IS THE MAIN FIX
            self.training_env.save(os.path.join(self.save_path, "vec_normalize.pkl"))

        return True


# ================= ENV SETUP =================
def make_env():
    return TrafficEnv()

env = DummyVecEnv([make_env])

# 🔥 VERY IMPORTANT: normalize observations + rewards
env = VecNormalize(
    env,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0
)


# ================= PPO MODEL =================
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,

    # 🔧 LEARNING CONTROL
    learning_rate=0.001,        # faster learning than 0.0003
    n_steps=2048,               # more stable updates
    batch_size=128,             # better gradient estimate

    # 🔧 RL CORE
    gamma=0.99,
    gae_lambda=0.95,

    # 🔧 PPO BEHAVIOR
    clip_range=0.2,
    ent_coef=0.02,              # more exploration (important for traffic)
    vf_coef=0.5,

    # 🔧 STABILITY
    max_grad_norm=0.5,
)


# ================= TRAIN =================
callback = SaveCallback(save_path="../models")

print("🚀 Training starting... (100k steps recommended)")
model.learn(total_timesteps=100000, callback=callback)


# ================= SAVE FINAL =================
os.makedirs("../models", exist_ok=True)

model.save("../models/final_ppo_model")
env.save("../models/vec_normalize.pkl")   #  IMPORTANT


print("✅ Training Finished")
env.close()