from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from traffic_env_context import TrafficEnv
import os


# ======================================================
# PATH SETUP
# ======================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models", "ppo_context")

os.makedirs(MODEL_DIR, exist_ok=True)


# ================= SAVE CALLBACK =================
class SaveCallback(BaseCallback):
    def __init__(self, save_path, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:

        if self.n_calls % 1000 == 0:

            print(f"🔥 Saving model at step: {self.n_calls}")

            os.makedirs(self.save_path, exist_ok=True)

            # ================= SAVE MODEL =================
            self.model.save(
                os.path.join(
                    self.save_path,
                    f"ppo_{self.n_calls}"
                )
            )

            # ================= SAVE NORMALIZATION =================
            self.training_env.save(
                os.path.join(
                    self.save_path,
                    "vec_normalize.pkl"
                )
            )

        return True


# ================= ENV SETUP =================
def make_env():
    return TrafficEnv()


env = DummyVecEnv([make_env])

# ================= NORMALIZATION =================
env = VecNormalize(
    env,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0
)


# ======================================================
# MODEL PATH
# ======================================================

model_path = os.path.join(
    MODEL_DIR,
    "ppo_15000.zip"
)


# ================= LOAD / CREATE MODEL =================
if os.path.exists(model_path):

    print("✅ Loading previous Context PPO model")

    model = PPO.load(
        model_path,
        env=env
    )

    # FORCE NEW LEARNING RATE
    model.learning_rate = 0.0005
    model.lr_schedule = lambda _: 0.0005

else:

    print("🚀 No previous model found, starting fresh")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0005,
        n_steps=2048,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )


# ================= TRAIN =================

callback = SaveCallback(save_path=MODEL_DIR)

print("🚦 Training starting... (50k steps recommended)")

model.learn(
    total_timesteps=50000,
    callback=callback,
    reset_num_timesteps=False
)


# ================= SAVE FINAL =================

model.save(
    os.path.join(
        MODEL_DIR,
        "final_context_ppo"
    )
)

env.save(
    os.path.join(
        MODEL_DIR,
        "vec_normalize.pkl"
    )
)

print("✅ Context PPO Training Finished")

env.close()