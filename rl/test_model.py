from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from traffic_env import TrafficEnv
import traci

MODEL_PATH = "../models/ppo_43000.zip"

# ===== LOAD ENV WITH GUI =====
env = DummyVecEnv([lambda: TrafficEnv()])
env = VecNormalize.load("../models/vec_normalize.pkl", env)

env.training = False
env.norm_reward = False

# 🔥 FORCE GUI
env.envs[0].sumo_cmd[0] = "sumo-gui"

# ===== LOAD MODEL =====
model = PPO.load(MODEL_PATH, env=env)

obs = env.reset()

print("🚀 Testing started with GUI...\n")

total_reward = 0

for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, done, info = env.step(action)

    total_reward += reward[0]

    # ===== PRINT EVERY 100 STEPS =====
    if step % 100 == 0:
        lane_ids = traci.lane.getIDList()

        # ❗ Ignore U-turn lanes (adjust name if needed)
        filtered_lanes = [l for l in lane_ids if "uturn" not in l.lower()]

        waiting_time = sum(traci.lane.getWaitingTime(l) for l in filtered_lanes)
        queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in filtered_lanes)

        # 🔥 Clip waiting to avoid explosion
        waiting_time = min(waiting_time, 10000)

        print(f"Step: {step}")
        print(f"   Action: {action}")
        print(f"   Reward: {reward[0]:.3f}")
        print(f"   Waiting Time: {waiting_time:.2f}")
        print(f"   Queue Length: {queue}")
        print("--------------------------------------------------")

    # ===== HANDLE DONE =====
    if done[0]:
        print("🔁 Episode finished, resetting...\n")
        obs = env.reset()

print("\n✅ Testing complete")
print(f"🎯 Total Reward: {total_reward:.2f}")

traci.close()