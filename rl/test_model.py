from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from traffic_env import TrafficEnv
import traci
import numpy as np

# ================= SETTINGS =================
MODEL_PATH = "../models/ppo_22000_old"
USE_MAIN_LOGIC = True   # 🔥 True = main.py logic, False = old test

# ================= ENV =================
env = DummyVecEnv([lambda: TrafficEnv()])
env = VecNormalize.load("../models/vec_normalize.pkl", env)

env.training = False
env.norm_reward = False

# GUI
env.envs[0].sumo_cmd[0] = "sumo-gui"

# ================= LOAD MODEL =================
model = PPO.load(MODEL_PATH, env=env)

print(f"🚀 Testing started for model: {MODEL_PATH}\n")

# =========================================================
# ================= MAIN.PY STYLE TEST =====================
# =========================================================
if USE_MAIN_LOGIC:

    obs = env.reset()
    start_time = traci.simulation.getTime()

    model.current_phase = 0
    model.last_action_time = 0

    emission_history = []
    total_vehicles_passed = 0
    total_time_spent = 0

    vehicle_entry_time = {}
    vehicle_exit_time = {}
    vehicle_last_lane = {}
    lane_pass_count = {}

    tls_ids = traci.trafficlight.getIDList()

    while traci.simulation.getMinExpectedNumber() > 0:

        if traci.simulation.getTime() - start_time >= 1000:
            break

        traci.simulationStep()

        vehicles = traci.vehicle.getIDList()

        # ===== EMISSION =====
        total_emission = sum(traci.vehicle.getCO2Emission(v) for v in vehicles)
        emission_history.append(total_emission)

        # ===== ENTRY TRACK =====
        for v in vehicles:
            if v not in vehicle_entry_time:
                vehicle_entry_time[v] = traci.simulation.getTime()
            try:
                vehicle_last_lane[v] = traci.vehicle.getLaneID(v)
            except:
                pass

        # ===== RL CONTROL =====
        tl = tls_ids[0]
        current_time = traci.simulation.getTime()

        if current_time - model.last_action_time >= 5:

            waiting_time = sum(traci.lane.getWaitingTime(l) for l in traci.lane.getIDList())
            queue_length = sum(traci.lane.getLastStepHaltingNumber(l) for l in traci.lane.getIDList())
            vehicle_count = len(vehicles)
            current_phase = traci.trafficlight.getPhase(tl)

            state = np.array([
                waiting_time / 1000.0,
                queue_length / 50.0,
                vehicle_count / 100.0,
                current_phase / 4.0
            ], dtype=np.float32)

            state = env.normalize_obs(state)

            action, _ = model.predict(state, deterministic=True)

            if action == 1:
                model.current_phase = 2 if model.current_phase == 0 else 0

            traci.trafficlight.setPhase(tl, model.current_phase)
            model.last_action_time = current_time

        # ===== EXIT TRACK =====
        current_vehicles = set(vehicles)

        for v in list(vehicle_entry_time.keys()):
            if v not in current_vehicles and v not in vehicle_exit_time:
                exit_t = traci.simulation.getTime()
                vehicle_exit_time[v] = exit_t

                travel_time = exit_t - vehicle_entry_time[v]
                total_time_spent += travel_time
                total_vehicles_passed += 1

                lane = vehicle_last_lane.get(v, "unknown")
                lane_pass_count[lane] = lane_pass_count.get(lane, 0) + 1

    # ===== RESULT =====
    print("\n========== 🚦 MAIN MODE RESULT ==========")

    avg_time = total_time_spent / (total_vehicles_passed + 1e-6)
    total_co2 = sum(emission_history)
    avg_co2 = total_co2 / len(emission_history)

    print(f"Total Vehicles Passed: {total_vehicles_passed}")
    print(f"🌿 Total CO2 Emission: {total_co2:.2f}")
    print(f"🌿 Average CO2 per Step: {avg_co2:.2f}")
    print(f"⏱ Average Time per Vehicle: {avg_time:.2f}")

    print("=========================================\n")

# =========================================================
# ================= OLD TEST MODE ==========================
# =========================================================
else:

    obs = env.reset()
    start_time = traci.simulation.getTime()

    total_reward = 0
    emission_history = []
    total_vehicles_passed = 0
    total_time_spent = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        total_reward += reward[0]
        total_vehicles_passed += traci.simulation.getArrivedNumber()

        total_time_spent += sum(
            traci.lane.getWaitingTime(l)
            for l in traci.lane.getIDList()
        )

        vehicles = traci.vehicle.getIDList()
        total_emission = sum(traci.vehicle.getCO2Emission(v) for v in vehicles)
        emission_history.append(total_emission)

        if traci.simulation.getTime() - start_time >= 1200:
            break

    print("\n========== 🚦 TEST MODE RESULT ==========")

    avg_time = total_time_spent / (total_vehicles_passed + 1e-6)
    total_co2 = sum(emission_history)
    avg_co2 = total_co2 / len(emission_history)

    print(f"Total Vehicles Passed: {total_vehicles_passed}")
    print(f"🌿 Total CO2 Emission: {total_co2:.2f}")
    print(f"🌿 Average CO2 per Step: {avg_co2:.2f}")
    print(f"⏱ Average Time per Vehicle: {avg_time:.2f}")

    print("=========================================\n")

traci.close()