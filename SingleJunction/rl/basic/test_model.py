from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from traffic_env import TrafficEnv
import traci
import numpy as np
import csv
import random

# ================= SETTINGS =================
MODEL_PATH = "../models/ppo_22000_old"
USE_MAIN_LOGIC = True

# ================= FUNCTION WRAPPER =================
def run_simulation(run_id, sim_time, scenario):   # 🔥 added sim_time + scenario

    print(f"\n🚀 Run {run_id} | Scenario: {scenario} | Time: {sim_time}s\n")

    # ===== RANDOMNESS =====
    np.random.seed()
    random.seed()

    # ================= ENV =================
    env = DummyVecEnv([lambda: TrafficEnv()])
    env = VecNormalize.load("../models/vec_normalize.pkl", env)

    env.training = False
    env.norm_reward = False

    env.envs[0].sumo_cmd[0] = "sumo-gui"

    # 🔥 add SUMO randomness
    env.envs[0].sumo_cmd += ["--seed", str(np.random.randint(1,10000))]

    # ================= LOAD MODEL =================
    model = PPO.load(MODEL_PATH, env=env)

    obs = env.reset()
    start_time = traci.simulation.getTime()

    model.current_phase = 0
    model.last_action_time = 0

    # ===== RESET TRACKERS =====
    emission_history = []
    total_vehicles_passed = 0
    total_time_spent = 0

    vehicle_entry_time = {}
    vehicle_exit_time = {}
    vehicle_last_lane = {}
    lane_pass_count = {}

    tls_ids = traci.trafficlight.getIDList()

    # ================= MAIN LOOP =================
    while traci.simulation.getMinExpectedNumber() > 0:

        # 🔥 dynamic simulation time
        if traci.simulation.getTime() - start_time >= sim_time:
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

            # 🔥 IMPORTANT FIX (variation)
            action, _ = model.predict(state, deterministic=False)

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
    avg_time = total_time_spent / (total_vehicles_passed + 1e-6)
    total_co2 = sum(emission_history)
    avg_co2 = total_co2 / (len(emission_history) + 1e-6)

    print(f"Run {run_id} → Vehicles: {total_vehicles_passed}, Avg Time: {avg_time:.2f}")

    traci.close()

    return {
        "Run_ID": run_id,
        "Scenario": scenario,              # 🔥 added
        "Simulation_Time": sim_time,       # 🔥 added
        "Vehicles_Passed": total_vehicles_passed,
        "Avg_Travel_Time": avg_time,
        "Total_CO2": total_co2,
        "Avg_CO2": avg_co2
    }


# ================= MULTI RUN =================
if __name__ == "__main__":

    NUM_RUNS = 3
    SIM_TIMES = [60, 120, 300]

    # 🔥 you manually control traffic → just label it
    SCENARIOS = ["Low", "Medium", "High"]  

    all_results = []
    run_id = 1

    for scenario in SCENARIOS:
        for sim_time in SIM_TIMES:
            for _ in range(NUM_RUNS):

                result = run_simulation(run_id, sim_time, scenario)
                all_results.append(result)

                run_id += 1

    # ===== SAVE =====
    keys = all_results[0].keys()

    with open("results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_results)

    print("\n✅ All runs completed & saved to results.csv")