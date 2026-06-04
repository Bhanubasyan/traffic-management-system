import os
import sys
import random

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traffic_env_context import TrafficEnv

import traci
import numpy as np
import csv


# ======================================================
# PATH SETUP
# ======================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "ppo_context"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "ppo_151000.zip"
)

if not os.path.exists(MODEL_PATH):
    alt_path = os.path.join(MODEL_DIR, "final_context_ppo.zip")
    if os.path.exists(alt_path):
        MODEL_PATH = alt_path

VEC_PATH = os.path.join(
    MODEL_DIR,
    "vec_normalize.pkl"
)


# ======================================================
# WEATHER + ZONE LABELS
# ======================================================

weather_map = {
    0: "Clear",
    1: "Rain",
    2: "Fog"
}

zone_map = {
    0: "Residential",
    1: "Commercial",
    2: "Industrial",
    3: "School"
}


# ======================================================
# RUN SIMULATION
# ======================================================

def run_simulation(run_id, sim_time, scenario):

    print(f"\n🚀 Run {run_id} | Scenario: {scenario} | Time: {sim_time}s\n")

    np.random.seed()
    random.seed()

    # ======================================================
    # ENV
    # ======================================================

    env = DummyVecEnv([lambda: TrafficEnv()])

    env = VecNormalize.load(
        VEC_PATH,
        env
    )

    env.training = False
    env.norm_reward = False

    env.envs[0].sumo_cmd[0] = "sumo-gui"

    seed = np.random.randint(1, 10000)

    env.envs[0].sumo_cmd += [
        "--seed",
        str(seed)
    ]

    # ======================================================
    # LOAD MODEL
    # ======================================================

    print("✅ Loading Context PPO model")

    model = PPO.load(
        MODEL_PATH,
        env=env
    )

    obs = env.reset()

    start_time = traci.simulation.getTime()

    model.current_phase = 0
    model.last_action_time = 0

    # ======================================================
    # TRACKERS
    # ======================================================

    emission_history = []

    fuel_history = []

    queue_history = []

    waiting_history = []

    speed_history = []

    total_vehicles_passed = 0

    total_time_spent = 0

    vehicle_entry_time = {}

    vehicle_exit_time = {}

    vehicle_last_lane = {}

    tls_ids = traci.trafficlight.getIDList()

    # ======================================================
    # MAIN LOOP
    # ======================================================

    while traci.simulation.getMinExpectedNumber() > 0:

        if traci.simulation.getTime() - start_time >= sim_time:
            break

        traci.simulationStep()

        vehicles = traci.vehicle.getIDList()

        # ======================================================
        # CO2 EMISSION (kg)
        # ======================================================

        total_emission = sum(
            traci.vehicle.getCO2Emission(v)
            for v in vehicles
        ) / 1_000_000

        emission_history.append(total_emission)

        # ======================================================
        # FUEL CONSUMPTION (liters approx)
        # ======================================================

        total_fuel = sum(
            traci.vehicle.getFuelConsumption(v)
            for v in vehicles
        ) / 1_000_000

        fuel_history.append(total_fuel)

        # ======================================================
        # WAITING + QUEUE
        # ======================================================

        waiting_time = sum(
            traci.lane.getWaitingTime(l)
            for l in traci.lane.getIDList()
        )

        queue_length = sum(
            traci.lane.getLastStepHaltingNumber(l)
            for l in traci.lane.getIDList()
        )

        queue_history.append(queue_length)

        waiting_history.append(waiting_time)

        # ======================================================
        # SPEED (km/h)
        # ======================================================

        if len(vehicles) > 0:

            avg_speed = np.mean([
                traci.vehicle.getSpeed(v)
                for v in vehicles
            ]) * 3.6

            speed_history.append(avg_speed)

        # ======================================================
        # ENTRY TRACKING
        # ======================================================

        for v in vehicles:

            if v not in vehicle_entry_time:
                vehicle_entry_time[v] = traci.simulation.getTime()

            try:
                vehicle_last_lane[v] = traci.vehicle.getLaneID(v)

            except:
                pass

        # ======================================================
        # RL CONTROL
        # ======================================================

        tl = tls_ids[0]

        current_time = traci.simulation.getTime()

        if current_time - model.last_action_time >= 5:

            vehicle_count = len(vehicles)

            current_phase = traci.trafficlight.getPhase(tl)

            weather = env.envs[0].weather / 2.0

            zone_type = env.envs[0].zone_type / 3.0

            road_capacity = env.envs[0].road_capacity

            arrival_rate = (
                traci.simulation.getDepartedNumber() / 5.0
            )

            arrival_rate = min(arrival_rate, 1.0)

            sim_time_now = traci.simulation.getTime()

            hour = (sim_time_now / 3600.0) % 24

            hour_sin = (
                np.sin(2 * np.pi * hour / 24) + 1
            ) / 2

            hour_cos = (
                np.cos(2 * np.pi * hour / 24) + 1
            ) / 2

            state = np.array([

                waiting_time / 1000.0,

                queue_length / 50.0,

                vehicle_count / 100.0,

                current_phase / 4.0,

                weather,

                hour_sin,

                hour_cos,

                zone_type,

                road_capacity,

                arrival_rate

            ], dtype=np.float32)

            state = env.normalize_obs(state)

            action, _ = model.predict(
                state,
                deterministic=True
            )

            if action == 1:
                model.current_phase = (
                    2 if model.current_phase == 0 else 0
                )

            traci.trafficlight.setPhase(
                tl,
                model.current_phase
            )

            model.last_action_time = current_time

        # ======================================================
        # EXIT TRACKING
        # ======================================================

        current_vehicles = set(vehicles)

        for v in list(vehicle_entry_time.keys()):

            if v not in current_vehicles and v not in vehicle_exit_time:

                exit_t = traci.simulation.getTime()

                vehicle_exit_time[v] = exit_t

                travel_time = (
                    exit_t - vehicle_entry_time[v]
                )

                total_time_spent += travel_time

                total_vehicles_passed += 1

    # ======================================================
    # FINAL METRICS
    # ======================================================

    avg_travel_time = (
        total_time_spent /
        (total_vehicles_passed + 1e-6)
    )

    total_co2 = sum(emission_history)

    avg_co2 = (
        total_co2 /
        (len(emission_history) + 1e-6)
    )

    total_fuel_used = sum(fuel_history)

    avg_fuel = (
        total_fuel_used /
        (len(fuel_history) + 1e-6)
    )

    avg_queue = np.mean(queue_history)

    avg_wait = (
        np.mean(waiting_history) /
        (total_vehicles_passed + 1e-6)
    )

    avg_speed = np.mean(speed_history)

    fairness = 1 - (
        np.std(queue_history) /
        (np.mean(queue_history) + 1e-6)
    )

    throughput = (
        total_vehicles_passed / sim_time
    )

    # ======================================================
    # RESULT PRINT
    # ======================================================

    print("\n========== 🚦 CONTEXT PPO RESULT ==========\n")

    print(f"Scenario                 : {scenario}")
    print(f"Simulation Time          : {sim_time} sec")

    print(f"\n🚗 Vehicles Passed        : {total_vehicles_passed} vehicles")

    print(f"\n⏳ Avg Waiting Time       : {avg_wait:.2f} sec/vehicle")

    print(f"🛑 Avg Queue Length       : {avg_queue:.2f} vehicles")

    print(f"\n🌿 Total CO₂ Emission     : {total_co2:.2f} kg")

    print(f"🌿 Avg CO₂ Emission       : {avg_co2:.4f} kg")

    print(f"\n⛽ Total Fuel Consumption : {total_fuel_used:.2f} L")

    print(f"⛽ Avg Fuel Consumption   : {avg_fuel:.4f} L")

    print(f"\n🚀 Average Speed          : {avg_speed:.2f} km/h")

    print(f"\n⚖️ Fairness Index         : {fairness:.2f}")

    print(f"\n🚀 Throughput             : {throughput:.2f} vehicles/sec")

    print(
        f"\n🌧 Weather                : "
        f"{weather_map[env.envs[0].weather]}"
    )

    print(
        f"🏙 Zone Type              : "
        f"{zone_map[env.envs[0].zone_type]}"
    )

    print(f"🛣 Road Capacity          : {env.envs[0].road_capacity:.2f}")

    print("\n===========================================\n")

    traci.close()

    return {

        "Run_ID": run_id,

        "Seed": seed,

        "Scenario": scenario,

        "Simulation_Time_sec": sim_time,

        "Vehicles_Passed": total_vehicles_passed,

        "Avg_Travel_Time_sec": avg_travel_time,

        "Avg_Waiting_Time_sec_per_vehicle": avg_wait,

        "Avg_Queue_Length_vehicles": avg_queue,

        "Total_CO2_kg": total_co2,

        "Avg_CO2_kg": avg_co2,

        "Total_Fuel_L": total_fuel_used,

        "Avg_Fuel_L": avg_fuel,

        "Avg_Speed_kmh": avg_speed,

        "Fairness_Index": fairness,

        "Weather": weather_map[env.envs[0].weather],

        "Zone_Type": zone_map[env.envs[0].zone_type],

        "Road_Capacity": env.envs[0].road_capacity,

        "Throughput_vehicles_per_sec": throughput
    }


# ======================================================
# MULTI RUN
# ======================================================

if __name__ == "__main__":

    NUM_RUNS = 3

    SIM_TIMES = [60, 120, 300]

    SCENARIOS = [
        "Low",
        "Medium",
        "High"
    ]

    all_results = []

    run_id = 1

    for scenario in SCENARIOS:

        for sim_time in SIM_TIMES:

            for _ in range(NUM_RUNS):

                result = run_simulation(
                    run_id,
                    sim_time,
                    scenario
                )

                all_results.append(result)

                run_id += 1

    # ======================================================
    # SAVE CSV
    # ======================================================

    keys = all_results[0].keys()

    from datetime import datetime

    csv_name = (
        f"context_results_"
        f"{datetime.now():%Y%m%d_%H%M%S}.csv"
    )

    with open(csv_name, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=keys
        )

        writer.writeheader()

        writer.writerows(all_results)

    print(f"\n✅ All runs completed & saved to {csv_name}")