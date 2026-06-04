import argparse
import csv
import os
import random
import sys
from datetime import datetime

import numpy as np
import traci
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_FILE = os.path.join(BASE_DIR, "config", "simulation.sumocfg")
CONTEXT_MODEL_DIR = os.path.join(BASE_DIR, "models", "ppo_context")
CONTEXT_MODEL_PATHS = [
    os.path.join(CONTEXT_MODEL_DIR, "ppo_151000.zip"),
    os.path.join(CONTEXT_MODEL_DIR, "final_context_ppo.zip"),
]
CONTEXT_VEC_PATH = os.path.join(CONTEXT_MODEL_DIR, "vec_normalize.pkl")

sys.path.insert(0, os.path.join(BASE_DIR, "rl", "context"))
from traffic_env_context import TrafficEnv  # noqa: E402


WEATHER_MAP = {
    0: "Clear",
    1: "Rain",
    2: "Fog",
}

ZONE_MAP = {
    0: "Residential",
    1: "Commercial",
    2: "Industrial",
    3: "School",
}

VEHICLE_TYPE_COLUMNS = {
    "car": "PassedCars",
    "bike": "PassedBikes",
    "truck": "PassedTrucks",
    "bus": "PassedBuses",
    "motorcycle": "PassedMotorcycles",
    "rickshaw": "PassedRickshaws",
    "emergency": "PassedEmergencyVehicles",
    "other": "PassedOtherVehicles",
}


def _vehicle_category(type_id):
    type_id = type_id.lower()

    if "rickshaw" in type_id or type_id == "auto":
        return "rickshaw"
    if (
        "motorcycle" in type_id or
        "motorbike" in type_id or
        "moped" in type_id or
        "scooter" in type_id
    ):
        return "motorcycle"
    if "bike" in type_id or "bicycle" in type_id:
        return "bike"
    if "truck" in type_id or "lorry" in type_id or "tanker" in type_id:
        return "truck"
    if "bus" in type_id:
        return "bus"
    if (
        "car" in type_id or
        "passenger" in type_id or
        "veh_passenger" in type_id or
        "jeep" in type_id or
        "van" in type_id
    ):
        return "car"
    if "ambulance" in type_id or "emergency" in type_id:
        return "emergency"

    return "other"


def _start_sumo(binary, seed):
    traci.start([
        binary,
        "-c",
        CONFIG_FILE,
        "--seed",
        str(seed),
        "--no-step-log",
        "true",
        "--no-warnings",
        "true",
        "--waiting-time-memory",
        "1000",
        "--time-to-teleport",
        "-1",
    ])


def _metric_track_step(trackers):
    vehicles = traci.vehicle.getIDList()
    now = traci.simulation.getTime()

    trackers["emission_history"].append(
        sum(traci.vehicle.getCO2Emission(v) for v in vehicles) / 1_000_000
    )
    trackers["fuel_history"].append(
        sum(traci.vehicle.getFuelConsumption(v) for v in vehicles) / 1_000_000
    )
    trackers["waiting_history"].append(
        sum(traci.lane.getWaitingTime(lane) for lane in traci.lane.getIDList())
    )
    trackers["queue_history"].append(
        sum(traci.lane.getLastStepHaltingNumber(lane) for lane in traci.lane.getIDList())
    )

    if vehicles:
        trackers["speed_history"].append(
            float(np.mean([traci.vehicle.getSpeed(v) for v in vehicles]) * 3.6)
        )

    for vehicle_id in vehicles:
        trackers["entry_time"].setdefault(vehicle_id, now)
        if vehicle_id not in trackers["vehicle_type"]:
            try:
                trackers["vehicle_type"][vehicle_id] = _vehicle_category(
                    traci.vehicle.getTypeID(vehicle_id)
                )
            except traci.TraCIException:
                trackers["vehicle_type"][vehicle_id] = "other"

    current_vehicles = set(vehicles)
    for vehicle_id, entry_time in list(trackers["entry_time"].items()):
        if vehicle_id not in current_vehicles and vehicle_id not in trackers["exit_time"]:
            trackers["exit_time"][vehicle_id] = now
            trackers["total_travel_time"] += now - entry_time
            trackers["vehicles_passed"] += 1
            vehicle_type = trackers["vehicle_type"].get(vehicle_id, "other")
            trackers["passed_by_type"][vehicle_type] += 1


def _new_trackers():
    return {
        "emission_history": [],
        "fuel_history": [],
        "waiting_history": [],
        "queue_history": [],
        "speed_history": [],
        "entry_time": {},
        "exit_time": {},
        "vehicle_type": {},
        "passed_by_type": {vehicle_type: 0 for vehicle_type in VEHICLE_TYPE_COLUMNS},
        "total_travel_time": 0.0,
        "vehicles_passed": 0,
    }


def _final_metrics(system, run_id, seed, sim_time, trackers, extra=None):
    vehicles_passed = trackers["vehicles_passed"]
    avg_travel = trackers["total_travel_time"] / max(vehicles_passed, 1)
    avg_wait = sum(trackers["waiting_history"]) / max(vehicles_passed, 1)
    total_fuel = sum(trackers["fuel_history"])
    total_co2 = sum(trackers["emission_history"])
    avg_queue = float(np.mean(trackers["queue_history"])) if trackers["queue_history"] else 0.0
    avg_speed = float(np.mean(trackers["speed_history"])) if trackers["speed_history"] else 0.0
    fairness = 0.0
    if trackers["queue_history"]:
        fairness = 1 - (
            float(np.std(trackers["queue_history"])) /
            (float(np.mean(trackers["queue_history"])) + 1e-6)
        )

    result = {
        "Run": run_id,
        "Seed": seed,
        "System": system,
        "SimulationTime": sim_time,
        "WaitingTime": round(avg_wait, 2),
        "TravelTime": round(avg_travel, 2),
        "Throughput": round(vehicles_passed / max(sim_time, 1), 3),
        "VehiclesPassed": vehicles_passed,
        "FuelConsumption": round(total_fuel, 4),
        "CO2Emission": round(total_co2, 4),
        "AvgQueue": round(avg_queue, 2),
        "AvgSpeedKmph": round(avg_speed, 2),
        "FairnessIndex": round(fairness, 3),
    }
    for vehicle_type, column_name in VEHICLE_TYPE_COLUMNS.items():
        result[column_name] = trackers["passed_by_type"].get(vehicle_type, 0)

    if extra:
        result.update(extra)
    return result


def run_fixed(run_id, sim_time, seed, binary):
    print(f"\n========== RUN {run_id} | FIXED SIGNAL | {sim_time}s ==========")
    _start_sumo(binary, seed)

    trackers = _new_trackers()
    tls_ids = traci.trafficlight.getIDList()
    green_phases = {}
    phase_cursor = {}
    last_switch = {}
    green_time = 20

    for tls_id in tls_ids:
        logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        phases = [
            index for index, phase in enumerate(logic.phases)
            if "G" in phase.state or "g" in phase.state
        ]
        green_phases[tls_id] = phases or [0]
        phase_cursor[tls_id] = 0
        last_switch[tls_id] = -green_time
        traci.trafficlight.setPhase(tls_id, green_phases[tls_id][0])

    try:
        while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < sim_time:
            traci.simulationStep()
            now = traci.simulation.getTime()

            for tls_id in tls_ids:
                if now - last_switch[tls_id] >= green_time:
                    phase_cursor[tls_id] = (phase_cursor[tls_id] + 1) % len(green_phases[tls_id])
                    traci.trafficlight.setPhase(tls_id, green_phases[tls_id][phase_cursor[tls_id]])
                    traci.trafficlight.setPhaseDuration(tls_id, green_time)
                    last_switch[tls_id] = now

            _metric_track_step(trackers)
    finally:
        traci.close()

    result = _final_metrics("Fixed", run_id, seed, sim_time, trackers)
    _print_result(result)
    return result


def _context_model_path():
    for path in CONTEXT_MODEL_PATHS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No context PPO model found in models/ppo_context")


def run_adaptive(run_id, sim_time, seed, binary):
    print(f"\n========== RUN {run_id} | ADAPTIVE CONTEXT PPO | {sim_time}s ==========")

    np.random.seed(seed)
    random.seed(seed)

    env = DummyVecEnv([lambda: TrafficEnv()])
    env = VecNormalize.load(CONTEXT_VEC_PATH, env)
    env.training = False
    env.norm_reward = False
    env.envs[0].sumo_cmd[0] = binary
    env.envs[0].sumo_cmd += ["--seed", str(seed)]

    model = PPO.load(_context_model_path(), env=env)
    env.reset()

    trackers = _new_trackers()
    tls_ids = traci.trafficlight.getIDList()
    if not tls_ids:
        raise RuntimeError("No traffic light found in SUMO network")

    tls_id = tls_ids[0]
    current_phase = 0
    last_action_time = 0

    try:
        while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < sim_time:
            traci.simulationStep()
            _metric_track_step(trackers)

            now = traci.simulation.getTime()
            if now - last_action_time < 5:
                continue

            vehicles = traci.vehicle.getIDList()
            waiting_time = sum(traci.lane.getWaitingTime(l) for l in traci.lane.getIDList())
            queue_length = sum(
                traci.lane.getLastStepHaltingNumber(l) for l in traci.lane.getIDList()
            )
            vehicle_count = len(vehicles)
            sumo_phase = traci.trafficlight.getPhase(tls_id)
            hour = (now / 3600.0) % 24

            state = np.array([
                waiting_time / 1000.0,
                queue_length / 50.0,
                vehicle_count / 100.0,
                sumo_phase / 4.0,
                env.envs[0].weather / 2.0,
                (np.sin(2 * np.pi * hour / 24) + 1) / 2,
                (np.cos(2 * np.pi * hour / 24) + 1) / 2,
                env.envs[0].zone_type / 3.0,
                env.envs[0].road_capacity,
                min(traci.simulation.getDepartedNumber() / 5.0, 1.0),
            ], dtype=np.float32)

            normalized_state = env.normalize_obs(state)
            action, _ = model.predict(normalized_state, deterministic=True)

            if int(action) == 1:
                current_phase = 2 if current_phase == 0 else 0

            traci.trafficlight.setPhase(tls_id, current_phase)
            traci.trafficlight.setPhaseDuration(tls_id, max(10, min(60, 10 + queue_length)))
            last_action_time = now
    finally:
        traci.close()

    result = _final_metrics(
        "Adaptive Context PPO",
        run_id,
        seed,
        sim_time,
        trackers,
        {
            "Weather": WEATHER_MAP[env.envs[0].weather],
            "ZoneType": ZONE_MAP[env.envs[0].zone_type],
            "RoadCapacity": round(float(env.envs[0].road_capacity), 3),
        },
    )
    _print_result(result)
    return result


def _print_result(result):
    print(f"System                  : {result['System']}")
    print(f"Vehicles Passed         : {result['VehiclesPassed']}")
    print(f"Passed Cars             : {result['PassedCars']}")
    print(f"Passed Bikes            : {result['PassedBikes']}")
    print(f"Passed Trucks           : {result['PassedTrucks']}")
    print(f"Passed Buses            : {result['PassedBuses']}")
    print(f"Passed Motorcycles      : {result['PassedMotorcycles']}")
    print(f"Passed Rickshaws        : {result['PassedRickshaws']}")
    print(f"Passed Emergency        : {result['PassedEmergencyVehicles']}")
    print(f"Passed Other Vehicles   : {result['PassedOtherVehicles']}")
    print(f"Average Waiting Time    : {result['WaitingTime']} sec/vehicle")
    print(f"Average Travel Time     : {result['TravelTime']} sec")
    print(f"Throughput              : {result['Throughput']} vehicles/sec")
    print(f"Fuel Consumption        : {result['FuelConsumption']} L")
    print(f"CO2 Emission            : {result['CO2Emission']} kg")
    print(f"Average Queue Length    : {result['AvgQueue']} vehicles")
    print(f"Average Speed           : {result['AvgSpeedKmph']} km/h")
    print(f"Fairness Index          : {result['FairnessIndex']}")
    if "Weather" in result:
        print(f"Context                 : {result['Weather']}, {result['ZoneType']}, capacity {result['RoadCapacity']}")


def save_results(results, output):
    if not results:
        return

    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fieldnames = []
    for result in results:
        for key in result:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved final comparison results to {output}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Final presentation demo for fixed vs adaptive traffic signal control."
    )
    parser.add_argument("--mode", choices=["fixed", "adaptive", "both"], default="both")
    parser.add_argument("--sim-time", type=int, default=300)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--output",
        default=os.path.join(
            "outputs",
            "results",
            f"final_comparison_{datetime.now():%Y%m%d_%H%M%S}.csv",
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    binary = "sumo-gui" if args.gui else "sumo"
    results = []
    run_id = 1

    for run_index in range(args.runs):
        seed = args.seed + run_index
        if args.mode in ("fixed", "both"):
            results.append(run_fixed(run_id, args.sim_time, seed, binary))
            run_id += 1
        if args.mode in ("adaptive", "both"):
            results.append(run_adaptive(run_id, args.sim_time, seed, binary))
            run_id += 1

    save_results(results, args.output)


if __name__ == "__main__":
    main()
