"""Evaluate MAPPO and print/save presentation-ready multi-junction metrics."""

import argparse
import csv
import os
import sys
from datetime import datetime

import numpy as np
import traci

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms.mappo import MAPPO
from environment.sumo_mappo_env import SUMOMultiAgentEnv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def metric_table(title, rows):
    print(f"\n{title}\n" + "-" * 72)
    for label, value, unit in rows:
        print(f"{label:<32} {value:>14}{(' ' + unit) if unit else ''}")


def print_result(result):
    print("\n" + "=" * 72)
    print(f"MULTI-JUNCTION TRAFFIC CONTROL RESULT | Run {result['Run_ID']} | Duration: {result['Simulation_Time_sec']:.0f}s")
    print(f"Model: {result['Model']}")
    print("=" * 72)
    metric_table("Traffic Performance", [
        ("Vehicles passed", result["Vehicles_Passed"], "vehicles"),
        ("Throughput", f"{result['Throughput_vehicles_per_sec']:.3f}", "veh/sec"),
        ("Avg waiting time", f"{result['Avg_Waiting_Time_sec_per_vehicle']:.2f}", "sec/vehicle"),
        ("Avg travel time", f"{result['Avg_Travel_Time_sec']:.2f}", "sec"),
        ("Avg queue length", f"{result['Avg_Queue_Length_vehicles']:.2f}", "vehicles"),
        ("Avg speed", f"{result['Avg_Speed_kmh']:.2f}", "km/h"),
        ("Fairness index", f"{result['Fairness_Index']:.3f}", "score"),
        ("Evaluation reward", f"{result['Evaluation_Reward']:.3f}", "score"),
    ])
    metric_table("Environment Impact", [
        ("Total CO2", f"{result['Total_CO2_kg']:.3f}", "kg"),
        ("Avg CO2", f"{result['Avg_CO2_kg_per_step']:.4f}", "kg/step"),
        ("Total fuel", f"{result['Total_Fuel_L']:.3f}", "L"),
        ("Avg fuel", f"{result['Avg_Fuel_L_per_step']:.4f}", "L/step"),
    ])
    print("\nPer-Junction Performance\n" + "-" * 72)
    print(f"{'Agent':<10} {'TLS':<8} {'Avg queue':>12} {'Avg wait':>14} {'Avg speed':>14}")
    for agent in sorted(result["Junction_Metrics"]):
        data = result["Junction_Metrics"][agent]
        print(f"{agent:<10} {data['tls']:<8} {data['queue']:>12.2f} {data['wait']:>14.2f} {data['speed']:>13.2f}")


def evaluate(run_id, model_path, controller="mappo", green_time=30, gui=False, seed=None):
    agent = None
    if controller == "mappo":
        agent = MAPPO(obs_dim=10)
        agent.load(model_path)
    env = SUMOMultiAgentEnv(gui=gui, sumo_seed=seed)
    obs, _ = env.reset()
    queues, waits, speeds, co2, fuel = [], [], [], [], []
    junction_history = {agent_id: {"queue": [], "wait": [], "speed": []} for agent_id in env.possible_agents}
    entries, travel_times, vehicle_waiting, completed_waits = {}, [], {}, []
    total_reward, steps, done = 0.0, 0, False

    while not done:
        if controller == "mappo":
            actions, _ = agent.select_action(obs, deterministic=True)
        else:
            # Change all signals every fixed green interval; env still protects min/max green time.
            actions = {agent_id: int(env.green_time[agent_id] >= green_time) for agent_id in env.agents}
        obs, rewards, terminated, truncated, _ = env.step(actions)
        total_reward += sum(rewards.values())
        steps += 1
        vehicles = traci.vehicle.getIDList()
        for vehicle in vehicles:
            entries.setdefault(vehicle, traci.simulation.getTime())
            vehicle_waiting[vehicle] = traci.vehicle.getAccumulatedWaitingTime(vehicle)
        for vehicle in traci.simulation.getArrivedIDList():
            if vehicle in entries:
                travel_times.append(traci.simulation.getTime() - entries.pop(vehicle))
                completed_waits.append(vehicle_waiting.pop(vehicle, 0.0))
        queues.append(sum(traci.lane.getLastStepHaltingNumber(lane) for lane in traci.lane.getIDList()))
        waits.append(sum(traci.lane.getWaitingTime(lane) for lane in traci.lane.getIDList()))
        if vehicles:
            speeds.append(np.mean([traci.vehicle.getSpeed(vehicle) for vehicle in vehicles]) * 3.6)
            co2.append(sum(traci.vehicle.getCO2Emission(vehicle) for vehicle in vehicles) / 1_000_000)
            fuel.append(sum(traci.vehicle.getFuelConsumption(vehicle) for vehicle in vehicles) / 1_000_000)
        for agent_id, tls in env.agent_to_tls.items():
            values = env._metrics(tls)
            junction_history[agent_id]["queue"].append(values["queue"])
            junction_history[agent_id]["wait"].append(values["wait"])
            junction_history[agent_id]["speed"].append(values["speed"] * 3.6)
        done = all(terminated.values()) or all(truncated.values())

    simulation_time = traci.simulation.getTime()
    passed = len(travel_times)
    result = {
        "Run_ID": run_id, "Seed": seed if seed is not None else "default", "Model": os.path.basename(model_path) if controller == "mappo" else f"Fixed-time ({green_time}s)", "Simulation_Time_sec": simulation_time,
        "Vehicles_Passed": passed, "Evaluation_Reward": total_reward,
        "Throughput_vehicles_per_sec": passed / max(simulation_time, 1),
        "Avg_Waiting_Time_sec_per_vehicle": float(np.mean(completed_waits)) if completed_waits else 0.0,
        "Avg_Travel_Time_sec": float(np.mean(travel_times)) if travel_times else 0.0,
        "Avg_Queue_Length_vehicles": float(np.mean(queues)) if queues else 0.0,
        "Avg_Speed_kmh": float(np.mean(speeds)) if speeds else 0.0,
        "Fairness_Index": 1 - float(np.std(queues)) / (float(np.mean(queues)) + 1e-6) if queues and np.mean(queues) else 0.0,
        "Total_CO2_kg": float(np.sum(co2)), "Avg_CO2_kg_per_step": float(np.mean(co2)) if co2 else 0.0,
        "Total_Fuel_L": float(np.sum(fuel)), "Avg_Fuel_L_per_step": float(np.mean(fuel)) if fuel else 0.0,
        "Junction_Metrics": {agent_id: {"tls": env.agent_to_tls[agent_id], **{key: float(np.mean(values)) if values else 0.0 for key, values in history.items()}} for agent_id, history in junction_history.items()},
    }
    env.close()
    return result


parser = argparse.ArgumentParser(description="Evaluate MAPPO with detailed traffic metrics.")
parser.add_argument("--model", default=os.path.join(BASE_DIR, "models", "mappo_v2.pt"))
parser.add_argument("--controller", choices=["mappo", "fixed"], default="mappo", help="Evaluate the learned model or a fixed-time baseline.")
parser.add_argument("--green-time", type=int, default=30, help="Fixed green duration in seconds (only for --controller fixed).")
parser.add_argument("--runs", type=int, default=1, help="Number of evaluation runs.")
parser.add_argument("--seed", type=int, default=None, help="Base SUMO seed; each run uses seed, seed+1, ...")
parser.add_argument("--gui", action="store_true", help="Show SUMO-GUI (use --runs 1).")
args = parser.parse_args()
if args.controller == "mappo" and not os.path.exists(args.model):
    raise FileNotFoundError(f"Model not found: {args.model}. Run scripts/train.py first.")

results = [evaluate(run_id, args.model, args.controller, args.green_time, args.gui, None if args.seed is None else args.seed + run_id - 1) for run_id in range(1, args.runs + 1)]
for result in results:
    print_result(result)

output_dir = os.path.join(BASE_DIR, "outputs", "results")
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, f"{args.controller}_results_{datetime.now():%Y%m%d_%H%M%S}.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as handle:
    fields = [key for key in results[0] if key != "Junction_Metrics"]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows([{key: value for key, value in result.items() if key in fields} for result in results])
print(f"\nAll runs completed. CSV saved to: {csv_path}")
