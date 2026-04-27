import traci
import os
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rl")))

emission_history = []
# ================= RL ENABLE =================
USE_RL = True

if USE_RL:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from traffic_env import TrafficEnv

    try:
        # ===== LOAD ENV + NORMALIZATION =====
        env = DummyVecEnv([lambda: TrafficEnv()])
        env = VecNormalize.load("models/vec_normalize.pkl", env)
        env.training = False
        env.norm_reward = False

        # ===== LOAD MODEL =====
        model = PPO.load("models/ppo_22000", env=env)

        print("✅ RL Model Loaded Successfully")

    except Exception as e:
        print("❌ RL Model Load Failed:", e)
        USE_RL = False


# ================= START SUMO =================
sumoCmd = ["sumo-gui", "-c", "config/simulation.sumocfg"]
traci.start(sumoCmd)

tls_ids = traci.trafficlight.getIDList()

# ================= PARAMETERS =================
alpha = 1.2
beta = 0.7
gamma = 1.0

BASE_TIME = 10
FACTOR = 0.5

MIN_GREEN = 5
MAX_GREEN = 200

SWITCH_DELAY = 2
MAX_WAIT_LIMIT = 50

last_switch = {}
prev_density = {}
rerouted = set()

# ================= RL TRACKING =================
if USE_RL:
    model.current_phase = 0
    model.last_action_time = 0

# ================= PERFORMANCE TRACK =================
start_time = traci.simulation.getTime()
vehicle_entry_time = {}
vehicle_exit_time = {}
lane_pass_count = {}
total_vehicles_passed = 0
total_time_spent = 0
vehicle_last_lane = {}


# ================= HELPER FUNCTIONS =================
def calculate_green_time(priority):
    priority = min(priority, 50)
    time_val = BASE_TIME + (priority * FACTOR)
    return max(min(time_val, MAX_GREEN), MIN_GREEN)


def predict_traffic(lane, current_density):
    prev = prev_density.get(lane, current_density)
    trend = current_density - prev
    predicted = current_density + trend
    prev_density[lane] = current_density
    return max(predicted, 0)


# ================= MAIN LOOP =================
while traci.simulation.getMinExpectedNumber() > 0:

    if traci.simulation.getTime() - start_time >= 300:
        break

    traci.simulationStep()

    # ✅ EMISSION BLOCK (ADD HERE)
    vehicles = traci.vehicle.getIDList()

    total_emission = 0
    for v in vehicles:
         total_emission += traci.vehicle.getCO2Emission(v)

    emission_history.append(total_emission)
    normalized_emission = total_emission / 1000
   # print("🌿 CO2 Emission:", total_emission , "| Normalized:", normalized_emission)


    

    # ===== Track vehicles =====
    for v in vehicles:
        if v not in vehicle_entry_time:
            vehicle_entry_time[v] = traci.simulation.getTime()
        try:
            vehicle_last_lane[v] = traci.vehicle.getLaneID(v)
        except:
            pass

    # ================= TRAFFIC LIGHT CONTROL =================
    for i, tl in enumerate(tls_ids):

        # ===== RL only on first signal =====
        if USE_RL and i == 0:
            try:
                current_time = traci.simulation.getTime()

                # ===== ACTION DELAY (stability) =====
                if current_time - model.last_action_time < 5:
                    continue

                # ===== STATE (MATCH TRAINING) =====
                waiting_time = sum(
                    traci.lane.getWaitingTime(l)
                    for l in traci.lane.getIDList()
                )
                
                # normalized_emission = total_emission / 1000

                queue_length = sum(
                    traci.lane.getLastStepHaltingNumber(l)
                    for l in traci.lane.getIDList()
                )

                vehicle_count = len(traci.vehicle.getIDList())

                current_phase = traci.trafficlight.getPhase(tl)

                state = np.array([
                    waiting_time / 1000.0,
                    queue_length / 50.0,
                    vehicle_count / 100.0,
                   
                    current_phase / 4.0
                ], dtype=np.float32)

                # ===== NORMALIZE =====
                state = env.normalize_obs(state)

                # ===== PREDICT =====
                action, _ = model.predict(state, deterministic=True)

                # ===== APPLY ACTION =====
                if action == 1:
                    model.current_phase = 2 if model.current_phase == 0 else 0

                traci.trafficlight.setPhase(tl, model.current_phase)

                model.last_action_time = current_time

                print(f"🤖 RL Action: {action} | Phase: {model.current_phase}")

                continue

            except Exception as e:
                print("❌ RL ERROR:", e)

        # ================= DEFAULT LOGIC =================
        if tl not in last_switch:
            last_switch[tl] = 0

        lanes = traci.trafficlight.getControlledLanes(tl)
        lane_data = {}

        for lane in lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)

            D = len(vehicles)
            total_wait = 0
            Q = 0

            for v in vehicles:
                total_wait += traci.vehicle.getWaitingTime(v)
                if traci.vehicle.getSpeed(v) < 0.1:
                    Q += 1

            avg_wait = total_wait / D if D > 0 else 0
            arrival = traci.lane.getLastStepVehicleNumber(lane)

            lane_data[lane] = {
                "density": D,
                "waiting": avg_wait,
                "queue": Q,
                "arrival": arrival
            }

        if not lane_data:
            continue

        lane_priority = {}

        for lane, data in lane_data.items():
            predicted_D = predict_traffic(lane, data["density"])

            W = data["waiting"]
            Q = data["queue"]
            A = data["arrival"]

            priority = (
                alpha * predicted_D +
                beta * W +
                gamma * Q +
                2.0 * A
            )

            if W > MAX_WAIT_LIMIT:
                priority += 30

            lane_priority[lane] = priority

        logic = traci.trafficlight.getAllProgramLogics(tl)[0]
        phases = logic.phases

        best_phase = 0
        best_score = -1

        for i, phase in enumerate(phases):
            score = 0
            for j, signal in enumerate(phase.state):
                if signal == 'G' and j < len(lanes):
                    score += lane_priority.get(lanes[j], 0)

            if score > best_score:
                best_score = score
                best_phase = i

        current_phase = traci.trafficlight.getPhase(tl)

        if current_phase != best_phase and (traci.simulation.getTime() - last_switch[tl] > SWITCH_DELAY):
            traci.trafficlight.setPhase(tl, best_phase)
            last_switch[tl] = traci.simulation.getTime()

        green_time = calculate_green_time(best_score)
        traci.trafficlight.setPhaseDuration(tl, green_time)

        print(f"🚦 TL: {tl} | Priority: {best_score:.2f} | Time: {green_time:.2f}")

    # ================= EXIT TRACK =================
    current_vehicles = set(traci.vehicle.getIDList())

    for v in list(vehicle_entry_time.keys()):
        if v not in current_vehicles and v not in vehicle_exit_time:
            exit_t = traci.simulation.getTime()
            vehicle_exit_time[v] = exit_t

            travel_time = exit_t - vehicle_entry_time[v]
            total_time_spent += travel_time
            total_vehicles_passed += 1

            lane = vehicle_last_lane.get(v, "unknown")
            lane_pass_count[lane] = lane_pass_count.get(lane, 0) + 1


# ================= RESULT =================
print("\n========== 🚦 SIMULATION RESULT ==========")

print(f"Total Vehicles Passed: {total_vehicles_passed}")

avg_time = (total_time_spent / total_vehicles_passed) if total_vehicles_passed > 0 else 0

# 🌿 EMISSION RESULT
total_co2 = sum(emission_history)
avg_co2 = total_co2 / (len(emission_history) + 1e-6)

max_co2 = max(emission_history)
min_co2 = min(emission_history)

print(f"🌿 Max CO2: {max_co2:.2f}")
print(f"🌿 Min CO2: {min_co2:.2f}")

print(f"\n🌿 Total CO2 Emission: {total_co2:.2f}")
print(f"🌿 Average CO2 per Step: {avg_co2:.2f}")


print("\n📍 Lane-wise Vehicle Count:")
for lane, count in lane_pass_count.items():
    print(f"{lane} → {count}")

print(f"\n⏱ Total Time Spent: {total_time_spent:.2f}")
print(f"⏱ Average Time per Vehicle: {avg_time:.2f}")


print("=========================================\n")

traci.close()