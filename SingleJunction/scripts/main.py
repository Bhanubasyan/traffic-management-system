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
vehicle_waiting = {}
fuel_consumption = 0

SIM_TIME = 1200
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

    if traci.simulation.getTime() - start_time >= SIM_TIME :
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

        # entry time
        if v not in vehicle_entry_time:
            vehicle_entry_time[v] = traci.simulation.getTime()

        # accumulated waiting time
        waiting = traci.vehicle.getAccumulatedWaitingTime(v)

        # fuel consumption
        try:
            fuel_consumption += traci.vehicle.getFuelConsumption(v)
        except:
            pass

        vehicle_waiting[v] = max(
            vehicle_waiting.get(v, 0),
            waiting
        )

        # last lane
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
                phase_changed = False
                if action == 1:
                   new_phase = 2 if model.current_phase == 0 else 0

                   if new_phase != model.current_phase:

                       model.current_phase = new_phase

                       phase_changed = True

                current_sumo_phase = traci.trafficlight.getPhase(tl)

                if phase_changed and current_sumo_phase != model.current_phase:

                    traci.trafficlight.setPhase(tl, model.current_phase)
                    green_time = 10 + queue_length

                    green_time = max(10, min(green_time, 60))

                    traci.trafficlight.setPhaseDuration(tl, green_time)
                    signal_state = traci.trafficlight.getRedYellowGreenState(tl)

                    remaining_time = green_time
                    # ===== ACTIVE DIRECTION =====
                    if model.current_phase == 0:
                        active_direction = "North-South"

                    else:
                        active_direction = "East-West"
                        
                    if remaining_time >= 22:
                        print("\n==============================")
                    
                        print(f"🚦 Junction ID      : {tl}")

                        print(f"🟢 Current Phase    : {model.current_phase}")
                        
                        print(f"Active Direction    : {active_direction}")
                        print(f"🚥 Signal State     : {signal_state}")

                        print(f"⏳ Green Time       : {remaining_time:.1f} sec")

                        print("==============================")

                model.last_action_time = current_time

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

    arrived_vehicles = traci.simulation.getArrivedIDList()

    for v in arrived_vehicles:

        if v not in vehicle_exit_time:

            exit_t = traci.simulation.getTime()

            vehicle_exit_time[v] = exit_t

            if v in vehicle_entry_time:

                travel_time = exit_t - vehicle_entry_time[v]

                total_time_spent += travel_time

            total_vehicles_passed += 1

            lane = vehicle_last_lane.get(v, "unknown")

            lane_pass_count[lane] = (
                lane_pass_count.get(lane, 0) + 1
            )



# =========================================================
#                   FINAL RESULT
# =========================================================



# Waiting Time
total_wait = sum(vehicle_waiting.values())

avg_wait = (
    total_wait / total_vehicles_passed
    if total_vehicles_passed > 0 else 0
)

# Travel Time
avg_time = (
    total_time_spent / total_vehicles_passed
    if total_vehicles_passed > 0 else 0
)

# Fuel in Liters
fuel_liters = fuel_consumption / 1000000

# CO2 in KG
total_co2 = sum(emission_history) / 1000000

avg_co2 = (
    total_co2 / len(emission_history)
    if len(emission_history) > 0 else 0
)

# Throughput
throughput = total_vehicles_passed / SIM_TIME


# =========================================================
#                   CLEAN OUTPUT
# =========================================================

print("\n========== ADAPTIVE SIGNAL RESULTS ==========\n")

print(f"Simulation Time           : {SIM_TIME} sec")

print(f"Vehicles Passed           : {total_vehicles_passed}")

print(f"Average Waiting Time      : {avg_wait:.2f} sec")

print(f"Average Travel Time       : {avg_time:.2f} sec")

print(f"Fuel Consumption          : {fuel_liters:.3f} L")

print(f"Total CO2 Emission        : {total_co2:.2f} kg")

print(f"Average CO2 per Seconds   : {avg_co2:.4f} kg")

print(f"Traffic Throughput        : {throughput:.2f} veh/sec")

print("\n=============================================")


traci.close()
