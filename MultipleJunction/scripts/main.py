import traci
import os
import numpy as np
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ================= PATH FIX =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))

from env.multi_env import MultiTrafficEnv   # ✅ correct import

class FlattenMultiAgentEnv(gym.Env):
    def __init__(self, env):
        super().__init__()

        self.env = env

        states, _ = self.env.reset()
        self.num_agents = len(states)

        flat_state = states.flatten()

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=flat_state.shape, dtype=np.float32
        )

        self.action_space = gym.spaces.MultiDiscrete([2] * self.num_agents)

    def reset(self, seed=None, options=None):
        states, _ = self.env.reset()
        return states.flatten(), {}

    def step(self, action):
        states, reward, done, _, _ = self.env.step(action)
        return states.flatten(), reward, done, False, {}

    def close(self):
        self.env.close()

# ================= LOAD ENV =================
env = DummyVecEnv([lambda: FlattenMultiAgentEnv(MultiTrafficEnv())])

env = VecNormalize.load(
    os.path.join(BASE_DIR, "../models/vec_normalize.pkl"),
    env
)

env.training = False
env.norm_reward = False
# ================= LOAD MODEL =================
model = PPO.load(
    os.path.join(BASE_DIR, "../models/ppo_10000.zip"),
    env=env
)

# ================= START SUMO =================
if traci.isLoaded():
    traci.close()


sumoCmd = ["sumo-gui", "-c", os.path.join(BASE_DIR, "../config/simulation.sumocfg")]
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

# ================= HELPER =================
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


# ================= STATE FOR RL =================
def get_state():
    states = []
    for tl in tls_ids:
        waiting = sum(traci.lane.getWaitingTime(l) for l in traci.lane.getIDList()) / 1000.0
        queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in traci.lane.getIDList()) / 50.0
        vehicles = len(traci.vehicle.getIDList()) / 100.0
        phase = traci.trafficlight.getPhase(tl) / 4.0

        states.append([waiting, queue, vehicles, phase])

    return np.array(states).flatten()

# ================= METRICS =================
SIM_TIME = 300   # change: 120 / 300 / 1200

# ❌ OLD GLOBAL VARIABLES (COMMENTED)
# total_wait = 0
# total_vehicles = 0
# total_co2 = 0
# total_speed = 0
# speed_count = 0


# ✅ ADD: JUNCTION STATS
junction_stats = {
    tl: {
        "passed": 0,
        "waiting": 0.0,
        "speed": 0.0,
        "co2": 0.0,
        "count": 0
    }
    for tl in tls_ids
}
vehicle_prev_lane = {}


# ================= MAIN LOOP =================
while traci.simulation.getTime() < SIM_TIME:

    traci.simulationStep()

    # ================= RL CONTROL =================
    state = get_state()
    action, _ = model.predict(state, deterministic=True)

    for i, tl in enumerate(tls_ids):
        if action[i] == 1:
            phase = traci.trafficlight.getPhase(tl)
            new_phase = 2 if phase == 0 else 0
            traci.trafficlight.setPhase(tl, new_phase)

    # ================= METRICS COLLECTION =================
    vehicle_ids = traci.vehicle.getIDList()

    # ✅ CORRECT JUNCTION-WISE MAPPING
    for tl in tls_ids:
        lanes = traci.trafficlight.getControlledLanes(tl)

        for v in vehicle_ids:
            lane = traci.vehicle.getLaneID(v)

            if lane in lanes:

                junction_stats[tl]["waiting"] += traci.vehicle.getWaitingTime(v)
                junction_stats[tl]["co2"] += traci.vehicle.getCO2Emission(v)

                speed = traci.vehicle.getSpeed(v)
                junction_stats[tl]["speed"] += speed
                junction_stats[tl]["count"] += 1
    #=================
   

    # ===== PASSED VEHICLES =====
   # ===== PASSED VEHICLES (CROSS DETECTION) =====
    for v in traci.vehicle.getIDList():

        current_lane = traci.vehicle.getLaneID(v)
        prev_lane = vehicle_prev_lane.get(v, None)

        for tl in tls_ids:
            lanes = traci.trafficlight.getControlledLanes(tl)

            # vehicle ne junction cross kiya
            if prev_lane in lanes and current_lane not in lanes:
                junction_stats[tl]["passed"] += 1

        vehicle_prev_lane[v] = current_lane


# ===== AFTER WHILE LOOP =====

print("\n=========== FINAL RESULT ===========")
print(f"Simulation Time: {SIM_TIME} sec\n")

for tl, data in junction_stats.items():

    count = max(data["count"], 1)

    avg_speed = data["speed"] / count
    avg_travel = data["waiting"] / count

    print(f"Junction {tl}:")
    print(f"  Simulation Time: {SIM_TIME} sec")
    print(f"  Vehicles Passed: {data['passed']}")
    print(f"  Total Waiting Time: {data['waiting']:.2f}")
    print(f"  Average Travel Time: {avg_travel:.2f}")
    print(f"  Average Speed: {avg_speed:.2f}")
    print(f"  Total CO2 Emission: {data['co2']:.2f}\n")

    # =====================================================
    # ❌ OLD HEURISTIC CONTROL (COMMENTED)
    # =====================================================

    """
    # 🚦 MULTI-JUNCTION LOOP
    for tl in tls_ids:

        if tl not in last_switch:
            last_switch[tl] = 0

        lanes = traci.trafficlight.getControlledLanes(tl)
        lane_data = {}

        # ===== COLLECT DATA =====
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

        # ===== PRIORITY =====
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

        # ===== PHASE SELECTION =====
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

        # ===== GREEN TIME =====
        green_time = calculate_green_time(best_score)
        traci.trafficlight.setPhaseDuration(tl, green_time)

        print(f"🚦 {tl} | Score: {best_score:.2f} | Time: {green_time:.2f}")
    """

# ================= CLOSE =================
traci.close()