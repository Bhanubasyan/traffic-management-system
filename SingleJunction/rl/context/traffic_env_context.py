import gymnasium as gym
from gymnasium import spaces
import traci
import numpy as np
import os


OBSERVATION_SIZE = 15
HEAVY_VEHICLE_KEYWORDS = (
    "bus",
    "truck",
    "lorry",
    "tanker",
    "van",
)
EMERGENCY_VEHICLE_KEYWORDS = (
    "ambulance",
    "emergency",
)


class TrafficEnv(gym.Env):
    def __init__(self):
        super().__init__()

        base_path = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        )
        config_path = os.path.join(base_path, "config", "simulation.sumocfg")

        self.sumo_cmd = [
            "sumo",
            "-c", config_path,
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1",
            "--start", "true",
            "--quit-on-end", "true"
        ]

        self.max_steps = 1000
        self.step_count = 0

        # =========================
        # CONTEXT-AWARE OBSERVATION SPACE
        # =========================
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(2)

        self.prev_wait = 0
        self.current_phase = 0

        # =========================
        # GREEN TIME CONTROL
        # =========================
        self.green_duration = 0
        self.min_green = 2
        self.max_green = 18

        # =========================
        # CONTEXT VARIABLES
        # =========================

        # Weather:
        # 0 = Clear
        # 1 = Rain
        # 2 = Fog
        # 3 = Heavy rain
        self.weather = 0

        # Zone Types:
        # 0 = Residential
        # 1 = Commercial
        # 2 = Industrial
        # 3 = School
        self.zone_type = 1

        # normalized road capacity
        self.road_capacity = 0.8

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.prev_wait = 0
        self.current_phase = 0
        self.green_duration = 0

        # =========================
        # RANDOM CONTEXT GENERATION
        # =========================

        self.weather = np.random.choice([0, 1, 2, 3])

        self.zone_type = np.random.choice([0, 1, 2, 3])

        self.road_capacity = np.random.uniform(0.5, 1.0)

        if traci.isLoaded():
            traci.close()

        traci.start(self.sumo_cmd)

        return self._get_state(), {}

    def step(self, action):

        tls_ids = traci.trafficlight.getIDList()

        if len(tls_ids) == 0:
            return np.zeros(OBSERVATION_SIZE, dtype=np.float32), 0, True, False, {}

        tls_id = tls_ids[0]

        # =========================
        # GREEN TIME CONTROL
        # =========================

        self.green_duration += 1

        if self.green_duration < self.min_green:
            action = 0

        elif self.green_duration >= self.max_green:
            action = 1

        # =========================
        # CONTEXT-AWARE DECISION LOGIC
        # =========================

        queue_length = self._get_queue_length()

        traffic_context = self._get_traffic_context()

        # Bad weather, heavy vehicles, emergency vehicles, and highly unbalanced
        # queues all need more stable green windows.
        if (
            (self.weather in (1, 3) and self.zone_type == 1 and queue_length > 10) or
            traffic_context["heavy_vehicle_ratio"] > 0.35 or
            traffic_context["emergency_vehicle_present"] > 0 or
            traffic_context["queue_imbalance"] > 0.5
        ):
            self.max_green = 25

        else:
            self.max_green = 18

        # switch phase
        if action == 1:
            self.current_phase = 2 if self.current_phase == 0 else 0
            self.green_duration = 0

        traci.trafficlight.setPhase(tls_id, self.current_phase)

        for _ in range(5):
            traci.simulationStep()

        # =========================
        # METRICS
        # =========================

        current_wait = self._get_waiting_time()

        queue_length = self._get_queue_length()

        arrived = traci.simulation.getArrivedNumber()

        vehicle_ids = traci.vehicle.getIDList()

        total_emission = sum(
            traci.vehicle.getCO2Emission(v)
            for v in vehicle_ids
        )

        normalized_emission = total_emission / 1000

        lane_queues = [
            traci.lane.getLastStepHaltingNumber(l)
            for l in traci.lane.getIDList()
        ]

        lane_waits = [
            traci.lane.getWaitingTime(l)
            for l in traci.lane.getIDList()
        ]

        traffic_context = self._get_traffic_context()

        # =========================
        # REWARD FUNCTION
        # =========================

        reward = 0

        # Queue pressure
        reward -= 1.5 * queue_length

        # Waiting time
        reward -= 0.05 * current_wait

        # Throughput
        reward += 5 * arrived

        # Fairness
        if len(lane_queues) > 0:
            imbalance = max(lane_queues) - min(lane_queues)
            reward -= 0.5 * imbalance

        # Switching penalty
        if action == 1:
            reward -= 2

        # Starvation penalty
        if len(lane_waits) > 0 and max(lane_waits) > 200:
            reward -= 20

        # Emission penalty
        reward -= 0.1 * normalized_emission

        # Green wave reward
        moving = sum(
            1 for v in vehicle_ids
            if traci.vehicle.getSpeed(v) > 5
        )

        reward += 0.5 * moving

        # Stop penalty
        stopped = sum(
            1 for v in vehicle_ids
            if traci.vehicle.getSpeed(v) < 0.1
        )

        reward -= 0.2 * stopped

        # Rich context rewards
        reward -= 8 * traffic_context["queue_imbalance"]
        reward -= 0.03 * traffic_context["max_lane_wait"]

        if traffic_context["is_peak_hour"] > 0:
            reward += 0.2 * moving

        if traffic_context["heavy_vehicle_ratio"] > 0.25:
            reward -= 0.5 * stopped

        if traffic_context["emergency_vehicle_present"] > 0:
            reward -= 0.1 * traffic_context["emergency_waiting_time"]
            if action == 1:
                reward -= 1

        # Delay reduction reward
        reward += (self.prev_wait - current_wait)

        # Idle green penalty
        if queue_length == 0 and action == 0:
            reward -= 2

        # Max wait penalty
        if len(lane_waits) > 0 and max(lane_waits) > 300:
            reward -= 30

        # Normalize reward
        reward = reward / 100.0

        # update
        self.prev_wait = current_wait
        self.step_count += 1

        done = self.step_count >= self.max_steps

        return self._get_state(), reward, done, False, {}

    def _get_state(self):

        tls_ids = traci.trafficlight.getIDList()

        if len(tls_ids) == 0:
            return np.zeros(OBSERVATION_SIZE, dtype=np.float32)

        tls_id = tls_ids[0]

        phase = traci.trafficlight.getPhase(tls_id)

        # =========================
        # BASIC TRAFFIC FEATURES
        # =========================

        waiting = self._get_waiting_time() / 1000.0

        queue = self._get_queue_length() / 50.0

        vehicles = len(traci.vehicle.getIDList()) / 100.0

        phase = phase / 4.0

        # =========================
        # ARRIVAL RATE
        # =========================

        arrival_rate = traci.simulation.getDepartedNumber() / 10.0
        arrival_rate = min(arrival_rate, 1.0)

        # =========================
        # TIME FEATURES
        # =========================

        sim_time = traci.simulation.getTime()

        hour = (sim_time / 3600.0) % 24

        hour_sin = (
            np.sin(2 * np.pi * hour / 24) + 1
        ) / 2

        hour_cos = (
            np.cos(2 * np.pi * hour / 24) + 1
        ) / 2

        # =========================
        # FINAL STATE VECTOR
        # =========================

        traffic_context = self._get_traffic_context()

        state = np.array([
            waiting,
            queue,
            vehicles,
            phase,
            traffic_context["weather_severity"],
            hour_sin,
            hour_cos,
            self.zone_type / 3.0,
            self.road_capacity,
            arrival_rate,
            traffic_context["heavy_vehicle_ratio"],
            traffic_context["emergency_vehicle_present"],
            traffic_context["queue_imbalance"],
            traffic_context["max_lane_wait"],
            traffic_context["is_peak_hour"],
        ], dtype=np.float32)

        return state

    def _get_traffic_context(self):
        lanes = traci.lane.getIDList()
        vehicles = traci.vehicle.getIDList()

        heavy_count = 0
        emergency_count = 0
        emergency_waiting_time = 0.0

        for vehicle_id in vehicles:
            try:
                type_id = traci.vehicle.getTypeID(vehicle_id).lower()
            except traci.TraCIException:
                type_id = ""

            if any(keyword in type_id for keyword in HEAVY_VEHICLE_KEYWORDS):
                heavy_count += 1

            if any(keyword in type_id for keyword in EMERGENCY_VEHICLE_KEYWORDS):
                emergency_count += 1
                try:
                    emergency_waiting_time += traci.vehicle.getWaitingTime(vehicle_id)
                except traci.TraCIException:
                    pass

        lane_queues = [
            traci.lane.getLastStepHaltingNumber(lane)
            for lane in lanes
        ]
        lane_waits = [
            traci.lane.getWaitingTime(lane)
            for lane in lanes
        ]

        max_queue = max(lane_queues) if lane_queues else 0
        min_queue = min(lane_queues) if lane_queues else 0
        max_lane_wait = max(lane_waits) if lane_waits else 0

        sim_time = traci.simulation.getTime()
        hour = (sim_time / 3600.0) % 24
        is_peak_hour = 1.0 if 7 <= hour <= 10 or 17 <= hour <= 20 else 0.0

        return {
            "weather_severity": self.weather / 3.0,
            "heavy_vehicle_ratio": min(heavy_count / max(len(vehicles), 1), 1.0),
            "emergency_vehicle_present": 1.0 if emergency_count > 0 else 0.0,
            "emergency_waiting_time": min(emergency_waiting_time / 300.0, 1.0),
            "queue_imbalance": min((max_queue - min_queue) / 50.0, 1.0),
            "max_lane_wait": min(max_lane_wait / 300.0, 1.0),
            "is_peak_hour": is_peak_hour,
        }

    def _get_waiting_time(self):
        return sum(
            traci.lane.getWaitingTime(lane)
            for lane in traci.lane.getIDList()
        )

    def _get_queue_length(self):
        return sum(
            traci.lane.getLastStepHaltingNumber(lane)
            for lane in traci.lane.getIDList()
        )

    def close(self):
        if traci.isLoaded():
            traci.close()
