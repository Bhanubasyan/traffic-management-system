import gymnasium as gym
from gymnasium import spaces
import traci
import numpy as np
import os


class TrafficEnv(gym.Env):
    def __init__(self):
        super().__init__()

        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(2)

        self.prev_wait = 0
        self.current_phase = 0

        # Green time control
        self.green_duration = 0
        self.min_green = 2
        self.max_green = 18

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.prev_wait = 0
        self.current_phase = 0
        self.green_duration = 0

        if traci.isLoaded():
            traci.close()

        traci.start(self.sumo_cmd)

        return self._get_state(), {}

    def step(self, action):
        tls_ids = traci.trafficlight.getIDList()

        if len(tls_ids) == 0:
            return np.zeros(4, dtype=np.float32), 0, True, False, {}

        tls_id = tls_ids[0]

        # ===== GREEN TIME CONTROL =====
        self.green_duration += 1

        if self.green_duration < self.min_green:
            action = 0
        elif self.green_duration >= self.max_green:
            action = 1

        if action == 1:
            self.current_phase = 2 if self.current_phase == 0 else 0
            self.green_duration = 0

        traci.trafficlight.setPhase(tls_id, self.current_phase)

        for _ in range(5):
            traci.simulationStep()

        # ===== METRICS =====
        current_wait = self._get_waiting_time()
        queue_length = self._get_queue_length()
        arrived = traci.simulation.getArrivedNumber()

        vehicle_ids = traci.vehicle.getIDList()

        total_emission = sum(traci.vehicle.getCO2Emission(v) for v in vehicle_ids)
        normalized_emission = total_emission / 1000

        lane_queues = [traci.lane.getLastStepHaltingNumber(l) for l in traci.lane.getIDList()]
        lane_waits = [traci.lane.getWaitingTime(l) for l in traci.lane.getIDList()]

        # ===== REWARD =====
        reward = 0

        # Global traffic pressure
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

        # Starvation
        if len(lane_waits) > 0 and max(lane_waits) > 200:
            reward -= 20

        # Emission
        reward -= 0.1 * normalized_emission

        # Green wave (flow)
        moving = sum(1 for v in vehicle_ids if traci.vehicle.getSpeed(v) > 5)
        reward += 0.5 * moving

        # Stop penalty
        stopped = sum(1 for v in vehicle_ids if traci.vehicle.getSpeed(v) < 0.1)
        reward -= 0.2 * stopped

        # Delay reduction
        reward += (self.prev_wait - current_wait)

        # Idle green
        if queue_length == 0 and action == 0:
            reward -= 2

        # Max wait penalty
        if len(lane_waits) > 0 and max(lane_waits) > 300:
            reward -= 30

        # Normalize
        reward = reward / 100.0

        # update
        self.prev_wait = current_wait
        self.step_count += 1
        done = self.step_count >= self.max_steps

        return self._get_state(), reward, done, False, {}

    def _get_state(self):
        tls_ids = traci.trafficlight.getIDList()

        if len(tls_ids) == 0:
            return np.zeros(4, dtype=np.float32)

        tls_id = tls_ids[0]
        phase = traci.trafficlight.getPhase(tls_id)

        waiting = self._get_waiting_time() / 1000.0
        queue = self._get_queue_length() / 50.0
        vehicles = len(traci.vehicle.getIDList()) / 100.0
        phase = phase / 4.0

        return np.array([waiting, queue, vehicles, phase], dtype=np.float32)

    def _get_waiting_time(self):
        return sum(traci.lane.getWaitingTime(lane) for lane in traci.lane.getIDList())

    def _get_queue_length(self):
        return sum(traci.lane.getLastStepHaltingNumber(lane) for lane in traci.lane.getIDList())

    def close(self):
        if traci.isLoaded():
            traci.close()