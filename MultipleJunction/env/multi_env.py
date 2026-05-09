import os
import gymnasium as gym
from gymnasium import spaces
import traci
import numpy as np

class MultiTrafficEnv(gym.Env):
    def __init__(self):
        super().__init__()

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.sumo_cmd = [
            "sumo",
            "-c",
            os.path.join(base_dir, "config", "simulation.sumocfg"),
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--log", "log.txt"
        ]

        self.tls_ids = []
        self.num_agents = 0

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(2)

        self.last_actions = []

    # ================= RESET =================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if traci.isLoaded():
            traci.close()

        traci.start(self.sumo_cmd)

        self.tls_ids = traci.trafficlight.getIDList()
        self.num_agents = len(self.tls_ids)

        self.last_actions = [0] * self.num_agents

        return self._get_states(), {}

    # ================= STEP =================
    def step(self, actions):

        switch_count = 0

        for i, tl in enumerate(self.tls_ids):
            action = actions[i]

            if action == 1:
                phase = traci.trafficlight.getPhase(tl)
                new_phase = 2 if phase == 0 else 0
                traci.trafficlight.setPhase(tl, new_phase)
                switch_count += 1

            self.last_actions[i] = action

        for _ in range(2):
            traci.simulationStep()

        states = self._get_states()

        reward_val = self._get_reward()
        if reward_val is None:
            reward_val = 0

        reward = reward_val - (2 * switch_count)

        done = traci.simulation.getMinExpectedNumber() == 0

        return states, float(reward), done, False, {}

    # ================= STATE =================
    def _get_states(self):
        states = []

        for tl in self.tls_ids:
            lanes = traci.trafficlight.getControlledLanes(tl)

            queue = 0
            wait = 0
            speed = 0
            count = 0

            for lane in lanes:
                queue += traci.lane.getLastStepHaltingNumber(lane)
                wait += traci.lane.getWaitingTime(lane)

                vehs = traci.lane.getLastStepVehicleIDs(lane)
                for v in vehs:
                    speed += traci.vehicle.getSpeed(v)
                    count += 1

            avg_speed = speed / (count + 1)

            state = np.array([
                min(queue / 50, 1),
                min(wait / 1000, 1),
                min(avg_speed / 15, 1),
                min(count / 50, 1)
            ], dtype=np.float32)

            states.append(state)

        return np.array(states)

    # ================= REWARD =================
    def _get_reward(self):
        total_wait = 0
        total_queue = 0
        total_arrived = traci.simulation.getArrivedNumber()

        lane_queues = []
        lane_waits = []

        vehicle_ids = traci.vehicle.getIDList()

        total_emission = 0
        moving = 0
        stopped = 0
        total_speed = 0

        for lane in traci.lane.getIDList():
            wait = traci.lane.getWaitingTime(lane)
            queue = traci.lane.getLastStepHaltingNumber(lane)

            total_wait += wait
            total_queue += queue

            lane_queues.append(queue)
            lane_waits.append(wait)

        for v in vehicle_ids:
            speed = traci.vehicle.getSpeed(v)

            total_speed += speed
            total_emission += traci.vehicle.getCO2Emission(v)

            if speed > 5:
                moving += 1
            if speed < 0.1:
                stopped += 1

        avg_speed = total_speed / (len(vehicle_ids) + 1)
        normalized_emission = total_emission / 1000

        reward = 0
        reward -= 0.15 * total_queue
        reward -= 0.03 * (total_wait / (len(vehicle_ids) + 1))
        reward += 3 * total_arrived

        if lane_queues:
            imbalance = max(lane_queues) - min(lane_queues)
            reward -= 0.7 * imbalance

        
        if lane_waits:  #and max(lane_waits) > 200:
            reward -= 0.8 * max(lane_waits)
        
        reward -= 0.02 * normalized_emission
        reward += 0.2 * moving
        reward -= 0.1 * stopped 
        reward -= 0.4 * max(lane_queues)
        reward += (getattr(self, "prev_wait", total_wait) - total_wait)

        self.prev_reward = getattr(self, "prev_reward", 0)
        reward = 0.3 * self.prev_reward + 0.7 * reward
        self.prev_reward = reward

        reward = np.clip(reward, -100, 100)
        self.prev_wait = total_wait

        return float(reward / 100.0)

    # ================= CLOSE =================
    def close(self):
        if traci.isLoaded():
            traci.close()