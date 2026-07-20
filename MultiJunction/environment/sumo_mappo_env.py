"""PettingZoo parallel environment for coordinated SUMO traffic signals."""

import os

import traci
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


class SUMOMultiAgentEnv(ParallelEnv):
    metadata = {"name": "sumo_mappo_v1"}

    def __init__(self, gui=False, max_steps=500, decision_interval=5, min_green=10, max_green=45, sumo_seed=None):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.sumo_cmd = [
            "sumo-gui" if gui else "sumo", "-c",
            os.path.join(self.base_dir, "network", "simulation.sumocfg"),
            "--no-step-log", "true", "--no-warnings", "true", "--time-to-teleport", "-1",
        ]
        self.agent_to_tls = {"agent_0": "1", "agent_1": "12", "agent_2": "2", "agent_3": "8"}
        self.possible_agents = list(self.agent_to_tls)
        self.agents = self.possible_agents[:]
        self.observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {agent: spaces.Discrete(2) for agent in self.possible_agents}
        self.max_steps = max_steps
        self.decision_interval = decision_interval
        self.min_green = min_green
        self.max_green = max_green
        self.sumo_seed = sumo_seed
        self.steps = 0
        self.previous_wait = {}
        self.green_time = {}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        if traci.isLoaded():
            traci.close()
        command = self.sumo_cmd[:]
        if self.sumo_seed is not None:
            command.extend(["--seed", str(self.sumo_seed)])
        traci.start(command)
        self.agents = self.possible_agents[:]
        self.steps = 0
        self.previous_wait = {agent: 0.0 for agent in self.agents}
        self.green_time = {agent: 0 for agent in self.agents}
        return self._get_observations(), {}

    def step(self, actions):
        switched = {agent: False for agent in self.agents}
        for agent in self.agents:
            self.green_time[agent] += self.decision_interval
            action = actions.get(agent, 0)
            # Protect realistic signal operation: no rapid switching and no endless green.
            if self.green_time[agent] < self.min_green:
                action = 0
            elif self.green_time[agent] >= self.max_green:
                action = 1
            if action == 1:
                tls = self.agent_to_tls[agent]
                current_phase = traci.trafficlight.getPhase(tls)
                traci.trafficlight.setPhase(tls, 2 if current_phase == 0 else 0)
                self.green_time[agent] = 0
                switched[agent] = True

        for _ in range(self.decision_interval):
            traci.simulationStep()
        self.steps += 1

        observations = self._get_observations()
        simulation_finished = traci.simulation.getMinExpectedNumber() == 0
        time_limit_reached = self.steps >= self.max_steps
        rewards = {agent: self._get_agent_reward(agent, switched[agent]) for agent in self.agents}
        terminations = {agent: simulation_finished for agent in self.agents}
        truncations = {agent: time_limit_reached for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        if simulation_finished or time_limit_reached:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def _metrics(self, tls):
        lanes = set(traci.trafficlight.getControlledLanes(tls))
        queues = [traci.lane.getLastStepHaltingNumber(lane) for lane in lanes]
        waits = [traci.lane.getWaitingTime(lane) for lane in lanes]
        vehicle_ids = [vehicle for lane in lanes for vehicle in traci.lane.getLastStepVehicleIDs(lane)]
        speeds = [traci.vehicle.getSpeed(vehicle) for vehicle in vehicle_ids]
        return {
            "queue": sum(queues), "wait": sum(waits), "vehicles": len(vehicle_ids),
            "speed": float(np.mean(speeds)) if speeds else 0.0,
            "imbalance": max(queues, default=0) - min(queues, default=0),
            "max_wait": max(waits, default=0),
        }

    def _get_observations(self):
        departed = traci.simulation.getDepartedNumber()
        arrived = traci.simulation.getArrivedNumber()
        observations = {}
        for agent, tls in self.agent_to_tls.items():
            metric = self._metrics(tls)
            phase = traci.trafficlight.getPhase(tls)
            observations[agent] = np.array([
                min(metric["queue"] / 50, 1), min(metric["wait"] / 1000, 1),
                min(metric["speed"] / 15, 1), min(metric["vehicles"] / 50, 1),
                min(phase / 4, 1), min(departed / 20, 1), min(arrived / 20, 1),
                min(metric["imbalance"] / 30, 1), min(metric["max_wait"] / 300, 1),
                min(self.green_time.get(agent, 0) / self.max_green, 1),
            ], dtype=np.float32)
        return observations

    def _get_agent_reward(self, agent, switched):
        metric = self._metrics(self.agent_to_tls[agent])
        # Delay reduction, local fairness and completed trips make agents cooperate through the shared critic.
        reward = (-1.5 * metric["queue"] - 0.02 * metric["wait"] - 0.5 * metric["imbalance"]
                  + 3.0 * traci.simulation.getArrivedNumber() + (self.previous_wait[agent] - metric["wait"]))
        if switched:
            reward -= 2.0
        if metric["max_wait"] > 200:
            reward -= 20.0
        self.previous_wait[agent] = metric["wait"]
        return float(np.clip(reward / 100, -2, 2))

    def close(self):
        if traci.isLoaded():
            traci.close()
