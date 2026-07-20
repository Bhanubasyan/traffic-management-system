import argparse
import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
from environment.sumo_mappo_env import SUMOMultiAgentEnv


parser = argparse.ArgumentParser(description="Smoke-test the SUMO multi-agent environment.")
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--gui", action="store_true")
args = parser.parse_args()

env = SUMOMultiAgentEnv(gui=args.gui)
obs, _ = env.reset()
print(f"Agents: {list(obs)}")

for step in range(args.steps):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminated, truncated, _ = env.step(actions)
    print(f"Step {step + 1}: reward={sum(rewards.values()):.3f}")
    if all(terminated.values()) or all(truncated.values()):
        break

env.close()
