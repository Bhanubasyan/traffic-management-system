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
from algorithms.mappo import MAPPO



parser = argparse.ArgumentParser(description="Train the multi-junction MAPPO controller.")
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--gui", action="store_true")
parser.add_argument("--model", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "mappo_v2.pt"))
parser.add_argument("--save-every", type=int, default=25, help="Save a checkpoint every N episodes.")
parser.add_argument("--resume", action="store_true", help="Continue from --model when it exists.")
args = parser.parse_args()

env = SUMOMultiAgentEnv(gui=args.gui)
agent = MAPPO(obs_dim=10)
if args.resume and os.path.exists(args.model):
    agent.load(args.model)
    print(f"Resumed model from {args.model}")
elif args.resume:
    print("No saved model found; starting a new model.")

os.makedirs(os.path.dirname(args.model), exist_ok=True)
completed_episodes = 0
print("\n" + "=" * 72)
print("MULTI-JUNCTION MAPPO TRAINING")
print("=" * 72)
print(f"Episodes: {args.episodes} | Agents: 4 | State features per agent: 10")
print(f"Checkpoint: {args.model} | Save interval: {args.save_every} episodes\n")
try:
    for ep in range(args.episodes):


        obs,_=env.reset()


        total_reward=0



        done=False


        step_count = 0
        transitions = []
        while not done:

            step_count += 1

            if step_count % 20 == 0:
                    print("Episode", ep + 1, "Step", step_count)

            actions, log_probs = agent.select_action(obs)



            next_obs,reward,terminated,truncated,info = env.step(actions)



            transitions.append({"obs": obs, "actions": actions, "log_probs": {agent: float(value.detach()) for agent, value in log_probs.items()}, "reward": sum(reward.values()), "done": all(terminated.values()) or all(truncated.values())})
            obs=next_obs



            total_reward += sum(reward.values())


            done = all(terminated.values()) or all(truncated.values())



        loss = agent.update(transitions)
        completed_episodes += 1
        print(f"Episode {ep + 1}/{args.episodes} | reward={total_reward:.3f} | loss={loss:.4f}")
        if completed_episodes % args.save_every == 0:
            agent.save(args.model)
            print(f"Checkpoint saved to {args.model}")
except KeyboardInterrupt:
    print("Training interrupted; saving the current model.")
finally:
    agent.save(args.model)
    env.close()
    print(f"Saved model to {args.model}")
