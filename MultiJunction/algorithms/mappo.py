import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class MAPPO:


    def __init__(
        self,
        num_agents=4,
        obs_dim=4,
        action_dim=2
    ):


        from .actor import Actor
        from .critic import Critic


        self.num_agents=num_agents
        self.obs_dim = obs_dim


        self.actors = nn.ModuleList(
            [
                Actor(
                    obs_dim,
                    action_dim
                )
                for _ in range(num_agents)
            ]
        )


        self.critic = Critic(num_agents * obs_dim)



        self.optimizer=torch.optim.Adam(
            list(self.actors.parameters())
            +
            list(self.critic.parameters()),
            lr=3e-4
        )



    def select_action(self, obs, deterministic=False):


        actions={}

        log_probs={}



        for i, agent in enumerate(sorted(obs)):
            state = obs[agent]


            state=torch.tensor(
                state,
                dtype=torch.float32
            )


            probs=self.actors[i](state)


            dist=Categorical(probs)


            action = torch.argmax(probs) if deterministic else dist.sample()



            actions[agent]=action.item()

            log_probs[agent]=dist.log_prob(action)



        return actions,log_probs




    def update(self, transitions, gamma=0.99, clip_ratio=0.2, epochs=4, value_coef=0.5, entropy_coef=0.01):
        """Perform clipped PPO updates using a centralized critic and local actors."""
        if not transitions:
            return 0.0

        returns = []
        running_return = 0.0
        for transition in reversed(transitions):
            running_return = transition["reward"] + gamma * running_return * (not transition["done"])
            returns.append(running_return)
        returns.reverse()
        returns = torch.tensor(returns, dtype=torch.float32)
        states = torch.tensor(np.array([
            np.concatenate([item["obs"][agent] for agent in sorted(item["obs"])]) for item in transitions
        ]), dtype=torch.float32)
        actions = torch.tensor([[item["actions"][agent] for agent in sorted(item["obs"])] for item in transitions])
        old_log_probs = torch.tensor([[item["log_probs"][agent] for agent in sorted(item["obs"])] for item in transitions])

        with torch.no_grad():
            advantages = returns - self.critic(states).squeeze(-1)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(epochs):
            new_log_probs, entropies = [], []
            for index, agent in enumerate(sorted(transitions[0]["obs"])):
                distribution = Categorical(self.actors[index](states[:, index * self.obs_dim:(index + 1) * self.obs_dim]))
                new_log_probs.append(distribution.log_prob(actions[:, index]))
                entropies.append(distribution.entropy())
            new_log_probs = torch.stack(new_log_probs, dim=1)
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
            actor_loss = -torch.min(ratio * advantages.unsqueeze(1), clipped_ratio * advantages.unsqueeze(1)).mean()
            value_loss = (self.critic(states).squeeze(-1) - returns).pow(2).mean()
            entropy = torch.stack(entropies, dim=1).mean()
            loss = actor_loss + value_coef * value_loss - entropy_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.actors.parameters()) + list(self.critic.parameters()), 0.5)
            self.optimizer.step()
        return float(loss.detach())

    def save(self, path):
        torch.save({"actors": self.actors.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path, map_location="cpu"):
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        self.actors.load_state_dict(checkpoint["actors"])
        self.critic.load_state_dict(checkpoint["critic"])
