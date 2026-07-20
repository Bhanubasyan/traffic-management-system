import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):

    def __init__(self, obs_dim=4, action_dim=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim,128),
            nn.ReLU(),

            nn.Linear(128,128),
            nn.ReLU(),

            nn.Linear(128,action_dim)
        )


    def forward(self,x):

        logits = self.net(x)

        return F.softmax(
            logits,
            dim=-1
        )