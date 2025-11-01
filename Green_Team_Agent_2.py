import random
import torch
from torch import nn

# Starting with RN
# Using GPU to train models
device = "cuda" if torch.cuda.is_available() else "cpu"

class RN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=51, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=7)
        )

    def forward(self, x):
        return self.model(x)

model = RN().to(device=device)
model.eval()


def policy(obs, agent_id):

    return random.randint(0, 6)



