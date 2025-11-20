import random
import torch
from torch import nn
import numpy as np
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
    
model = RN().to(device)
model.eval()


class AG:
    def __init__(self, model, pop_ini = 30, prob_mut=0.2, mut_scale=0.1, elite_frac=0.2):
        self.model = model
        self.pop_ini = pop_ini
        self.prob_mut = prob_mut
        self.mut_scale = mut_scale
        self.elite_frac = elite_frac

    def num_param(self):
        sum = 0
        for p in self.model.parameters():
            sum +=p.numel()
        return sum

    def generate_pop(self):
        lim_inf, lim_max = (-1, 1)
        return np.random.uniform(lim_inf, lim_max, (self.pop_ini, self.num_param()))

    def aptidao(self, individuo):
        return 0



def policy(obs, agent_id):
    grid = obs['grid'].flatten()
    holding = obs['holding']
    stun = obs['stun']

    obs_vector = np.concatenate([grid, holding, stun])
    obs_tensor = torch.tensor(obs_vector, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        action = model(obs_tensor).argmax(1).item()

    return action