import os
import torch
from train_agent_ag2 import RN, device
import numpy as np

model = RN().to(device)
if os.path.exists("best_model.pt"):
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

'''def policy(obs, agent_id):
    import numpy as np
    grid = obs['grid'].flatten()
    holding = obs['holding']
    stun = obs['stun']
    obs_vector = np.concatenate([grid, holding, stun])
    obs_tensor = torch.tensor(obs_vector, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        action = model(obs_tensor).argmax(1).item()
    return action'''

import random

EPSILON = 0.1  # 10% do tempo escolhe ação aleatória

def policy(obs, agent_id):
    grid = obs['grid'].flatten()
    holding = obs['holding']
    stun = obs['stun']
    obs_vector = np.concatenate([grid, holding, stun])
    obs_tensor = torch.tensor(obs_vector, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        q_values = model(obs_tensor).squeeze(0).cpu().numpy()
    
    if random.random() < EPSILON:
        # ação aleatória (exploração)
        action = random.randint(0, len(q_values)-1)
    else:
        # ação com maior Q-value (explotação)
        action = int(np.argmax(q_values))
    
    return action

