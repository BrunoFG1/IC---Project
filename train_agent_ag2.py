# treino.py
import random
import torch
from torch import nn
import numpy as np
from jogo import run_match, make_env

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

class AG:
    def __init__(self, model, pop_ini=30, prob_mut=0.2, mut_scale=0.1, elite_frac=0.2):
        self.model = model
        self.pop_ini = pop_ini
        self.prob_mut = prob_mut
        self.mut_scale = mut_scale
        self.elite_frac = elite_frac

    def num_param(self):
        return sum(p.numel() for p in self.model.parameters())

    def generate_pop(self):
        return np.random.uniform(-1, 1, (self.pop_ini, self.num_param()))

    def set_weights(self, individuo):
        index = 0
        for p in self.model.parameters():
            size = p.numel()
            part = individuo[index:index+size].reshape(p.shape)
            p.data = torch.tensor(part, dtype=p.dtype, device=p.device)
            index += size

    def aptidao(self, individuo):
        self.set_weights(individuo)
        self.model.eval()
        fitness = 0
        n_jogos = 20 
        for _ in range(n_jogos):
            env = make_env()
            results = run_match(env, render=False)  
            team_reward = sum(r for a, r in results.items() if env.team[a] == 0)
            fitness += team_reward
        return fitness / n_jogos

    def crossover(self, p1, p2):
        corte = random.randint(0, len(p1)-1)
        return np.concatenate([p1[:corte], p2[corte:]])

    def mutacao(self, individuo):
        for i in range(len(individuo)):
            if random.random() < self.prob_mut:
                individuo[i] += np.random.normal(0, self.mut_scale)
        return individuo

    def evoluir(self, n_geracoes=20):
        populacao = self.generate_pop()
        for g in range(n_geracoes):
            fitness = np.array([self.aptidao(ind) for ind in populacao])
            idx = np.argsort(fitness)[::-1]
            populacao = populacao[idx]
            fitness = fitness[idx]
            print(f"Geração {g} | Melhor fitness = {fitness[0]:.2f}")

            # Elitismo
            n_elite = int(self.elite_frac * self.pop_ini)
            nova_pop = populacao[:n_elite].tolist()  

            # Reproduzir
            while len(nova_pop) < self.pop_ini:
                p1, p2 = random.sample(populacao[:10].tolist(), 2)
                filho = self.crossover(np.array(p1), np.array(p2))
                filho = self.mutacao(filho)
                nova_pop.append(filho)

            populacao = np.array(nova_pop)
        return populacao[0]


def policy(obs, agent_id):
    grid = obs['grid'].flatten()
    holding = obs['holding']
    stun = obs['stun']
    obs_vector = np.concatenate([grid, holding, stun])
    obs_tensor = torch.tensor(obs_vector, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        action = model(obs_tensor).argmax(1).item()
    return action


if __name__ == "__main__":
    model = RN().to(device)
    ag = AG(model)
    print("Treinando agente evolutivo...")
    best_weights = ag.evoluir(n_geracoes=10)  
    ag.set_weights(best_weights)

    torch.save(model.state_dict(), "best_model.pth")
    print("Pesos guardados em best_model.pth")