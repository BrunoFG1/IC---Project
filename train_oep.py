import numpy as np
import random
from tqdm import trange, tqdm
import time

import jogo
import Red_Team_Agent_1 as R1

SWARM_SIZE = 30
ITERATIONS = 50
C1 = 1.5    # acceleration constant cognitive (personal best)
C2 = 1.5    # acceleration constant social (group best)         # NOTES: C1 + C2 <= 4, so the velocities and positions from the particles dont diverge, which means if the sum is bigger than 4 the particles will be far from the swarm best solution and their solution will be even worse in the further iteractions, making that they can't find a solution better than their personal best.
W = 0.9     # inertia -> high values for the inertia makes that a bigger zone in the space is searched by the particles, small values make the search zone tiny
W_MIN = 0.51 # min value for the inertia
V_MAX = 1.0 # max velocity -> to make the particles avoid move too fast in the search space, doesn't limit the boundaries of the search zone, but controls the step size for stability.

N_EVAL = 3        # Quantos jogos rodar para testar cada partícula (média)
MUTATION_PROB = 0.1
MUTATION_STD = 0.2

class Particle:
    def __init__(self, dim):
        self.position = np.random.uniform(0.1, 3.0, dim)  # Random weights [0.1, 3.0]: positive values required for A*, spread for initial diversity.
        self.velocity = np.zeros(dim)
        self.pbest_position = self.position.copy()
        self.cur_fit = -np.inf
        self.pbest_val = -np.inf

# igual ao algoritmo genético
def calc_fitness(weights_r1):
    """Function that calcs the fitness for one particle (one team) in a game"""
    
    # add the weights from the set_weights function to the agents
    R1.set_model_weights(weights_r1)
        
    total_diff = 0.0 # a diferença de scores entre equipa verde e vermelha

    for i in range(N_EVAL):
        env = jogo.make_env()
        seed = random.randint(0, 100000) # for different maps for each test

        scores = jogo.run_match(env, render=False, seed=seed)
        # calculate the score for each team
        scr_green = sum([v for k,v in scores.items() if "Green" in k])
        scr_red   = sum([v for k,v in scores.items() if "Red" in k])

        total_diff += (scr_red - scr_green)

    return total_diff / N_EVAL
    
def main():
    dim_r1 = R1.get_model_particle_shape()

    total_dim = dim_r1
        
    swarm = [Particle(total_dim) for i in range(SWARM_SIZE)]
    gbest_position = np.zeros(total_dim)
    gbest_val = -np.inf

    start_time = time.time()

    for i in trange(ITERATIONS, desc="Iterations PSO"):
        # Decrement the inertia so we can optimize the performance
        cur_W = W - ((W - W_MIN) * (i/float(ITERATIONS))) # the equation result will be always higher than 0.4
        for p in tqdm(swarm, desc="Eval particles", leave=False):
            
            w_r1 = p.position

            fit = calc_fitness(w_r1)
            p.cur_fit = fit

            if fit > p.pbest_val:
                p.pbest_val = fit
                p.pbest_position = p.position.copy()
            if fit > gbest_val:
                gbest_val = fit
                gbest_position = p.position.copy()
                tqdm.write(f"New Record for Fit: {gbest_val:.2f} | Weights R1: {w_r1}")
        
        for p in swarm:
            r1 = np.random.rand(total_dim) # U(0,1)
            r2 = np.random.rand(total_dim) # U(0,1)
            
            # ró
            ro1 = r1 * C1
            ro2 = r2 * C2
            
            new_vel = cur_W * p.velocity + ro1 * (p.pbest_position - p.position) + ro2 * (gbest_position - p.position)
            new_vel = np.clip(new_vel, -V_MAX, V_MAX) # guarantees that the new velocity value is always in the array's interval [-V_MAX, V_MAX]  
            p.velocity = new_vel
            
            p.position = p.position + p.velocity

            if random.random() < MUTATION_PROB:
                p.position += np.random.normal(0, MUTATION_STD, total_dim)

            # guarantee that the weights are always > 0 (if not, that would broke the A* algorithm)
            p.position = np.maximum(p.position, 0.01)
    

    total_time = time.time() - start_time
    print(f"Training done in {total_time/60.0:.2f} min. Best fitness: {gbest_val:.3f}")
    print(f"Best weights for R1: {gbest_position[:dim_r1]}")
    np.save("melhores_pesos_finais.npy", gbest_position)
    print("Saved file in 'melhores_pesos_finais.npy'")

if __name__ == "__main__":
    main()
        
        
