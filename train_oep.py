import numpy as np
import random
from tqdm import trange, tqdm

import jogo
import Green_Team_Agent_1
import Green_Team_Agent_2 
import Red_Team_Agent_1 as R1
import Red_Team_Agent_2 as R2

SWARM_SIZE = 30
ITERATIONS = 50
C1 = 1.5    # acceleration constant cognitive (personal best)
C2 = 1.5    # acceleration constant social (group best)         # NOTES: C1 + C2 <= 4, so the velocities and positions from the particles dont diverge, which means if the sum is bigger than 4 the particles will be far from the swarm best solution and their solution will be even worse in the further iteractions, making that they can't find a solution better than their personal best.
W = 0.9     # inertia -> high values for the inertia makes that a bigger zone in the space is searched by the particles, small values make the search zone tiny
W_MIN = 0.6 # min value for the inertia
V_MAX = 1.0 # max velocity -> to make the particles avoid move too fast in the search space, doesn't limit the boundaries of the search zone, but controls the step size for stability.

class Particle:
    def __init__(self, team_part):
        self.position = np.random.rand(team_part) * 2 - 1  # array with 3559 numbers between -1 and 1
        self.velocity = np.zeros(team_part)
        self.pbest_position = self.position.copy()
        self.cur_fit = -np.inf
        self.pbest_val = -np.inf

# igual ao algoritmo genético
def calc_fitness(team_part, ag1_len, ag2_len):
    """Function that calcs the fitness for one particle (one team) in a game"""
    p1 = team_part[0 : ag1_len]
    p2 = team_part[ag1_len : ag1_len + ag2_len]
        
    # add the weights from the set_weights function to the agents
    R1.set_model_weights(p1)
    R2.set_model_weights(p2)
        
    # run the game environment
    env = jogo.make_env()
        
    # the run_match() function will use the policies with the new weights, and will return the total score for each agent
    agents_tot = jogo.run_match(env, render=False, seed=None)
        
    # calcs the team score against the enemy team
    name_team1 = env.team_names[0] # green
    name_team2 = env.team_names[1] # red
        
    scr_team1 = 0.0 # green
    scr_team2 = 0.0 # red
        
    for ag_name, scr in agents_tot.items():
        if ag_name.startswith(name_team1):
            scr_team1 += scr
        elif ag_name.startswith(name_team2):
            scr_team2 += scr
    fitness = scr_team2 - scr_team1 # if fitness is > 0 red team won else green team won
    return fitness
    
def main():
    ag1_len = R1.get_model_particle_shape()
    ag2_len = R2.get_model_particle_shape()
    team_part = ag1_len + ag2_len
        
    swarm = [Particle(team_part) for i in range(SWARM_SIZE)]
    gbest_position = np.zeros(team_part)
    gbest_val = -np.inf
        
    for i in trange(ITERATIONS, desc="Iterations"):
        # Decrement the inertia so we can optimize the performance
        cur_W = W - ((W - W_MIN) * (i/ITERATIONS)) # the equation result will be always higher than 0.6
        for p in tqdm(swarm, desc="Partículas do enxame", leave=False):
            fit = calc_fitness(p.position, ag1_len, ag2_len)
            p.cur_fit = fit
            if p.cur_fit > p.pbest_val:
                p.pbest_val = p.cur_fit
                p.pbest_position = p.position.copy()
            
            if p.cur_fit > gbest_val:
                gbest_val = p.cur_fit
                gbest_position = p.position.copy()
        
        for p in swarm:
            r1 = np.random.rand() # U(0,1)
            r2 = np.random.rand() # U(0,1)
            
            # ró
            ro1 = r1 * C1
            ro2 = r2 * C2
            
            new_vel = cur_W * p.velocity + ro1 * (p.pbest_position - p.position) + ro2 * (gbest_position - p.position)
            new_vel = np.clip(new_vel, -V_MAX, V_MAX) # guarantees that the new velocity value is always in the array's interval [-V_MAX, V_MAX]  
            p.velocity = new_vel
            
            p.position = p.position + p.velocity
    
    print(f"Iteration {i+1}/{ITERATIONS} -> Best score: {gbest_val}")
    file = "best_particle_red_team.npy"
    np.save(file, gbest_position)
    print(f"Best team file saved to: {file}")
    
if __name__ == "__main__":
    main()   
        
        
