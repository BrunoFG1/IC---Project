import numpy as np
import random
from tqdm import trange, tqdm # bar progress

# import game env
import jogo

# import agents from green team with CE and RN startegy
import Green_Team_Agent_1 as G1
import Green_Team_Agent_2 as G2

# import agents from red team with random strategy 
import Red_Team_Agent_1
import Red_Team_Agent_2

POPULATION_SIZE = 30     # Teams/Populations for generation
NUM_GENERATIONS = 50     # Number of generations for training. 
MUTATION_RATE = 0.05     # 5% chance of mutation in a weight 
MUTATION_STRENGTH = 0.1  # mutation strength (a bit of noise)
ELITISM_COUNT = 5        # Number of teams that go through to the next generation

def create_pop(team_chromo_len):
    """Creates an initial population with random team chromosomes."""
    pop = []
    for i in range(POPULATION_SIZE):
        team_chromo = np.random.rand(team_chromo_len) * 2 - 1
        pop.append(team_chromo)
    return pop

def calc_fitness(team_chromo, ag1_len, ag2_len):
    """Function that calcs the fitness for one chromossome (one team) in a game"""
    c1 = team_chromo[0 : ag1_len]
    c2 = team_chromo[ag1_len : ag1_len + ag2_len]
    
    # add the weights from the set_weights function to the agents
    G1.set_model_weights(c1)
    G2.set_model_weights(c2)
    
    # run the game environment
    env = jogo.make_env()
    
    # the run_match() function will use the policies with the new weights, and will return the total score for each agent
    agents_tot = jogo.run_match(env, render=False, seed=None)
    
    # calcs the team score against the enemy team
    name_team1 = env.team_names[0]
    name_team2 = env.team_names[1]
    
    scr_team1 = 0.0
    scr_team2 = 0.0
    
    for ag_name, scr in agents_tot.items():
        if ag_name.startswith(name_team1):
            scr_team1 += scr
        elif ag_name.startswith(name_team2):
            scr_team2 += scr
    fitness = scr_team1 - scr_team2
    return fitness

def selection(pops, fits):
    """Select the parents for the next generation"""
    parents = []
    for i in range(POPULATION_SIZE):
        idx1 = random.randint(0, POPULATION_SIZE - 1)
        idx2 = random.randint(0, POPULATION_SIZE - 1)
        if fits[idx1] > fits[idx2]:
            parents.append(pops[idx1])
        else:
            parents.append(pops[idx2])
    return parents

def crossover(par1, par2):
    """Performs single-point crossover with two parents"""
    point = random.randint(1, len(par1) - 2)
    
    # here we create the children by mixing the genes from their parents 
    child1 = np.concatenate([par1[:point], par2[point:]])
    child2 = np.concatenate([par2[:point], par1[point:]])
    return child1, child2

def mutate(chromo):
    """Mutation to the chromossome weights"""
    for i in range(len(chromo)):
        if random.random() < MUTATION_RATE:
            mut = np.random.normal(0, MUTATION_STRENGTH)
            chromo[i] += mut
    return chromo



def main():
    """Main function to train the GA"""
    ag1_len = G1.get_model_chromo_shape()
    ag2_len = G2.get_model_chromo_shape()
    
    team_chromo_len = ag1_len + ag2_len
    print(f"Total chromossome size: {team_chromo_len}")
    
    pop = create_pop(team_chromo_len)
    best_fit = -np.inf # initialize with the lowest possible value
    best_team = None
    
    for gen in trange(NUM_GENERATIONS, desc="Generations"): # progress bar for generations
        fits = []
        # show inner progress for evaluating the population
        for tc in tqdm(pop, desc=f"Gen {gen+1} eval", leave=False):
            fit = calc_fitness(tc, ag1_len, ag2_len)
            fits.append(fit)
        mean_fit = np.mean(fits)
        best_fit_gen = np.max(fits)
        
        if best_fit_gen > best_fit:
            best_fit = best_fit_gen
            best_team = pop[np.argmax(fits)].copy() # return the index of the most fit chromossome in the population
        
        print(f"Generation {gen+1}/{NUM_GENERATIONS}")
        print(f"The best score: {best_fit_gen:.2f}")
        print(f"The mean score: {mean_fit:.2f}")

        sort_idxs = np.argsort(fits)[::-1] # indexes from highest to lowest fitness score
        new_pop = []
        for i in sort_idxs[:ELITISM_COUNT]:
            new_pop.append(pop[i])
        
        parents = selection(pop, fits)

        while len(new_pop) < POPULATION_SIZE:
            # 2 random parents are chosen
            id1 = np.random.randint(0, len(parents))
            p1 = parents[id1]

            id2 = np.random.randint(0, len(parents))
            p2 = parents[id2]

            while id1 == id2:
                id2 = np.random.randint(0, len(parents))
            p2 = parents[id2]
            # 2 children are created
            c1, c2 = crossover(p1, p2)
            # added to the new population the mutated chromossomes
            new_pop.append(mutate(c1))
            if len(new_pop) < POPULATION_SIZE:
                new_pop.append(mutate(c2))  
    
    pop = new_pop
    # train over
    print("GA train is done!\n")
    print(f"Best team by fitness: {best_fit}")
    # save the file for the best chromossome
    file = "best_chrome_team.npy"
    np.save(file, best_team)
    print(f"Best team saved in npy file: {file}")

if __name__ == "__main__":
    main()

