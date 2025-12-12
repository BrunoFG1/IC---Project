import numpy as np
import heapq


class AgentState:
    def __init__(self):
        self.global_pos = (0, 0)
        self.base_global_pos = None
        self.visited = set()
        self.last_action = 0
        
my_state = AgentState()

# [Cost, Heuristic, Repulsion]
PSO_WEIGHTS = [2.35977368, 1.84191254, 0.01] 

# PSO functions
def get_model_particle_shape():
    return 3

def set_model_weights(weights):
    global PSO_WEIGHTS
    PSO_WEIGHTS = weights

# A* algorithm
def manhattan_dist(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])

def path_rebuild(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.insert(0, current)
    return path

def a_star_pso(grid, start, end, teammate_pos, weights):
    w_g, w_h, w_team = weights
    rows, cols = grid.shape
    
    open_list = []
    start_h = manhattan_dist(start, end) * w_h
    heapq.heappush(open_list, (start_h, start))
    
    came_from = {}
    g_score = {start: 0}
    
    while open_list:
        current = heapq.heappop(open_list)[1]

        if current == end:
            return path_rebuild(came_from, current)

        neighbors = [
            (current[0]-1, current[1]), (current[0]+1, current[1]),
            (current[0], current[1]-1), (current[0], current[1]+1)
        ]

        for neighbor in neighbors:
            ni, nj = neighbor
            
            # limits and walls
            if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                continue
            if grid[ni, nj] == 50:
                continue


            move_cost = 1
            
            # teammate repulsion, if the teammate is in the agent's way the cost will be na weight of repulsion
            if teammate_pos and manhattan_dist(neighbor, teammate_pos) <= 1:
                move_cost += w_team 

            tentative_g_score = g_score[current] + (move_cost * w_g)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                h = manhattan_dist(neighbor, end) * w_h
                f = tentative_g_score + h
                heapq.heappush(open_list, (f, neighbor))

    return []


def clamp_target_to_grid(local_target, rows, cols):
    ty, tx = local_target
    cy = max(0, min(ty, rows-1))
    cx = max(0, min(tx, cols-1))
    return (cy, cx)


def policy(obs, agent_id=0):
    global my_state, PSO_WEIGHTS
    
    grid = obs["grid"][:, :, 0]
    vision = grid.shape[0] // 2
    agent_center_pos = (vision, vision)
    holding_flag = obs["holding"][0]
    
    # refresh gps
    move_map = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1), 0: (0, 0)}
    if my_state.last_action in move_map:
        dy, dx = move_map[my_state.last_action]
        my_state.global_pos = (my_state.global_pos[0] + dy, my_state.global_pos[1] + dx)

    # gps calibration
    bases = np.argwhere(grid == 250)
    if len(bases) > 0:
        by, bx = bases[0]
        rel_y, rel_x = by - vision, bx - vision
        
        # absolute position from the base that we are seeing
        observed_base_global = (my_state.global_pos[0] + rel_y, my_state.global_pos[1] + rel_x)
        
        # we only refresh the variable if it is the first time (game's beginning) or if the agent is close (< 15) from his base
        if my_state.base_global_pos is None:
            my_state.base_global_pos = observed_base_global
        else:
            dist = abs(my_state.base_global_pos[0] - observed_base_global[0]) + \
                   abs(my_state.base_global_pos[1] - observed_base_global[1])
            
            if dist < 15: # Our base, recalibrate the GPS
                my_state.base_global_pos = observed_base_global
                print("MY BASE!!!!")
            else: print("ENEMY BASE!!!")
            # If dist > 15, then it is the enemy base, ignore.

    my_state.visited.add(my_state.global_pos)
    
    teammate_pos = None 

    blockers = np.argwhere((grid != 0) & (grid != 50) & (grid != 100) & (grid != 250))
    for b in blockers:
        if tuple(b) != agent_center_pos:
            teammate_pos = tuple(b) # a mobile obstacle found 
            break
    
    # decision
    target = None
    if holding_flag:
        # back to the base
        if my_state.base_global_pos:
            diff_y = my_state.base_global_pos[0] - my_state.global_pos[0]
            diff_x = my_state.base_global_pos[1] - my_state.global_pos[1]
            raw_target = (vision + diff_y, vision + diff_x)
            target = clamp_target_to_grid(raw_target, grid.shape[0], grid.shape[1])
        else:
            target = (vision, vision-5)
    else:
        # go get the flag
        flags = np.argwhere(grid == 100)
        if len(flags) > 0:
            target = tuple(flags[0])


    action = 0  # the agent stays put
    if target:
        path = a_star_pso(grid, agent_center_pos, target, teammate_pos, PSO_WEIGHTS)
        if len(path) > 1:
            next_node = path[1]
            dy, dx = next_node[0] - vision, next_node[1] - vision
            if dy == -1: action = 1
            elif dy == 1: action = 2
            elif dx == -1: action = 3
            elif dx == 1: action = 4

    # to fix when agents become stuck
    dy, dx = move_map[action]
    next_cell = (vision + dy, vision + dx)
    if teammate_pos and next_cell == teammate_pos:
        action = 0

    # exploration
    if action == 0:
        possible = [1, 2, 3, 4]
        np.random.shuffle(possible)
        for act in possible:
            dy, dx = move_map[act]
            test_cell = (vision + dy, vision + dx)
            if grid[vision+dy, vision+dx] == 50:
                continue
            if teammate_pos and test_cell == teammate_pos:
                continue
            future_global = (my_state.global_pos[0] + dy, my_state.global_pos[1] + dx)
            if future_global not in my_state.visited:
                action = act
                break
        
        if action == 0:
            for act in possible:
                dy, dx = move_map[act]
                test_cell = (vision + dy, vision + dx)
                if grid[vision+dy, vision+dx] != 50:
                    if teammate_pos and test_cell == teammate_pos:
                        continue
                    action = act
                    break

    my_state.last_action = action
    return action