import numpy as np
import heapq

path_traveled = []
path_to_flag = []
base_pos = None
visited = set()  
goal_base_pos = None


def fuzzy_distance_to_base(dist):

    # Perto: 0 a 3
    if dist <= 1:
        perto = 1
    elif dist <= 3:
        perto = (3 - dist) / 2
    else:
        perto = 0

    # Médio: 2 a 7
    if 2 <= dist <= 5:
        medio = (dist - 2) / 3
    elif 5 < dist <= 7:
        medio = (7 - dist) / 2
    else:
        medio = 0

    # Longe: 5 a 20
    if dist >= 10:
        longe = 1
    elif dist >= 5:
        longe = (dist - 5) / 5
    else:
        longe = 0

    return perto, medio, longe


def fuzzy_decision(perto, medio, longe):
    """
    Regras:
    - PERTO → tentar A*
    - MÉDIO → reverse path
    - LONGE → random
    """
    rules = {
        "astar": perto,
        "reverse": medio,
        "random": longe
    }

    # decisão = maior pertença
    decision = max(rules, key=rules.get)
    return decision


def fuzzy_choose_action(dist):
    perto, medio, longe = fuzzy_distance_to_base(dist)
    return fuzzy_decision(perto, medio, longe)



def manhattan_dist(x, y):
    return np.sum(np.abs(np.array(x) - np.array(y)))
   
def path_rebuild(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.insert(0, current)
    return path

def a_star(grid, start, end):
    open_list = []
    heapq.heappush(open_list, (manhattan_dist(start, end), start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: manhattan_dist(start, end)}

    while open_list:
        current = heapq.heappop(open_list)[1]

        if current == end:
            return path_rebuild(came_from, current)

        neighbors = [
            (current[0]-1, current[1]),
            (current[0]+1, current[1]),
            (current[0], current[1]-1),
            (current[0], current[1]+1)
        ]

        for neighbor in neighbors:
            i, j = neighbor
            if i < 0 or i >= grid.shape[0] or j < 0 or j >= grid.shape[1]:
                continue
            if grid[i, j] == 50:
                continue

            tentative_g_score = g_score[current] + 1

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + manhattan_dist(neighbor, end)
                heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return []


def clamp_target_to_grid(local_target, rows, cols):
    ty, tx = local_target
    cy = max(0, min(ty, rows-1))
    cx = max(0, min(tx, cols-1))
    return (cy, cx)


def policy(obs, agent):
    global path_traveled, path_to_flag, base_pos, visited, goal_base_pos

    grid = obs["grid"][:, :, 0]
    vision = grid.shape[0] // 2
    cy, cx = vision, vision
    holding_flag = obs["holding"][0]

    valid_moves = []
    flag = []

    reverse_action = {1:2, 2:1, 3:4, 4:3}
    moves_ = {'up': 1, 'down': 2, 'left':3, 'right':4}
    visited.add((cy, cx))

    if cy > 0 and grid[cy-1, cx] != 50: valid_moves.append('up')
    if cy < grid.shape[0]-1 and grid[cy+1, cx] != 50: valid_moves.append('down')
    if cx > 0 and grid[cy, cx-1] != 50: valid_moves.append('left')
    if cx < grid.shape[1]-1 and grid[cy, cx+1] != 50: valid_moves.append('right')

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            valor = grid[i, j]
            dy = i - cy
            dx = j - cx

            if valor == 50:
                continue

            if valor == 250 and base_pos is None:
                base_pos = (dy, dx)
                print("Base é aqui:", base_pos)

            if valor == 100:
                flag.append((dy, dx))



    if holding_flag:

        if goal_base_pos is None:
            goal_base_pos = base_pos

        agent_pos = (vision, vision)
        raw_target = (agent_pos[0] + goal_base_pos[0],
              agent_pos[1] + goal_base_pos[1])

        goal = clamp_target_to_grid(raw_target, grid.shape[0], grid.shape[1])

        dist = manhattan_dist(agent_pos, goal)

        decision = fuzzy_choose_action(dist)
        print("Fuzzy decision:", decision, "Dist =", dist)

        # DECISÃO 1: usar A*
        if decision == "astar":
            path = a_star(grid, agent_pos, goal)
            if len(path) > 2:
                ny, nx = path[1]
                dy2 = ny - agent_pos[0]
                dx2 = nx - agent_pos[1]

                if dy2 < 0: return 1
                if dy2 > 0: return 2
                if dx2 < 0: return 3
                if dx2 > 0: return 4
            else:
                decision = "reverse" 

        # DECISÃO 2: reverse path
        if decision == "reverse" and path_to_flag:
            last_move = path_to_flag.pop()
            return reverse_action[last_move]

        # DECISÃO 3: random
        move = np.random.choice(valid_moves)
        return moves_[move]



    if flag:
        agent_pos = (vision, vision)
        fy, fx = flag[0]
        target = (vision + fy, vision + fx)
        path = a_star(grid, agent_pos, target)

        if len(path) < 2:
            return np.random.choice(move_options)
        else:
            ny, nx = path[1]
            dy2 = ny - agent_pos[0]
            dx2 = nx - agent_pos[1]
            if dy2 < 0: return 1
            if dy2 > 0: return 2
            if dx2 < 0: return 3
            if dx2 > 0: return 4

    move_options = []
    if 'up' in valid_moves and (cy-1, cx) not in visited: move_options.append(1)
    if 'down' in valid_moves and (cy+1, cx) not in visited: move_options.append(2)
    if 'left' in valid_moves and (cy, cx-1) not in visited: move_options.append(3)
    if 'right' in valid_moves and (cy, cx+1) not in visited: move_options.append(4)
                
    if move_options:
        return np.random.choice(move_options)
    else:
        move_map = {'up':1, 'down':2, 'left':3, 'right':4}
        return move_map[np.random.choice(valid_moves)]