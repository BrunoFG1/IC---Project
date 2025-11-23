# Código para IC, ano letivo 2025-26, UBI
# Install: pip install pettingzoo gymnasium numpy pygame

import random
import time
import json
import sys
import math
import numpy as np
from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces

# ---------- Environment config ----------
GRID_W = 20
GRID_H = 20
NUM_FLAGS = 8
TEAM_SIZE = 2  # total agents = 2 * TEAM_SIZE
VISION = 3     # L_inf vision radius
MAX_STEPS = 300
STUN_TIME = 10  # steps stunned when tagged
TEAM_NAMES = ["Green Team", "Red Team"]  # Configurable team names
CAMERA_ZOOM = 32
# ----------------------------------------

def make_env(grid_w=GRID_W, grid_h=GRID_H, team_size=TEAM_SIZE, num_flags=NUM_FLAGS,
             vision=VISION, max_steps=MAX_STEPS, team_names=TEAM_NAMES):
    return CaptureTheFlagParallel(grid_w, grid_h, team_size, num_flags, vision, max_steps, team_names)

class CaptureTheFlagParallel(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, grid_w=15, grid_h=11, team_size=4, num_flags=5, vision=3, max_steps=300, team_names=["Team 1", "Team 2"]):
        super().__init__()
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.team_size = team_size
        self.num_flags = num_flags
        self.vision = vision
        self.max_steps = max_steps
        self.team_names = team_names

        self.agents = [f"{self.team_names[0]} Agent {i+1}" for i in range(team_size)] + \
                      [f"{self.team_names[1]} Agent {i+1}" for i in range(team_size)]
        self.pos = {}
        self.team = {a: 0 if a.startswith(self.team_names[0]) else 1 for a in self.agents}
        self.flag_positions = []
        self.initial_flag_positions = []  # Store initial flag positions
        self.flag_holder = {i: None for i in range(num_flags)}
        #self.base_pos = {0: (1, self.grid_h//2), 1: (self.grid_w-2, self.grid_h//2)}


        self.stun_timers = {a: 0 for a in self.agents}
        self.holding_time = {a: 0 for a in self.agents}
        self.cum_team_rewards = {0: 0.0, 1: 0.0}
        self.captures = {0: 0, 1: 0}
        self.steps = 0

        # Action space: 0:no-op,1:up,2:down,3:left,4:right,5:tag,6:drop
        act_space = spaces.Discrete(7)
        obs_dim = ((2*vision+1), (2*vision+1))
        self.observation_spaces = {a: spaces.Dict({
            "grid": spaces.Box(low=0, high=255, shape=(obs_dim[0], obs_dim[1], 1), dtype=np.uint8),
            "holding": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8),
            "stun": spaces.Box(low=0, high=STUN_TIME, shape=(1,), dtype=np.int8)
        }) for a in self.agents}
        self.action_spaces = {a: act_space for a in self.agents}

        # simple wall layout
        self.walls = set()
        for x in range(self.grid_w):
            self.walls.add((x,0)); self.walls.add((x,self.grid_h-1))
        for y in range(self.grid_h):
            self.walls.add((0,y)); self.walls.add((self.grid_w-1,y))
        for i in range((self.grid_w*self.grid_h)//20):
            x = random.randint(2, self.grid_w-3)
            y = random.randint(2, self.grid_h-3)
            self.walls.add((x,y))
            sx = self.grid_w-1-x; sy = self.grid_h-1-y
            self.walls.add((sx,sy))

        # Random row for bases, same for both teams
        base_y = random.randint(1, self.grid_h-2)  # Avoid border walls
        attempts = 0
        while (not self._in_bounds(1, base_y) or not self._in_bounds(self.grid_w-2, base_y)) and attempts < 10:
            base_y = random.randint(1, self.grid_h-2)
            attempts += 1
        self.base_pos = {0: (1, base_y), 1: (self.grid_w-2, base_y)}


    def seed(self, s):
        random.seed(s); np.random.seed(s)

    def reset(self, seed=None, return_info=False, options=None):
        if seed is not None:
            self.seed(seed)
        self.steps = 0
        self.cum_team_rewards = {0: 0.0, 1: 0.0}
        self.captures = {0: 0, 1: 0}
        self.pos = {}
        for i, a in enumerate(self.agents):
            t = self.team[a]
            bx, by = self.base_pos[t]
            rx = bx + (1 if t==0 else -1)
            ry = by + (i % 3) - 1
            self.pos[a] = (max(1, min(self.grid_w-2, rx)), max(1, min(self.grid_h-2, ry)))
            self.stun_timers[a] = 0
            self.holding_time[a] = 0
        # place flags symmetrically in middle band, avoiding walls
        self.flag_positions = []
        self.initial_flag_positions = []  # Reset initial positions
        mid_x = self.grid_w // 2
        start_x = max(2, mid_x - self.num_flags // 2)
        half_flags = self.num_flags // 2
        placed_positions = set()
        for k in range(half_flags):
            base_x = start_x + (k % half_flags)
            base_y = self.grid_h // 2 + (k % 3) - 1
            x_left = base_x
            y_left = base_y
            x_right = self.grid_w - 1 - x_left
            y_right = y_left
            attempts = 0
            while (not self._in_bounds(x_left, y_left) or
                   not self._in_bounds(x_right, y_right) or
                   (x_left, y_left) in placed_positions or
                   (x_right, y_right) in placed_positions) and attempts < 10:
                y_left = (y_left + 1) % (self.grid_h - 2) + 1
                y_right = y_left
                x_right = self.grid_w - 1 - x_left
                attempts += 1
            if attempts < 10:
                self.flag_positions.append((x_left, y_left))
                self.flag_positions.append((x_right, y_right))
                self.initial_flag_positions.append((x_left, y_left))
                self.initial_flag_positions.append((x_right, y_right))
                placed_positions.add((x_left, y_left))
                placed_positions.add((x_right, y_right))
        self.flag_holder = {i: None for i in range(self.num_flags)}
        observations = self._get_obs_all()
        if return_info:
            return observations, {}
        return observations



    def _in_bounds(self, x,y):
        return 0 <= x < self.grid_w and 0 <= y < self.grid_h and (x,y) not in self.walls

    def _get_obs_all(self):
        obs = {}
        for a in self.agents:
            obs[a] = self._get_obs(a)
        return obs

    def _draw_stats_overlay(self, surf, cell, stats_x=0, stats_w=300):
        try:
            import pygame
        except Exception:
            return
        w, h = surf.get_size()
        stats_w = int(300 * (cell / 20.0))
        panel_height = int(h * (cell / 20.0))
        panel_rect = pygame.Rect(stats_x + 8, 8, stats_w - 16, panel_height - 16)
        pygame.draw.rect(surf, (18, 18, 18), panel_rect)
        pygame.draw.rect(surf, (120, 120, 120), panel_rect, 2)

        title_font_size = int(20 * (cell / 20.0))
        small_font_size = int(16 * (cell / 20.0))
        title_font = pygame.font.SysFont(None, title_font_size)
        small_font = pygame.font.SysFont(None, small_font_size)

        title_surf = title_font.render(f"Match time: {self.steps}/{self.max_steps}", True, (230, 230, 230))
        surf.blit(title_surf, (panel_rect.x + 10, panel_rect.y + 8))

        t0 = small_font.render(f"{self.team_names[0]} score: {self.cum_team_rewards[0]:.2f}", True, (0, 200, 0))
        t1 = small_font.render(f"{self.team_names[1]} score: {self.cum_team_rewards[1]:.2f}", True, (200, 0, 0))
        surf.blit(t0, (panel_rect.x + 10, panel_rect.y + int(36 * (cell / 20.0))))
        surf.blit(t1, (panel_rect.x + 10, panel_rect.y + int(56 * (cell / 20.0))))

        header = small_font.render("Agents:", True, (200, 200, 200))
        surf.blit(header, (panel_rect.x + 10, panel_rect.y + int(82 * (cell / 20.0))))

        line_y = panel_rect.y + int(104 * (cell / 20.0))
        line_h = int(18 * (cell / 20.0))
        for a in self.agents:
            team_col = (0, 200, 0) if self.team[a] == 0 else (200, 0, 0)
            name_surf = small_font.render(a, True, team_col)  # Uses team name + Agent i
            surf.blit(name_surf, (panel_rect.x + 10, line_y))

            holding = any(self.flag_holder[f] == a for f in self.flag_holder)
            hold_text = "Flag:Yes" if holding else "Flag:No"
            hold_surf = small_font.render(hold_text, True, (255, 215, 0))
            surf.blit(hold_surf, (panel_rect.x + int(130 * (cell / 20.0)), line_y))

            stun = int(self.stun_timers.get(a, 0))
            stun_surf = small_font.render(f"stun:{stun}", True, (200, 200, 200) if stun == 0 else (255, 100, 100))
            surf.blit(stun_surf, (panel_rect.x + int(190 * (cell / 20.0)), line_y))

            line_y += line_h
            if line_y > panel_rect.y + panel_rect.height - int(12 * (cell / 20.0)):
                break


    def _get_obs(self, agent):
        ax, ay = self.pos[agent]
        vis = self.vision
        size = 2*vis+1
        grid = np.zeros((size, size, 1), dtype=np.uint8)
        # encode: 0 empty, 50 wall, 100 flag, 150 ally, 200 enemy, 250 base
        for dx in range(-vis, vis+1):
            for dy in range(-vis, vis+1):
                x = ax + dx; y = ay + dy
                cx = dx+vis; cy = dy+vis
                if not (0<=x<self.grid_w and 0<=y<self.grid_h):
                    grid[cy, cx, 0] = 50
                    continue
                if (x,y) in self.walls:
                    grid[cy, cx, 0] = 50
                    continue
                # flags
                for fi, fp in enumerate(self.flag_positions):
                    if self.flag_holder[fi] is None and (x,y)==fp:
                        grid[cy, cx, 0] = 100
                # bases
                if (x,y)==self.base_pos[0] or (x,y)==self.base_pos[1]:
                    grid[cy, cx, 0] = 250
                # agents
                for b, p in self.pos.items():
                    if p==(x,y):
                        grid[cy, cx, 0] = 150 if self.team[b]==self.team[agent] else 200
        holding = np.array([1 if any(self.flag_holder[f]==agent for f in self.flag_holder) else 0], dtype=np.int8)
        stun = np.array([self.stun_timers[agent]], dtype=np.int8)
        return {"grid": grid, "holding": holding, "stun": stun}


    def step(self, actions):
        self.steps += 1
        rewards = {a: 0.0 for a in self.agents}
        dones = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}

        # process actions
        desired_pos = {}
        for a, act in actions.items():
            if self.stun_timers[a] > 0:
                self.stun_timers[a] -= 1
                desired_pos[a] = self.pos[a]
                continue
            x, y = self.pos[a]
            if act == 0:
                desired_pos[a] = (x, y)
            elif act == 1:
                desired_pos[a] = (x, y-1)
            elif act == 2:
                desired_pos[a] = (x, y+1)
            elif act == 3:
                desired_pos[a] = (x-1, y)
            elif act == 4:
                desired_pos[a] = (x+1, y)
            elif act == 5:  # tag
                desired_pos[a] = (x, y)
            elif act == 6:  # drop
                desired_pos[a] = (x, y)
            else:
                desired_pos[a] = (x, y)

        # resolve movements
        new_pos = dict(self.pos)
        occupied = set(self.pos.values())
        agent_list = list(self.agents)
        random.shuffle(agent_list)
        for a in agent_list:
            p = desired_pos.get(a, self.pos[a])
            if p == self.pos[a]:
                continue
            if not self._in_bounds(*p):
                continue
            if p in occupied:
                continue
            #occupied.remove(self.pos[a])
            if self.pos[a] in occupied:
                occupied.remove(self.pos[a])
            occupied.add(p)
            new_pos[a] = p
        self.pos = new_pos

        # process tag actions
        for a, act in actions.items():
            if act == 5 and self.stun_timers[a] == 0:
                ax, ay = self.pos[a]
                for b, (bx, by) in self.pos.items():
                    if self.team[b] == self.team[a]:
                        continue
                    if abs(ax - bx) <= 1 and abs(ay - by) <= 1:
                        self.stun_timers[b] = STUN_TIME
                        # Drop any flag the stunned agent is holding to its initial position
                        for fi, holder in self.flag_holder.items():
                            if holder == b:
                                self.flag_holder[fi] = None
                                self.flag_positions[fi] = self.initial_flag_positions[fi]
                                rewards[a] += 0.2  # Reward for tagging
                                rewards[b] -= 0.1  # Penalty for being tagged

        # process pickups and drops
        for a in self.agents:
            if actions[a] == 6:
                for fi, holder in self.flag_holder.items():
                    if holder == a:
                        self.flag_holder[fi] = None
                        self.flag_positions[fi] = self.initial_flag_positions[fi]  # Respawn at initial position
            # Only pick up if not holding a flag
            if not any(self.flag_holder[fi] == a for fi in self.flag_holder):
                for fi, fp in enumerate(self.flag_positions):
                    if self.flag_holder[fi] is None and self.pos[a] == fp:
                        self.flag_holder[fi] = a
                        self.flag_positions[fi] = (-1, -1)
                        break

        # scoring: return to base
        for fi, holder in list(self.flag_holder.items()):
            if holder is None:
                continue
            base = self.base_pos[self.team[holder]]
            if self.pos[holder] == base:
                self.captures[self.team[holder]] += 1
                rewards[holder] += 3.0
                for ally in self.agents:
                    if self.team[ally] == self.team[holder] and ally != holder:
                        rewards[ally] += 0.1
                mid_x = self.grid_w // 2
                start_x = max(2, mid_x - self.num_flags // 2)
                x = start_x + (fi % (self.num_flags // 2))
                y = self.grid_h // 2 + (fi % 3) - 1
                spawn = (x, y) if fi < self.num_flags // 2 else (self.grid_w - 1 - x, y)
                self.flag_positions[fi] = spawn
                self.flag_holder[fi] = None

        # holding time reward
        for a in self.agents:
            if any(self.flag_holder[f] == a for f in self.flag_holder):
                rewards[a] += 0.01
                self.holding_time[a] += 1

        # accumulate rewards
        for a, r in rewards.items():
            self.cum_team_rewards[self.team[a]] += r

        obs = self._get_obs_all()
        done_flag = self.steps >= self.max_steps
        dones = {a: done_flag for a in self.agents}
        return obs, rewards, dones, infos


    def render(self, mode="human", camera_zoom=CAMERA_ZOOM):
        try:
            import pygame
        except Exception:
            return None
        cell = camera_zoom
        grid_px_w = self.grid_w * cell
        grid_px_h = self.grid_h * cell
        stats_w = int(300 * (camera_zoom / 20.0))  # Scale stats panel width
        total_w = grid_px_w + stats_w
        total_h = max(grid_px_h, int(200 * (camera_zoom / 20.0)))  # Scale height

        if not hasattr(self, "_screen"):
            pygame.init()
            pygame.font.init()
            self._screen = pygame.display.set_mode((total_w, total_h))
            pygame.display.set_caption("UBI -- IC 2025-26")

        # Process Pygame events to prevent "not responding" warning
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        surf = self._screen
        surf.fill((30, 30, 30))

        grid_rect = pygame.Rect(0, 0, grid_px_w, grid_px_h)
        pygame.draw.rect(surf, (20, 20, 20), grid_rect)

        for x in range(self.grid_w):
            for y in range(self.grid_h):
                px = x * cell
                py = y * cell
                if (x, y) in self.walls:
                    pygame.draw.rect(surf, (80, 80, 80), (px, py, cell, cell))

        for fi, fp in enumerate(self.flag_positions):
            x, y = fp
            if 0 <= x < self.grid_w and 0 <= y < self.grid_h:
                pygame.draw.circle(surf, (255, 200, 0), (x * cell + cell // 2, y * cell + cell // 2), cell // 3)

        for t, bp in self.base_pos.items():
            x, y = bp
            col = (50, 150, 50) if t == 0 else (150, 50, 50)
            pygame.draw.rect(surf, col, (x * cell, y * cell, cell, cell))

        for a, p in self.pos.items():
            x, y = p
            t = self.team[a]
            col = (0, 200, 0) if t == 0 else (200, 0, 0)
            pygame.draw.circle(surf, col, (x * cell + cell // 2, y * cell + cell // 2), cell // 3)
            if any(self.flag_holder[f] == a for f in self.flag_holder):
                pygame.draw.circle(surf, (255, 215, 0), (x * cell + cell // 2, y * cell + cell // 2), cell // 6)
            stun = int(self.stun_timers.get(a, 0))
            if stun > 0:
                pygame.draw.circle(surf, (255, 100, 100), (x * cell + cell // 2, y * cell + cell // 2), cell // 2, 2)

        stats_x = grid_px_w
        self._draw_stats_overlay(surf, cell, stats_x=stats_x, stats_w=stats_w)

        pygame.display.flip()
        time.sleep(1.0 / self.metadata.get("render_fps", 10))
        return None


def run_match(env, render=False, seed=None):
    obs = env.reset(seed=seed)
    total_rewards = {a: 0.0 for a in env.agents}
    done = False
    steps = 0
    # Load policies dynamically from agent files
    policies = {}
    for a in env.agents:
        # Convert agent name to valid filename (replace spaces with underscores)
        module_name = a.replace(" ", "_") + ".py"
        try:
            module = __import__(module_name[:-3])  # Remove .py extension
            policies[a] = module.policy
        except ImportError:
            print(f"Error: Could not load policy for {a} from {module_name}")
            policies[a] = lambda obs, agent_id: random.randint(0, 4)  # Fallback to random
    while True:
        actions = {}
        for a in env.agents:
            actions[a] = int(policies[a](obs[a], a))
        obs, rewards, dones, infos = env.step(actions)
        for a, r in rewards.items():
            total_rewards[a] += r
        steps += 1
        if render:
            env.render()
        if all(dones.values()):
            break
        if steps > env.max_steps + 5:
            break
    return total_rewards


if __name__ == "__main__":
    env = make_env()
    totals = run_match(env, render=True, seed=5472)
    team_scores = {0: 0.0, 1: 0.0}
    for a, r in totals.items():
        team_scores[env.team[a]] += r
    print("Per-agent totals:", {a: round(r, 2) for a, r in totals.items()})  # Uses team name + Agent i
    print("Per-team totals:", {env.team_names[t]: round(r, 2) for t, r in team_scores.items()})
