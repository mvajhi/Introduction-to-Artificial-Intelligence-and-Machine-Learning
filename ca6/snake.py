from cube import Cube
from constants import *
from utility import *

import random
import random
import numpy as np


class Snake:
    body = []
    turns = {}

    def __init__(self, color, pos, file_name=None):
        # pos is given as coordinates on the grid ex (1,5)
        self.color = color
        self.head = Cube(pos, color=color)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
        try:
            self.q_table = np.load(file_name)
        except:
            self.q_table = np.zeros((2,2,))

        self.lr = 0.5
        self.discount_factor = 0.7
        self.epsilon = 0.2

    def get_optimal_policy(self, state):
        # TODO: Get optimal policy
        pass

    def make_action(self, state):
        chance = random.random()
        if chance < self.epsilon:
            action = random.randint(0, 3)
        else:
            action = self.get_optimal_policy(state)
        return action

    def update_q_table(self, state, action, next_state, reward):
        # TODO: Update Q-table
        pass
    
    def create_state(self, snack, other_snake):
        snake_distance, pos = self.calc_snake_distance(other_snake)
        snake_side = self.calc_snake_side(pos)
        snack_distance = self.calc_snack_distance(snack)
        snack_side = self.calc_snack_side(snack)
        return {
            "snake_distance": snake_distance,
            "snake_side": snake_side,
            "snack_distance": snack_distance,
            "snack_side": snack_side
        }
        
    def calc_snack_distance(self, snack):
        # calc manhattan distance between snake head and snack
        return abs(snack.pos[0] - self.head.pos[0]) + abs(snack.pos[1] - self.head.pos[1])
        
    def calc_snack_side(self, snack):
        # calc snack side in relation to the snake
        if snack.pos[0] < self.head.pos[0]:
            return 0
        if snack.pos[0] > self.head.pos[0]:
            return 1
        if snack.pos[1] < self.head.pos[1]:
            return 2
        if snack.pos[1] > self.head.pos[1]:
            return 3
        return -1
        
    def calc_snake_side(self, pos):
        # calc snake side in relation to the snake
        if pos[0] < self.head.pos[0]:
            return 0
        if pos[0] >= self.head.pos[0]:
            return 1
        if pos[1] < self.head.pos[1]:
            return 2
        if pos[1] > self.head.pos[1]:
            return 3
        return -1

    def calc_snake_distance(self, other_snake):
        # calc manhattan distance between two snakes head and nerest cube
        nearest_cube = other_snake.body[0]
        nearest_val = abs(nearest_cube.pos[0] - self.head.pos[0]) + abs(nearest_cube.pos[1] - self.head.pos[1])
        for cube in other_snake.body:
            val = abs(cube.pos[0] - self.head.pos[0]) + abs(cube.pos[1] - self.head.pos[1])
            if val < nearest_val:
                nearest_val = val
                nearest_cube = cube
        return nearest_val, nearest_cube
        
        

    def move(self, snack, other_snake):
        state = self.create_state(snack, other_snake)
        action = self.make_action(state)

        if action == 0: # Left
            self.dirnx = -1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 1: # Right
            self.dirnx = 1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 2: # Up
            self.dirny = -1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 3: # Down
            self.dirny = 1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.dirnx, c.dirny)

        new_state = self.create_state(snack, other_snake)
        return state, new_state, action
    
    def check_out_of_board(self):
        headPos = self.head.pos
        if headPos[0] >= ROWS - 1 or headPos[0] < 1 or headPos[1] >= ROWS - 1 or headPos[1] < 1:
            self.reset((random.randint(3, 18), random.randint(3, 18)))
            return True
        return False
    
    def calc_kill_reward(self, other_snake):
        return KILL_REWARD + len(other_snake.body) * KILL_REWARD
    
    def calc_reward(self, snack, other_snake):
        reward = 0
        win_self, win_other = False, False
        
        if self.check_out_of_board():
            win_other = True
            reward -= self.calc_kill_reward(other_snake)
            reset(self, other_snake)
        
        if self.head.pos == snack.pos:
            self.addCube()
            snack = Cube(randomSnack(ROWS, self), color=(0, 255, 0))
            reward += SNACK_REWARD
            
        if self.head.pos in list(map(lambda z: z.pos, self.body[1:])):
            win_other = True
            reset(self, other_snake)
            reward -= self.calc_kill_reward(other_snake)
            
            
        if self.head.pos in list(map(lambda z: z.pos, other_snake.body)):
            if self.head.pos != other_snake.head.pos:
                win_other = True
                reward -= self.calc_kill_reward(other_snake)
            else:
                if len(self.body) > len(other_snake.body):
                    win_self = True
                    reward += self.calc_kill_reward(other_snake)
                elif len(self.body) == len(other_snake.body):
                    reward += 0
                else:
                    win_other = True
                    reward -= self.calc_kill_reward(other_snake)
                    
            reset(self, other_snake)
            
        return snack, reward, win_self, win_other
    
    def reset(self, pos):
        self.head = Cube(pos, color=self.color)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1

    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0] - 1, tail.pos[1]), color=self.color))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0] + 1, tail.pos[1]), color=self.color))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] - 1), color=self.color))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] + 1), color=self.color))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy

    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)

    def save_q_table(self, file_name):
        np.save(file_name, self.q_table)
        