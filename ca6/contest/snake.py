from cube import Cube
from constants import *
from utility import *

import random
import numpy as np
import pickle 
import matplotlib.pyplot as plt
import copy


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, input_shape, action_size):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * input_shape[1] * input_shape[2], 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    
    def save_weights(model, file_name):
        torch.save(model.state_dict(), file_name)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward):
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQNAgent:
    def __init__(self, file_name):
        self.input_shape = (1,20,20)
        self.action_size = 4
        self.batch_size = BATCH_SIZE
        self.memory = ReplayMemory(10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = QNetwork(self.input_shape, self.action_size)
        if file_name is not None:
            try:
                self.model.load_state_dict(torch.load(file_name))
                print(f"Loaded model weights from '{file_name}'.")
            except FileNotFoundError:
                print(f"File '{file_name}' not found. Creating a new model.")
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def remember(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = self.memory.sample(batch_size)
        for state, action, next_state, reward in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if next_state is not None:
                target = reward + self.gamma * torch.max(self.model(next_state)[0]).item()
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, file_name):
        self.model.save_weights(file_name)

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

        self.agent = DQNAgent(file_name)

    def make_action(self, state):
        return self.agent.act(state)

    def update_q_table(self, state, action, next_state, reward):
        self.agent.remember(state, action, next_state, reward)
    
    def create_state(self, snack, other_snake):
        board = [[BLANK for j in range(ROWS)] for i in range(ROWS)]
        for i in range(ROWS):
            for j in range(ROWS):
                if i < 1 or i >= ROWS - 1 or j < 1 or j >= ROWS - 1:
                    board[i][j] = WALL
                elif (i, j) == self.head.pos:
                    board[i][j] = MY_HEAD
                elif (i, j) == other_snake.head.pos:
                    board[i][j] = OTHER_HEAD
                elif (i, j) in list(map(lambda z: z.pos, self.body)):
                    board[i][j] = MY_BODY
                elif (i, j) in list(map(lambda z: z.pos, other_snake.body)):
                    board[i][j] = OTHER_BODY
                elif (i, j) == snack.pos:
                    board[i][j] = FOOD
        return board
                    
        
    def move(self, snack, other_snake):
        self.pre_head = copy.deepcopy(self.head)
        state = self.create_state(snack, other_snake)
        action = self.agent.act(state)

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
    
    def calc_snack_reward(self, snack):
        dist1 = abs(snack.pos[0] - self.pre_head.pos[0]) + abs(snack.pos[1] - self.pre_head.pos[1])
        dist2 = abs(snack.pos[0] - self.head.pos[0]) + abs(snack.pos[1] - self.head.pos[1])
        if dist2 - dist1 < 0:
            return SNACK_RATE
        else:
            return -SNACK_RATE * 1.5
    
    
    def calc_reward(self, snack, other_snake):
        reward = 0
        win_self, win_other = False, False
        
        reward += self.calc_snack_reward(snack)
        
        if self.check_out_of_board():
            win_other = True
            reward += LOSE_REWARD
            reset(self, other_snake, win_other)
        
        if self.head.pos == snack.pos:
            self.addCube()
            snack = Cube(randomSnack(ROWS, self), color=(0, 255, 0))
            reward += EAT_REWARD
            
            
        if self.head.pos in list(map(lambda z: z.pos, self.body[1:])):
            win_other = True
            reward += LOSE_REWARD
            reset(self, other_snake, win_other)
            
            
        if self.head.pos in list(map(lambda z: z.pos, other_snake.body)):
            
            if self.head.pos != other_snake.head.pos:
                reward += LOSE_REWARD
                win_other = True
            else:
                if len(self.body) > len(other_snake.body):
                    reward += WIN_REWARD
                    win_self = True
                elif len(self.body) == len(other_snake.body):
                    pass
                else:
                    reward += LOSE_REWARD
                    win_other = True
                    
            reset(self, other_snake, win_other)
        return snack, reward, win_self, win_other

    def reset(self, pos):
        self.head = Cube(pos, color=self.color)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1
        
        if len(self.agent.memory) > self.agent.batch_size:
            self.agent.replay(self.agent.batch_size)
            

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
        self.agent.save_model(file_name)
        