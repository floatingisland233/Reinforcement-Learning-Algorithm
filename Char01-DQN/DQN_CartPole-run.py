# this version works well, which can be used for teaching
# students should try build NN with different layers or nodes
# try different hyperparameters and define a better reward
import argparse
import pickle
from collections import namedtuple, deque
from itertools import count

import os, time
import numpy as np
import matplotlib.pyplot as plt
import random
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter

from custom_cartpole_env import CartPoleEnv

# Hyper-parameters
# seed = 1
render = True
# render = False
num_episodes = 1200
max_exploration_rate = 1
min_exploration_rate = 0.01
exp_decay = 1e-3

# env = gym.make('CartPole-v0').unwrapped
env = gym.make('CartPole-v0')
num_state = env.observation_space.shape[0]
num_action = env.action_space.n
print("num_state",num_state,"num_action",num_action)
# torch.manual_seed(seed)
# env.seed(seed)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.fc1 = nn.Linear(num_state, 24)
        # self.fc2 = nn.Linear(24, 12)
        # self.fc3 = nn.Linear(12, num_action)

        self.fc1 = nn.Linear(num_state, 256)
        self.fc2 = nn.Linear(256, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_value = self.fc2(x)
        return action_value

class DQN():
    # capacity = 100000
    capacity = 10000
    # capacity = 1000
    learning_rate = 1e-3
    memory_count = 0
    batch_size = 64
    gamma = 0.99

    def __init__(self):
        super(DQN, self).__init__()
        self.target_net, self.act_net = Net(), Net()
        self.target_net.load_state_dict(self.act_net.state_dict())
        self.optimizer = optim.Adam(self.act_net.parameters(), self.learning_rate)
        self.loss_func = nn.MSELoss()
        self.eps = 1
        self.memory = deque(maxlen=self.capacity)

    def select_action(self, state):
        if np.random.rand() <= self.eps:
            # action = env.action_space.sample()
            action = np.random.choice(range(num_action), 1).item()
            return action
        else:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            value = self.act_net(state)
            _, action = torch.max(value, 1)
            return action.item()

    def store_transition(self, state, action, reward, next_state, done):
        transition = Transition(state, action, reward, next_state, done)
        self.memory_count += 1
        self.memory.append(transition)

    def update(self):
        if self.memory_count >= self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            state = torch.tensor([t.state for t in batch]).float()
            action = torch.tensor([t.action for t in batch]).view(-1, 1).long()
            reward = torch.tensor([t.reward for t in batch]).float()
            next_state = torch.tensor([t.next_state for t in batch]).float()
            done = torch.tensor([t.done for t in batch]).float()

            with torch.no_grad():
                target_v = reward + (1-done) * self.gamma * self.target_net(next_state).max(dim=1)[0]

            loss = self.loss_func(self.act_net(state).gather(1, action), target_v.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            print("Memory Buff is too less")

def main():
    render = False
    agent = DQN()
    step = 0
    trainReward = []

    for i_ep in range(num_episodes):
        if i_ep >= num_episodes-5:
            render = True

        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward    # duration of the upright pole

            # reward = -100 * (abs(next_state[2]) - abs(state[2])) # use a better reward function

            if render: env.render()
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            step = step +1
            if step % 4 == 0 or done:
                agent.update()

            if done:
                print(
                    f'Episode rewards: {episode_reward} after episodes = {i_ep} eps: {agent.eps}')
                trainReward.append(episode_reward)
                if step >= 100:
                    print('Copying main network weights to the target network weights')
                    agent.target_net.load_state_dict(agent.act_net.state_dict())
                    step = 0

        agent.eps = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exp_decay * i_ep)

    plt.plot(trainReward)
    plt.show()

if __name__ == '__main__':
    main()

