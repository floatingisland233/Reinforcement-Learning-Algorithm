# this version works well, which can be used for teaching
# compare with PPO-CartPole_run.py, this file consider static advantage for updating actor
# see lines 110-114, 130-131

import argparse
import pickle
from collections import namedtuple
from itertools import count

import os, time
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter

# env = gym.make('CartPole-v0').unwrapped
env = gym.make('CartPole-v0')
num_state = env.observation_space.shape[0]
num_action = env.action_space.n

seed = 1
torch.manual_seed(seed)
env.seed(seed)
Transition = namedtuple('Transition', ['state', 'action', 'a_prob', 'reward', 'next_state'])


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.action_head = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.state_value = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value


class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 2 # each "update()" corresponds to 2 updates
    buffer_capacity = 1000
    batch_size = 32

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor()
        self.critic_net = Critic()
        self.buffer = []

        self.gamma = 0.99
        self.lr = 1e-3

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), self.lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), self.lr)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item()


    def store_transition(self, transition):
        self.buffer.append(transition)


    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state (because update is done for each episode)
        old_action_prob = torch.tensor([t.a_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + self.gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)

        V = self.critic_net(state)
        delta = Gt.view(-1, 1) - V
        advantage = delta.detach()

        # print("The agent is updating....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):

                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index]).gather(1, action[index])  # new policy

                ratio = (action_prob / old_action_prob[index])
                surr1 = ratio * advantage[index] # adv is static
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage[index]

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt[index].view(-1, 1), self.critic_net(state[index])) #Gt are static targets
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()


        del self.buffer[:]  # clear experience


def main():
    agent = PPO()
    trainReward = []
    render = False
    for i_epoch in range(500):

        if i_epoch > 490:
            render = True

        state = env.reset()
        if render: env.render()
        episode_reward = 0

        for t in count():
            action, action_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            trans = Transition(state, action, action_prob, reward, next_state)

            if render: env.render()
            agent.store_transition(trans)

            state = next_state
            episode_reward += reward

            if done:
                agent.update(i_epoch)
                print("epoch",i_epoch,"step",t+1)
                trainReward.append(episode_reward)
                break

    plt.plot(trainReward)
    plt.show()

if __name__ == '__main__':
    main()