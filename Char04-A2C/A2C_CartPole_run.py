# this version works well, which can be used for teaching
# students should try build NN with different layers or nodes
# try different hyperparameters and define a better reward

# this is advantage actor critic (A2C) method,
# where the n-step advantage estimate is used to update the actor
# note that actor or critic update is carried out with in a window of n steps
# note that entropy regularisation is used.

import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

env = gym.make('CartPole-v0')
num_state= env.observation_space.shape[0]
num_action = env.action_space.n
print("num_state",num_state,"num_action",num_action)

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_state, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_state, 128),
            nn.ReLU(),
            nn.Linear(128, num_action),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs) # distribution of all actions
        return dist, value

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []

    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)

    return returns

#Hyper params:
learning_rate = 1e-3

model = ActorCritic(num_state, num_action)#.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

i_episode = 0
episode_reward = 0
trainReward = []

frame_idx    = 0
max_frames   = 30000 # each frame corresponds to one transition
num_steps   = 5
state = env.reset()
while frame_idx < max_frames:

    log_probs = []
    values = []
    rewards = []
    masks = []
    entropy = []

    # rollout trajectory
    for _ in range(num_steps):
        state = torch.from_numpy(state).float().unsqueeze(0)
        dist, value = model(state)

        action = dist.sample()
        next_state, reward, done, _ = env.step(action.item())

        log_prob = dist.log_prob(action) # log probability
        log_probs.append(log_prob)
        values.append(value) # state value
        rewards.append(reward) # immediate reward
        entropy.append(dist.entropy())  # entropy regularisation, H[\pi(.|s)]
        masks.append(1 - done) # judge if the terminal has arrived
        
        state = next_state
        frame_idx += 1
        episode_reward += reward

        if done:
            state = env.reset()
            i_episode += 1
            print(f'reach terminal at: {frame_idx}, Episode rewards: {episode_reward} after episodes = {i_episode}')
            trainReward.append(episode_reward)
            episode_reward = 0
            break

    next_state = torch.from_numpy(next_state).float().unsqueeze(0)
    _, next_value = model(next_state)
    returns = compute_returns(next_value, rewards, masks) # used for n-step advantage estimate

    log_probs = torch.cat(log_probs) # torch.Size([num_steps])
    returns = torch.cat(returns).detach() # torch.Size([num_steps, 1]), returns should not be optimized, so use .detach()
    values = torch.cat(values) # torch.Size([num_steps, 1])
    entropy = torch.cat(entropy) # torch.Size([num_steps])

    advantage = returns - values

    actor_loss  = -(log_probs * advantage.detach()).mean() # optimize actor
    critic_loss = advantage.pow(2).mean() # optimize critic

    loss = actor_loss + 0.5 * critic_loss - 5e-3 * entropy.mean() # entropy regularisation

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(trainReward)
plt.show()
