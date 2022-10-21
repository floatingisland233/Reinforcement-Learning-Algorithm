# this version is used for testing
# see AC_Pendulum-run_debug1.py, which is a typical actor critic approach

import gym, os
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

#Parameters
env = gym.make('Pendulum-v1')
# env = env.unwrapped
# env.seed(1)
# torch.manual_seed(1)

num_state= env.observation_space.shape[0]
num_action = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
print("num_state", num_state,"num_action", num_action)
print("max_action", max_action)

#Hyperparameters
gamma = 0.99
num_episodes = 1000
render = True
eps = np.finfo(np.float32).eps.item()
log_prob_value = namedtuple('log_prob_value', ['log_prob', 'state_value'])

class ActorNet(nn.Module):

    def __init__(self):
        super(ActorNet, self).__init__()
        self.fc = nn.Linear(num_state, 100)
        self.mu_head = nn.Linear(100, num_action)
        self.sigma_head = nn.Linear(100, num_action)
        self.episode_logProb = []


    def forward(self, x):
        a_x = F.relu(self.fc(x))
        mu = max_action * F.tanh(self.mu_head(a_x))
        sigma = F.softplus(self.sigma_head(a_x))

        return mu, sigma

class CriticNet(nn.Module):

    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.fc2 = nn.Linear(100, 1)
        self.episode_value = []
        self.rewards = []

    def forward(self, x):
        state_value = self.fc2(F.relu(self.fc1(x)))
        return state_value # V(s)

actor_net = ActorNet()
critic_net = CriticNet()
optimizer_a = optim.Adam(actor_net.parameters(), lr=1e-4)
optimizer_c = optim.Adam(critic_net.parameters(), lr=1e-4)

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    mu, sigma = actor_net(state)
    state_value = critic_net(state)

    dist = Normal(mu, sigma)
    action = dist.sample()

    actor_net.episode_logProb.append(dist.log_prob(action))
    critic_net.episode_value.append(state_value)

    action = action.clamp(-max_action, max_action)
    return action.item()


def update(next_value):

    R = next_value
    returns = []

    for reward in critic_net.rewards[::-1]:   # value function at the terminal state is 0
        R = reward + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns).view(-1, 1)
    # returns = (returns - returns.mean()) / (returns.std() + eps)
    log_prob = torch.cat(actor_net.episode_logProb)
    state_value = torch.cat(critic_net.episode_value)
    r = returns.detach()

    for index in BatchSampler(SubsetRandomSampler(range(len(critic_net.rewards))), 200, False):
        # print(len(index))
        index =[i for i in range(len(critic_net.rewards))]
        # print(index)
        advantage = (r[index] - state_value[index]).detach() #MC advantage estimate，if only states in a window are used, it needs n-step advantage estimation
        action_loss = (-log_prob[index] * advantage).mean()
        optimizer_a.zero_grad()
        action_loss.backward()
        optimizer_a.step()

        critic_loss = F.smooth_l1_loss(state_value[index], r[index]).mean()
        optimizer_c.zero_grad()
        critic_loss.backward()
        optimizer_c.step()

        break

    del critic_net.rewards[:]
    del critic_net.episode_value[:]
    del actor_net.episode_logProb[:]


def main():
    render = False
    trainReward = []

    for i_episode in range(num_episodes):
        # if i_episode >= num_episodes-5:
        #     render = True

        state = env.reset()
        done = False
        episode_reward = 0
        step = 0
        while not done:
            action = select_action(state)
            next_state, reward, done, _ = env.step([action])

            episode_reward += reward  # duration of the upright pole
            critic_net.rewards.append(reward)

            if render: env.render()
            state = next_state

            step += 1
            if step % 200 == 0 or done:
                if not done:
                    next_state = torch.from_numpy(next_state).float().unsqueeze(0)
                    next_value = critic_net(next_state)
                else:
                    next_value = 0
                update(next_value)
                # print(f'update: eps:{i_episode}, step:{step}')

            if done:
                print(f'Episode rewards: {episode_reward} after episodes = {i_episode}')
                trainReward.append(episode_reward)
                # update(0)

    plt.plot(trainReward)
    plt.show()

if __name__ == '__main__':
    main()
