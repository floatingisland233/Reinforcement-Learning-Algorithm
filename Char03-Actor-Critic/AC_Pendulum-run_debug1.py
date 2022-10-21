# this version is used for testing
# this is actor critic method, but it does not work well for the pendulum with a continuous action
# this file is very similar to AC_CartPole-run.py

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
num_episodes = 4500
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
optimizer_a = optim.Adam(actor_net.parameters(), lr=2e-4)
optimizer_c = optim.Adam(critic_net.parameters(), lr=2e-4)

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


def update():

    R = 0
    returns = []

    actor_loss = []
    critic_loss = []

    for reward in critic_net.rewards[::-1]:   # value function at the terminal state is 0
        R = reward + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns).view(-1, 1)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for log_prob, state_value, r in zip(actor_net.episode_logProb, critic_net.episode_value, returns):
        advantage = r - state_value.item()  # MC advantage estimate，if only states in a window are used, it needs n-step advantage estimation
        actor_loss.append(-log_prob * advantage)
        critic_loss.append(F.smooth_l1_loss(torch.squeeze(state_value), torch.tensor([r])))  # r is a return until terminal

    action_loss = torch.stack(actor_loss).sum()
    optimizer_a.zero_grad()
    action_loss.backward()
    optimizer_a.step()

    critic_loss = torch.stack(critic_loss).sum()
    optimizer_c.zero_grad()
    critic_loss.backward()
    optimizer_c.step()


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

        while not done:
            action = select_action(state)
            next_state, reward, done, _ = env.step([action])

            episode_reward += reward  # duration of the upright pole
            critic_net.rewards.append(reward)

            if render: env.render()
            state = next_state

            if done:
                print(f'Episode rewards: {episode_reward} after episodes = {i_episode}')
                trainReward.append(episode_reward)
                update()

    plt.plot(trainReward)
    plt.show()

if __name__ == '__main__':
    main()
