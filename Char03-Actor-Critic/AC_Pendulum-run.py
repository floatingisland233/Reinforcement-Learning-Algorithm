# this version works well, which can be used for teaching
# this file is a simplification of PPO-Pendulum_run.py,
# compared with AC, this code contains importance sampling.
# If importance sampling is ignored, the update gradient is wrong
# see off-policy actor critic for the rationale of this file

import argparse
import pickle
from collections import namedtuple

import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

parser = argparse.ArgumentParser(description='Solve the Pendulum-v1 with PPO')
parser.add_argument(
    '--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.9)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')

args = parser.parse_args()

torch.manual_seed(args.seed)

TrainingRecord = namedtuple('TrainingRecord', ['episode', 'reward'])
Transition = namedtuple('Transition', ['s', 'a', 'a_log_p', 'r', 's_'])


class ActorNet(nn.Module):

    def __init__(self):
        super(ActorNet, self).__init__()
        self.fc = nn.Linear(3, 100)
        self.mu_head = nn.Linear(100, 1)
        self.sigma_head = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        mu = 2.0 * F.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))
        return (mu, sigma)


class CriticNet(nn.Module):

    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc = nn.Linear(3, 100)
        self.v_head = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        state_value = self.v_head(x)
        return state_value


class Agent():

    ppo_epoch = 5
    buffer_capacity= 800
    batch_size = 32

    def __init__(self):
        self.actor_net = ActorNet().float()
        self.critic_net = CriticNet().float()
        self.buffer = []
        self.counter = 0

        self.optimizer_a = optim.Adam(self.actor_net.parameters(), lr=1e-4)
        self.optimizer_c = optim.Adam(self.critic_net.parameters(), lr=1e-3)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            (mu, sigma) = self.actor_net(state)
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        action = action.clamp(-2.0, 2.0)
        return action.item(), action_log_prob.item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(), 'param/ppo_anet_params.pkl')
        torch.save(self.critic_net.state_dict(), 'param/ppo_cnet_params.pkl')

    def store(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def update(self):

        s = torch.tensor([t.s for t in self.buffer], dtype=torch.float)
        a = torch.tensor([t.a for t in self.buffer], dtype=torch.float).view(-1, 1)
        r = torch.tensor([t.r for t in self.buffer], dtype=torch.float).view(-1, 1)
        s_ = torch.tensor([t.s_ for t in self.buffer], dtype=torch.float)

        old_action_log_probs = torch.tensor(
            [t.a_log_p for t in self.buffer], dtype=torch.float).view(-1, 1)

        r = (r - r.mean()) / (r.std() + 1e-5)
        with torch.no_grad():
            target_v = r + args.gamma * self.critic_net(s_) # this leads to static targets for critic updates

        adv = (target_v - self.critic_net(s)).detach()

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(
                    SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                (mu, sigma) = self.actor_net(s[index])
                dist = Normal(mu, sigma)
                action_log_probs = dist.log_prob(a[index])

                ratio = torch.exp(action_log_probs - old_action_log_probs[index]) # this can be interpreted as importance sampling
                surr1 = ratio * adv[index] # here adv[index] correspond to adv under the old policy, which are static as they are determined before the update loop
                action_loss = - surr1.mean()

                # action_loss = - (action_log_probs * adv[index]).mean() # this can be interpreted as importance samplingwithout importance sampling

                self.optimizer_a.zero_grad()
                action_loss.backward()
                self.optimizer_a.step()

                value_loss = F.smooth_l1_loss(self.critic_net(s[index]), target_v[index]) # static targets
                self.optimizer_c.zero_grad()
                value_loss.backward()
                self.optimizer_c.step()

        del self.buffer[:]


def main():
    env = gym.make('Pendulum-v1')
    env.seed(args.seed)

    agent = Agent()
    trainReward = []
    render = False

    for i_ep in range(2000):
        if i_ep > 1990:
            render = True

        state = env.reset()
        episode_reward = 0

        for t in range(200):
            action, action_log_prob = agent.select_action(state)
            state_, reward, done, _ = env.step([action])

            if render: env.render()
            if agent.store(Transition(state, action, action_log_prob, reward, state_)):
                agent.update()

            state = state_
            episode_reward += reward

            if done:
                print("Episode: \t{}, Total step: \t{}, Total Reward: \t{:0.2f}".format(i_ep, t, episode_reward))
                trainReward.append(episode_reward)
                break

    plt.plot(trainReward)
    plt.show()

if __name__ == '__main__':
    main()
