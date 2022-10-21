# this version works well, which can be used for teaching
# when updating the policy, the return (until the terminal state) is used.

import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Categorical

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
args = parser.parse_args()

env = gym.make('CartPole-v0')
num_state = env.observation_space.shape[0]
num_action = env.action_space.n
print("num_state",num_state,"num_action",num_action)
num_episodes = 400
render = True

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_state, 256)
        self.affine2 = nn.Linear(256, num_action)

        self.episode_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
    probs = policy(state)
    action = np.random.choice(np.arange(num_action), p=probs[0].detach().numpy())
    policy.episode_log_probs.append(torch.log(probs[:, action]))

    return action.item()

def update_policy():
    R = 0
    returns = []

    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    log_probs = torch.cat(policy.episode_log_probs)

    # returns = (returns - returns.mean()) / (returns.std() + eps)
    policy_loss = torch.sum(-returns * log_probs, dim=0)
    # policy_loss = torch.mean(-returns * log_probs, dim=0) # the length of each episode is different, so sum() is better than mean()

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    del policy.rewards[:]
    del policy.episode_log_probs[:]


def main():
    render = False
    trainReward = []

    for i_episode in range(num_episodes):
        if i_episode >= num_episodes-5:
            render = True

        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward  # duration of the upright pole
            policy.rewards.append(reward)

            if render: env.render()

            if done:
                print(f'Episode rewards: {episode_reward} after episodes = {i_episode}')
                trainReward.append(episode_reward)
                update_policy()

    plt.plot(trainReward)
    plt.show()

if __name__ == '__main__':
    main()
