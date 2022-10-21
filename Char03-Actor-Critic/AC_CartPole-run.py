# this version works well, which can be used for teaching
# students should try build NN with different layers or nodes
# try different hyperparameters and define a better reward
# try to define two separate classes for actor and critic

# this is actor critic method,
# where the MC advantage estimate is used to update the actor
# MC advantage estimate is the return (until the terminal state) minus V(s).
# note that actor or critic update is carried out for each episode

import gym, os
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

#Parameters
env = gym.make('CartPole-v0')
# env = env.unwrapped
# env.seed(1)
# torch.manual_seed(1)

num_state= env.observation_space.shape[0]
num_action = env.action_space.n
print("num_state",num_state,"num_action",num_action)

#Hyperparameters
learning_rate = 0.01
gamma = 0.99
num_episodes = 400
render = True
eps = np.finfo(np.float32).eps.item()
log_prob_value = namedtuple('log_prob_value', ['log_prob', 'state_value'])

class Actor_critic(nn.Module):
    def __init__(self):
        super(Actor_critic, self).__init__()

        self.fc1 = nn.Linear(num_state, 64)
        self.fc2 = nn.Linear(64, num_action)
        self.fc3 = nn.Linear(num_state, 32)
        self.fc4 = nn.Linear(32, 1)

        self.episode_logProb_value = []
        self.rewards = []

    def forward(self, x):
        action_score = self.fc2(F.relu(self.fc1(x)))
        state_value = self.fc4(F.relu(self.fc3(x)))

        return F.softmax(action_score, dim=-1), state_value # predict \pi(a|s) and V(s)

agent = Actor_critic()
optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs, state_value = agent(state)

    m = Categorical(probs)
    action = m.sample()

    agent.episode_logProb_value.append(log_prob_value(m.log_prob(action), state_value))
    return action.item()


def update():
    R = 0
    returns = []
    actor_loss = []
    critic_loss = []

    for reward in agent.rewards[::-1]:
        R = reward + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    # returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, state_value), r in zip(agent.episode_logProb_value, returns):
        advantage = r - state_value.item() #MC advantage estimate，if only states in a window are used, it needs n-step advantage estimation
        actor_loss.append(-log_prob * advantage)
        critic_loss.append(F.smooth_l1_loss(torch.squeeze(state_value), torch.tensor([r])))  # r is a return until terminal

    loss = torch.stack(actor_loss).sum() + torch.stack(critic_loss).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    del agent.rewards[:]
    del agent.episode_logProb_value[:]

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
            state, reward, done, _ = env.step(action)
            episode_reward += reward  # duration of the upright pole
            agent.rewards.append(reward)

            if render: env.render()

            if done:
                print(f'Episode rewards: {episode_reward} after episodes = {i_episode}')
                trainReward.append(episode_reward)
                update()

    plt.plot(trainReward)
    plt.show()

if __name__ == '__main__':
    main()
