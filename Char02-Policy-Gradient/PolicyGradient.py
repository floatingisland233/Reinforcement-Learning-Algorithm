
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
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', default='False', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

# a = [1,2,3,4]
# # print(type(a))
# # print(a[::-1]) #list[<start>:<stop>:<step>]
# # print(a[0])
# a.insert(0,10)
# print(a)
# a.insert(3,10)
# print(a)
#
# languages = ['Java', 'Python', 'JavaScript']
# versions = [14, 3, 6]
#
# result = zip(languages, versions)
# print(list(result))
# for x in count(1):
#     print(x)
# exit()

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()
# print(eps)
# exit()

def select_action(state):
    # print("test1",type(state)) # <class 'numpy.ndarray'>
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    # print(probs.shape) # torch.Size([1, 2])
    m = Categorical(probs)

    # print("test100",probs.shape) # torch.Size([1, 2])

    action = m.sample()
    # print(action.shape)  # torch.Size([1])

    policy.saved_log_probs.append(m.log_prob(action)) # calculate the log probability
    # print("testkk",m,action)
    # print(policy.saved_log_probs) # e.g., [tensor([-0.6917], grad_fn=<SqueezeBackward1>) .....]
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

    # print("test3",rewards.shape,type(rewards)) # tensor
    # print("test30", len(policy.saved_log_probs), type(policy.saved_log_probs)) # list

    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        # print("test4",type(log_prob),type(reward))
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()

    # print("test2",policy_loss)
    # print("test21", torch.cat(policy_loss))

    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

    # print("test5",policy.rewards)


def main():
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            # if args.render:
            #     env.render()
            policy.rewards.append(reward)
            if done:
                break
        finish_episode()
        # if i_episode ==1: break
        running_reward = running_reward * 0.99 + t * 0.01  # t is the upright duration of the pole
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t+1, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
