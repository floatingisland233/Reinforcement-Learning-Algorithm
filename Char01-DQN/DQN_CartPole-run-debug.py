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
# num_episodes = 2000
num_episodes = 1000
max_exploration_rate = 1
min_exploration_rate = 0.01
exp_decay = 1e-3

# env = gym.make('CartPole-v0').unwrapped
env = CartPoleEnv()
num_state = env.observation_space.shape[0]
num_action = env.action_space.n
print("num_state",num_state,"num_action",num_action)
# torch.manual_seed(seed)
# env.seed(seed)

# exit()
# a = torch.tensor([[1,2],[3,4],[5,6]])
# print(a)
# print(a[0:2])
# b= torch.tensor([1,2,3])
# index=[1,2]
# print(a[index])
# print(a[1])
# print(b[index])
# print(b[1])
# index=torch.tensor([[0],[1],[0]])
# index=torch.tensor([0,1,0])
# print(index)
# print(a.gather(0,index))
# print(a.gather(1,index))
# print(a.gather(1,index)[2])
# print(a)
# print(a.max(dim=1))
# print(a.max(dim=1)[0])
# print(a.max(dim=1)[1])
# print(torch.max(a,1))
# print(list(BatchSampler(SubsetRandomSampler(range(10)), batch_size=3, drop_last=False)))
# print(np.random.rand(1))
# print(type(np.random.choice(range(5), 1).item()))
# print(range(5))
# exit()

# Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'terminal'))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_state, 24)
        self.fc2 = nn.Linear(24, 12)
        self.fc3 = nn.Linear(12, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_value = self.fc3(x)
        return action_value

class DQN():

    # capacity = 8000
    capacity = 100000
    learning_rate = 1e-3
    memory_count = 0
    # batch_size = 256
    batch_size = 64
    gamma = 0.99
    update_count = 0

    def __init__(self):
        super(DQN, self).__init__()
        self.target_net, self.act_net = Net(), Net()
        self.target_net.load_state_dict(self.act_net.state_dict())
        self.memory = []
        # self.memory = [None]*self.capacity
        # print("test2", [None]*10)
        self.optimizer = optim.Adam(self.act_net.parameters(), self.learning_rate)
        self.loss_func = nn.MSELoss()
        # self.writer = SummaryWriter('./DQN/logs')
        self.eps = 1
        self.memory = deque(maxlen=100000)


    def select_action(self,state,env):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        value = self.act_net(state)

        if np.random.rand() <= self.eps:
            action = env.action_space.sample()
            action = np.random.choice(range(num_action), 1).item()
            return action
        else:
             _, action = torch.max(value, 1)
             return action.item()
             # return action.cpu().numpy()[0]

        # state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        # # print("state shape",state.shape)
        # value = self.act_net(state)
        # action_max_value, index = torch.max(value, 1)
        # # print("test22",index,type(index))
        # # print("test23",index.item(),type(index.item()))
        # action = index.item()
        # # if np.random.rand(1) >= 0.6:  # epslion greedy
        # #     action = np.random.choice(range(num_action), 1).item()
        # if np.random.rand() <= self.eps: # epslion greedy
        #     # print("eps",self.eps)
        #     action = np.random.choice(range(num_action), 1).item()
        # return action

    def store_transition(self,transition):
        # index = self.memory_count % self.capacity
        # self.memory.append(transition)
        # # self.memory[index] = transition
        self.memory_count += 1
        # # return self.memory_count >= self.capacity
        self.memory.append(transition)

    def update(self):
        # if self.memory_count >= self.capacity:
        if self.memory_count >= self.batch_size:
            # print("test0", len(self.memory))
            # print("test",self.memory[0].state)
            # state = torch.tensor([t.state for t in self.memory[0:self.memory_count]]).float()
            # action = torch.LongTensor([t.action for t in self.memory[0:self.memory_count]]).view(-1,1).long()
            # reward = torch.tensor([t.reward for t in self.memory[0:self.memory_count]]).float()
            # next_state = torch.tensor([t.next_state for t in self.memory[0:self.memory_count]]).float()
            # terminal = torch.tensor([t.terminal for t in self.memory[0:self.memory_count]]).float()
            # print(torch.LongTensor([t.action for t in self.memory]).shape)
            # print(state.shape,action.shape,reward.shape,next_state.shape)

            # reward = (reward - reward.mean()) / (reward.std() + 1e-7)

            # with torch.no_grad():
            #     target_v = reward + terminal * self.gamma * self.target_net(next_state).max(dim=1)[0]

                # print("test0",reward.shape,self.target_net(next_state).shape,self.target_net(next_state).max(dim=1)[0].shape)
                # print("test1",self.target_net(next_state))
                # print("test11",self.target_net(next_state).max)

            #Update...
            # print(type(self.memory))
            batch = random.sample(self.memory,self.batch_size)
            state = torch.tensor([t.state for t in batch]).float()
            action = torch.tensor([t.action for t in batch]).view(-1, 1).long()
            reward = torch.tensor([t.reward for t in batch]).float()
            next_state = torch.tensor([t.next_state for t in batch]).float()
            terminal = torch.tensor([t.terminal for t in batch]).float()
            # print("test dim",state.shape,reward.shape)
            with torch.no_grad():
                target_v = reward + terminal * self.gamma * self.target_net(next_state).max(dim=1)[0]
            loss = self.loss_func((self.act_net(state).gather(1, action)), target_v.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))), batch_size=self.batch_size, drop_last=False):
            # for index in BatchSampler(SubsetRandomSampler(range(self.memory_count)), batch_size=self.batch_size, drop_last=False):
                # print("test1",len(self.memory),self.memory_count)
                # print("test2",index)
                # v = (self.act_net(state).gather(1, action))[index]
                # print("test2",self.act_net(state).shape)# [5,2]
                # print("test21", action.shape) # [5,1]
                # print("test3",self.act_net(state).gather(1, action).shape) # [5,1]
                # print("test30",index)
                # print("test31",self.act_net(state).gather(1, action)[index].shape) # [5,1]
                # print("test22",target_v.shape) # [5]
                # print("test221",target_v[index].shape) # [5]
                # print("test222",target_v[index].unsqueeze(1).shape) # [5,1]
                # loss = self.loss_func(target_v[index].unsqueeze(1), (self.act_net(state).gather(1, action))[index])
                # loss = self.loss_func((self.act_net(state).gather(1, action))[index], target_v[index].unsqueeze(1))
                # self.optimizer.zero_grad()
                # loss.backward()
                # self.optimizer.step()
                # self.writer.add_scalar('loss/value_loss', loss, self.update_count)
                # self.update_count +=1
                # if self.update_count % 25 ==0:
                #     self.target_net.load_state_dict(self.act_net.state_dict())
                # break
        else:
            print("Memory Buff is too less")
def main():
    render = False
    agent = DQN()
    totalReward = []
    sumReward = 0
    step = 0
    for i_ep in range(num_episodes):
        state = env.reset()

        # if i_ep >= 950:
        #     # print("i_ep {}".format(i_ep))
        #     render = True
        # if render: env.render()

        done = False
        total_rewards_per_episode = 0
        while not done:
            action = agent.select_action(state,env)
            next_state, reward, done, info = env.step(action)

            if render: env.render()
            # transition = Transition(state, action, reward, next_state)
            # print("test3:",transition)
            # print("test3:",transition.state)
            terminal = 0 if done else 1
            transition = Transition(state, action, reward, next_state, terminal)
            agent.store_transition(transition)

            state = next_state
            total_rewards_per_episode += reward
            # sumReward = sumReward + reward

            step = step +1
            if step % 4 == 0 or done:
                agent.update()
            # print(t,done)
            if done:
                print(
                    f'Total training rewards: {total_rewards_per_episode} after n episodes = {i_ep} with eps: {agent.eps}')
                totalReward.append(total_rewards_per_episode)
                if step >= 100:
                    print('Copying main network weights to the target network weights')
                    agent.target_net.load_state_dict(agent.act_net.state_dict())
                    step = 0

        # agent.writer.add_scalar('live/finish_step', t+1, global_step=i_ep)
        # agent.update()  # update for each rollout or episode
                # if i_ep % 200 == 0: agent.eps = agent.eps * 0.999

        agent.eps = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exp_decay * i_ep)
        # if i_ep > 100:
        #     agent.eps -= 0.0001
        #     agent.eps = max(agent.eps, 0.1)
        # break
        # if i_ep % 20 == 0:
        #     totalReward.append(sumReward)
        #     print("episodes {}, sumReward is {}, eps {} ".format(i_ep, sumReward/20.0, agent.eps))
        #     sumReward = 0
            # print("episodes {}, step is {}".format(i_ep, t))
            # print("episodes {}, step is {}, eps {} ".format(i_ep, t, agent.eps))

    plt.plot(totalReward)
    # print(totalReward)
    plt.show()

if __name__ == '__main__':
    main()

