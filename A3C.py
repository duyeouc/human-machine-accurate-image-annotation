# -*- coding: utf-8 -*-
import numpy as np
import gym
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
import torch.multiprocessing as mp

GAME = 'CartPole-v0'
OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = 3
MAX_GLOBAL_EP = 3000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 100
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 0.001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
STEP = 3000  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode
GLOBAL_EP = 0

env = gym.make(GAME)
N_S = env.observation_space.shape[0]
N_A = env.action_space.n


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        shared_param._grad = param.grad


class Actor(torch.nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.actor_fc1 = nn.Linear(N_S, 200)
        self.actor_fc1.weight.data.normal_(0, 0.6)
        self.actor_fc2 = nn.Linear(200, N_A)
        self.actor_fc2.weight.data.normal_(0, 0.6)

    def forward(self, s):
        l_a = F.relu6(self.actor_fc1(s))
        a_prob = F.softmax(self.actor_fc2(l_a))
        return a_prob


class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.critic_fc1 = nn.Linear(N_S, 100)
        self.critic_fc1.weight.data.normal_(0, 0.6)
        self.critic_fc2 = nn.Linear(100, 1)
        self.critic_fc2.weight.data.normal_(0, 0.6)

    def forward(self, s):
        l_c = F.relu6(self.critic_fc1(s))
        v = self.critic_fc2(l_c)
        return v


class ACNet(torch.nn.Module):
    def __init__(self):
        super(ACNet, self).__init__()
        self.actor = Actor()
        self.critic = Critic()
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=0.001)
        self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(), lr=0.001)

    def return_td(self, s, v_target):
        self.v = self.critic(s)
        td = v_target - self.v
        return td.detach()

    def return_c_loss(self, s, v_target):
        self.v = self.critic(s)
        self.td = v_target - self.v
        self.c_loss = (self.td ** 2).mean()
        return self.c_loss

    def return_a_loss(self, s, td, a_his):
        self.a_prob = self.actor(s)
        a_his = a_his.unsqueeze(1)
        one_hot = torch.zeros(a_his.shape[0], N_A).scatter_(1, a_his, 1)
        log_prob = torch.sum(torch.log(self.a_prob + 1e-5) * one_hot, dim=1, keepdim=True)
        exp_v = log_prob * td
        entropy = -torch.sum(self.a_prob * torch.log(self.a_prob + 1e-5), dim=1, keepdim=True)

        self.exp_v = ENTROPY_BETA * entropy + exp_v
        self.a_loss = (-self.exp_v).mean()
        return self.a_loss

    def choose_action(self, s):  # run by a local
        prob_weights = self.actor(torch.FloatTensor(s[np.newaxis, :]))
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.detach().numpy().ravel())  # select action w.r.t the actions prob
        return action


def work(name, AC, lock):
    env = gym.make(GAME).unwrapped
    global GLOBAL_RUNNING_R, GLOBAL_EP
    total_step = 1
    buffer_s, buffer_a, buffer_r = [], [], []
    model = ACNet()
    actor_optimizer = AC.actor_optimizer
    critic_optimizer = AC.critic_optimizer
    while GLOBAL_EP < MAX_GLOBAL_EP:
        lock.acquire()
        model.load_state_dict(AC.state_dict())
        s = env.reset()
        ep_r = 0
        while True:

            # if self.name == 'W_0':
            #     self.env.render()
            a = model.choose_action(s)
            s_, r, done, info = env.step(a)
            if done: r = -5
            ep_r += r
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append(r)

            if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net

                if done:
                    v_s_ = 0  # terminal
                else:
                    # if not done, v(s)
                    v_s_ = model.critic(torch.FloatTensor(s_[np.newaxis, :]))[0, 0]
                buffer_v_target = []
                # create buffer_v_target
                for r in buffer_r[::-1]:  # reverse buffer r
                    v_s_ = r + GAMMA * v_s_
                    buffer_v_target.append(v_s_)
                buffer_v_target.reverse()
                buffer_s, buffer_a, buffer_v_target = torch.FloatTensor(buffer_s), torch.LongTensor(
                    buffer_a), torch.FloatTensor(buffer_v_target)

                td_error = model.return_td(buffer_s, buffer_v_target)
                c_loss = model.return_c_loss(buffer_s, buffer_v_target)
                critic_optimizer.zero_grad()
                c_loss.backward()
                ensure_shared_grads(model, AC)
                critic_optimizer.step()

                a_loss = model.return_a_loss(buffer_s, td_error, buffer_a)
                actor_optimizer.zero_grad()
                a_loss.backward()
                ensure_shared_grads(model, AC)
                actor_optimizer.step()

                buffer_s, buffer_a, buffer_r = [], [], []
            s = s_
            total_step += 1

            if done:
                if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                    GLOBAL_RUNNING_R.append(ep_r)
                else:
                    GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                #               print(
                #                        self.name,
                #                        "Ep:", GLOBAL_EP,
                #                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                #                          )
                print("name: %s" % name, "| Ep_r: %i" % GLOBAL_RUNNING_R[-1])
                print(name)
                GLOBAL_EP += 1
                break
        lock.release()


if __name__ == "__main__":
    GLOBAL_AC = ACNet()  # we only need its params
    GLOBAL_AC.share_memory()
    workers = []
    lock = mp.Lock()
    # Create worker
    for i in range(N_WORKERS):
        p = mp.Process(target=work, args=(i, GLOBAL_AC, lock,))
        p.start()
        workers.append(p)
    for p in workers:
        p.join()

    total_reward = 0
    for i in range(TEST):
        state = env.reset()
        for j in range(STEP):
            #            env.render()
            action = GLOBAL_AC.choose_action(state)  # direct action for test
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
    ave_reward = total_reward / TEST
    print('episode: ', GLOBAL_EP, 'Evaluation Average Reward:', ave_reward)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()

