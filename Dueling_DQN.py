import gym
import torch as th
import numpy as np
from env_single import env1
from rule_base import *
import wandb
wandb.init(project="my-project", entity="ZhangLin")

batch_size = 256
lr = 0.001
episilon = 0.9
replay_memory_size = 2000
gamma = 0.9
target_update_iter = 100
env = env1()
device = 'cuda:0' if th.cuda.is_available() else 'cpu'
n_state = 291
hidden1 = 256
hidden2 = 256

class net(th.nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.fc1 = th.nn.Linear(n_state, hidden1)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = th.nn.Linear(hidden1, hidden2)
        self.fc2.weight.data.normal_(0, 0.1)
        self.V = th.nn.Linear(hidden2, 1)
        self.V.weight.data.normal_(0, 0.1)
        self.A = th.nn.Linear(hidden2, 9)#n_action
        self.A.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = th.nn.functional.relu(x)
        x = self.fc2(x)
        x = th.nn.functional.relu(x)
        v = self.V(x)
        a = self.A(x)
        if len(a.shape) == 1:
            a -= th.mean(a, dim=0)
        else:  # for batch
            a -= th.mean(a, dim=-1).unsqueeze(1).repeat(1, 5)
        if len(a.shape) == 1:
            out = a + v.squeeze()
        else:  # for batch
            out = a + v.repeat(1, 5)
        return out


class replay_memory():
    def __init__(self):
        self.memory_size = replay_memory_size
        self.memory = np.array([])
        self.cur = 0
        self.new = 0

    def size(self):
        return self.memory.shape[0]

    # [s,a,r,s_,done]
    def store_transition(self, trans):
        if (self.memory.shape[0] < self.memory_size):
            if self.new == 0:
                self.memory = np.array(trans)
                self.new = 1
            elif self.memory.shape[0] > 0:
                self.memory = np.vstack((self.memory, trans))

        else:
            self.memory[self.cur, :] = trans
            self.cur = (self.cur + 1) % self.memory_size

    def sample(self):
        if self.memory.shape[0] < batch_size:
            return -1
        sam = np.random.choice(self.memory.shape[0], batch_size)
        return self.memory[sam]


class  Dueling_DQN(object):
    def __init__(self):
        self.eval_q_net, self.target_q_net = net().to(device), net().to(device)
        self.replay_mem = replay_memory()
        self.iter_num = 0
        self.optimizer = th.optim.Adam(self.eval_q_net.parameters(), lr=lr)
        self.loss = th.nn.MSELoss().to(device)

    def choose_action(self, qs):
        if np.random.uniform() < episilon:
            return th.argmax(qs).tolist()
        else:
            return np.random.randint(0, 9)#n_action

    def greedy_action(self, qs):
        return th.argmax(qs)

    def learn(self):
        if (self.iter_num % target_update_iter == 0):
            self.target_q_net.load_state_dict(self.eval_q_net.state_dict())
        self.iter_num += 1

        batch = self.replay_mem.sample()
        b_s = th.FloatTensor(batch[:, 0].tolist()).to(device)
        b_a = th.LongTensor(batch[:, 1].astype(int).tolist()).to(device)
        b_r = th.FloatTensor(batch[:, 2].tolist()).to(device)
        b_s_ = th.FloatTensor(batch[:, 3].tolist()).to(device)
        b_d = th.FloatTensor(batch[:, 4].tolist()).to(device)
        q_target = th.zeros((batch_size, 1)).to(device)
        q_eval = self.eval_q_net(b_s)
        q_eval = th.gather(q_eval, dim=1, index=th.unsqueeze(b_a, 1))
        q_next = self.target_q_net(b_s_).detach()
        for i in range(b_d.shape[0]):
            if (int(b_d[i].tolist()[0]) == 0):
                q_target[i] = b_r[i] + gamma * th.unsqueeze(th.max(q_next[i], 0)[0], 0)
            else:
                q_target[i] = b_r[i]
        td_error = self.loss(q_eval, q_target)

        self.optimizer.zero_grad()
        td_error.backward()
        self.optimizer.step()

import argparse
dqn = Dueling_DQN()
parser = argparse.ArgumentParser()
parser.add_argument('--iteration', default=20000, type=int) #  num of  games
parser.add_argument('--action_list_scheduling',type=list,default=rule_base)
parser.add_argument('--log_interval', default=40, type=int)
args = parser.parse_args()

running_reward = 0
go_reward = 0
go_loadb=0
running_ocu = 0
running_Qv = 0
running_loadb = 0
avg_length = 0
timestep = 0
tr = 0


for i_episode in range(args.iteration):
    env.reset()
    s=env.observations()
    t = 0
    r = 0.0
    while (t < 30000):
        t += 1
        qs = dqn.eval_q_net(th.FloatTensor(s).to(device))
        a = dqn.choose_action(qs)
        r, ocu_list, Qv, load_b=env.step(args.action_list_scheduling[a],t)
        s_=env.observations()
        done = False if env.is_running() == False else True
        transition = [s, a, [r], s_, [done]]
        dqn.replay_mem.store_transition(transition)
        s = s_

        go_reward += r
        go_loadb += load_b
        running_reward += r
        running_Qv += Qv

        if dqn.replay_mem.size() > batch_size:
            dqn.learn()
        if done:
            break
    ocu = sum(ocu_list) / len(ocu_list)
    running_ocu += ocu
    running_loadb += go_loadb / t


    wandb.log({
        "go_Qv":Qv,
        "go_reward": go_reward,
        "go_loadb": go_loadb/t,
        "ocu": ocu,
        "i_episode": i_episode}, step=i_episode)

    go_reward = 0
    go_loadb=0

    avg_length += t  # 累计步长
    if i_episode % args.log_interval == 0:

        running_ocu = (running_ocu / args.log_interval)  # 这么多局的平均占有率
        avg_length = int(avg_length / args.log_interval)  # 计算H.log_interval这么多局数的 平均每局的步长
        running_reward = int((running_reward / args.log_interval))  # 计算H.log_interval这么多局数的 平均每局的总回报

        running_Qv = running_Qv / args.log_interval
        running_loadb = running_loadb / args.log_interval
        wandb.log({
            "running_loadb": running_loadb,
            "running_reward": running_reward,
            "running_ocu": running_ocu,
            "running_Qv":running_Qv
        })
        running_reward = 0
        running_ocu = 0
        running_Qv = 0

        running_loadb = 0

wandb.save('modelDueling_DQN.h5')
wandb.finish()