import argparse
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import copy
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from module import Memory, VMPO, PPO,SAC
import numpy as np
import torch
from collections import deque, namedtuple
from env_single import env1
from rule_base import *
import wandb
wandb.init(project="my-project", entity="ZhangLin")


class ReplayBuffer:
    """用于存储经验元组的固定大小缓冲区。s."""
    def __init__(self, buffer_size, batch_size, device):
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """为memory增添新的ex。"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """从记memory中随机抽取一批经验"""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


def save(args, save_name, model, wandb, ep=None):
    import os
    save_dir = './trained_models/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
        wandb.save(save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")
        wandb.save(save_dir + args.run_name + save_name + ".pth")


def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="SAC", help="Run name, default: SAC")
    parser.add_argument("--env", type=str, default="CartPole-v0", help="Gym environment name, default: CartPole-v0")
    parser.add_argument("--episodes", type=int, default=20000, help="Number of episodes, default: 100")
    parser.add_argument("--buffer_size", type=int, default=2000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=1000, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")
    parser.add_argument("--log_interval", type=int, default=40)
    parser.add_argument("--update_timestep", type=int, default=100)
    parser.add_argument("--action_list_scheduling", type=list, default=rule_base)
    args = parser.parse_args()
    return args


def collect_random(env, dataset, num_samples=2100):
    env.reset()
    state=env.observations()
    config = get_config()
    for _ in range(num_samples):
        action = random.choice([i for i in range(len(config.action_list_scheduling))])
        reward, ocu_list, Qv, load_b  = env.step(config.action_list_scheduling[action],0)
        next_state = env.observations()
        if env.is_running() == False:  # ！！判断环境是否判定游戏结束，并记录
            done = False
        else:
            done = True
        assert state is not None
        dataset.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            env.reset()
            state=env.observations()

def train(config):

    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    env = env1()
    env.reset()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    steps = 0

    agent = SAC(state_size=len(env.observations()),
                     action_size=5,
                     device=device)

    buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device)

    collect_random(env=env, dataset=buffer, num_samples=555)#随机数据

    running_reward = 0
    go_reward=0
    go_loadb=0
    running_ocu = 0
    running_Qv = 0
    running_loadb=0
    timestep=0

    for i_episode in range(1, config.episodes+1):
        env.reset()
        for t in range(30000):
            img = env.observations()  # 输入状态
            action = agent.get_action(img)
            steps += 1
            timestep +=1
            reward, ocu_list, Qv, load_b = env.step(config.action_list_scheduling[action], t)
            next_state=env.observations()
            done = False if env.is_running()==False else True
            buffer.add(img, action, reward, next_state, done)
            if timestep>=config.update_timestep:
                policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha = agent.learn(steps, buffer.sample(), gamma=0.96)
                timestep=0


            running_reward += reward  #累计回报
            go_reward += reward
            go_loadb+=load_b
            running_Qv +=Qv
            running_loadb+=load_b/t

            if done or t == 29999:
                break

        ocu = sum(ocu_list) / len(ocu_list)
        running_ocu += ocu
        running_loadb += go_loadb / t

        wandb.log({
            "go_Qv": Qv,
            "go_reward": go_reward,
            "go_loadb": go_loadb/t,
            "ocu": ocu,
            "i_episode": i_episode}, step=i_episode)

        go_reward = 0
        go_loadb = 0

        # Logging
        if i_episode % config.log_interval == 0:
            running_ocu = (running_ocu / config.log_interval)  # 这么多局的平均占有率
            avg_length = int(avg_length / config.log_interval)  # 计算H.log_interval这么多局数的 平均每局的步长
            running_reward = int((running_reward / config.log_interval))  # 计算H.log_interval这么多局数的 平均每局的总回报
            running_Qv = running_Qv / config.log_interval
            running_loadb = running_loadb / config.log_interval
            wandb.log({
                "running_loadb": running_loadb,
                "running_reward": running_reward,
                "running_ocu": running_ocu,
                "running_Qv": running_Qv
            })
            running_reward = 0
            running_ocu = 0
            running_Qv = 0
            running_loadb = 0
    wandb.save('model.h5')
    wandb.finish()

if __name__ == "__main__":
    config = get_config()
    train(config)
