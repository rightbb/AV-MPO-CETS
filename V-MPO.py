import os
import time
import torch
import numpy as np
from module import Memory,VMPO
from utils import set_up_hyperparams
from tensorboardX import SummaryWriter
import random
from env_single import env1
from rule_base import *
import wandb
wandb.init(project="my-project", entity="ZhangLin")

def main():
    H, logprint = set_up_hyperparams()

    env = env1()
    H.img_size = 64
    H.input_state_dim = 291
    H.device = H.gpu if H.gpu is not None else 'cpu'

    memory = Memory()  # 存储
    agent = VMPO(H)

    running_reward = 0
    go_reward = 0
    running_ocu = 0
    running_Qv = 0
    running_loadb = 0

    avg_length = 0
    timestep = 0
    go_loadb = 0

    # Training loop
    for i_episode in range(1, 20000):  # 玩多少局
        env.reset()  ##“环境初始化”
        for t in range(H.max_timesteps):  # 每一局玩到什么时候结束
            # order_togolist,img = env.observations() #输入状态
            Obs = env.observations()  # 输入状态
            timestep += 1
            action = agent.policy_old.act(t, Obs,memory)
            reward, ocu_list, Qv, load_b = env.step(H.action_list_scheduling[action], t)
            if env.is_running() == False:
                done = False
            else:
                done = True
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            running_reward += reward
            go_reward += reward
            running_Qv += Qv
            go_loadb += load_b
            if env.is_running() == True:  # 如果结束那么本局游戏结束
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

        # Update if its time
        if timestep > H.update_timestep:
            agent.update(memory)  # 如果步数是已经到了最大的步长了，那么更新agent
            memory.clear_memory()
            timestep = 0  # 置0

        avg_length += t  # 累计步长
        if i_episode % H.log_interval == 0:
            running_ocu = (running_ocu / H.log_interval)  # 这么多局的平均占有率？
            avg_length = int(avg_length / H.log_interval)  # 计算H.log_interval这么多局数的 平均每局的步长
            running_reward = int((running_reward / H.log_interval))  # 计算H.log_interval这么多局数的 平均每局的总回报

            running_Qv = running_Qv / H.log_interval
            running_loadb = running_loadb / H.log_interval
            wandb.log({
                "running_loadb": running_loadb,
                "running_reward": running_reward,
                "running_ocu": running_ocu,
                "running_Qv":running_Qv})
            running_reward = 0
            running_ocu = 0
            running_Qv = 0

            running_loadb = 0

    wandb.save('modelVMPO.h5')


if __name__ == '__main__':
    main()
