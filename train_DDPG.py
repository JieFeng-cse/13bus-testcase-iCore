### import collections
from cv2 import phase
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import linalg as LA
import gym
import os
import random
import sys
from gym import spaces
from gym.utils import seeding
import copy

from scipy.io import loadmat
import pandapower as pp
import pandapower.networks as pn
import pandas as pd 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from IEEE_13_3p import IEEE13bus3p, create_13bus3p
from safeDDPG import ValueNetwork, DDPG,  ReplayBufferPI, SafePolicy3phase

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

seed = 10
torch.manual_seed(seed)

"""
Create the environment
"""
vlr = 2e-4
plr = 1e-4
batch_size = 256
status = 'test'
# status = 'train'


# injection_bus = np.array([675,633,680])
injection_bus = np.array([633,671,645,646,692,675,652,632,680,684])
pp_net, injection_bus_dict = create_13bus3p(injection_bus) 
max_ac = 0.5
env = IEEE13bus3p(pp_net,injection_bus_dict)
num_agent = len(injection_bus)
ph_num=3


obs_dim = env.obs_dim
action_dim = env.action_dim
hidden_dim = 100
if ph_num == 3:
    type_name = 'three-phase'
else:
    type_name = 'single-phase'

"""
Create Agent list and replay buffer
"""
agent_list = []
replay_buffer_list = []

for i in range(num_agent):
    if ph_num == 3:
        obs_dim = len(env.injection_bus[env.injection_bus_str[i]])
        action_dim = obs_dim
    value_net  = ValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    policy_net = SafePolicy3phase(env, obs_dim, action_dim, hidden_dim, env.injection_bus_str[i]).to(device)
    target_policy_net = SafePolicy3phase(env, obs_dim, action_dim, hidden_dim, env.injection_bus_str[i]).to(device)
    target_value_net  = ValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)    

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(param.data)

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(param.data)

    agent = DDPG(policy_net=policy_net, value_net=value_net,
                 target_policy_net=target_policy_net, target_value_net=target_value_net, value_lr=vlr, policy_lr=plr)
    
    replay_buffer = ReplayBufferPI(capacity=1000000)
    
    agent_list.append(agent)
    replay_buffer_list.append(replay_buffer)

if status =='train':
    FLAG = 1
else:
    FLAG = 0

def get_id(phases):
    if phases == 'abc':
        id = [0,1,2]
    elif phases == 'ab':
        id = [0,1]
    elif phases == 'ac':
        id = [0,2]
    elif phases == 'bc':
        id = [1,2]
    elif phases == 'a':
        id = [0]
    elif phases == 'b':
        id = [1]
    elif phases == 'c':
        id = [2]
    else:
        print("error!")
        exit(0)
    return id

if (FLAG ==0): 
    # load trained policy
    # for i in range(num_agent):
    #     #change the path
    #     valuenet_dict = torch.load(f'checkpoints/three-phase/13bus3p/safe-ddpg/three_single/value_net_checkpoint_a{injection_bus[i]}.pth')
    #     policynet_dict = torch.load(f'checkpoints/three-phase/13bus3p/safe-ddpg/three_single/policy_net_checkpoint_a{injection_bus[i]}.pth')
     for i in range(num_agent):        
        valuenet_dict = torch.load(f'checkpoints/three-phase/13bus3p/safe-ddpg/three_single copy/value_net_checkpoint_a{i}.pth')
        policynet_dict = torch.load(f'checkpoints/three-phase/13bus3p/safe-ddpg/three_single copy/policy_net_checkpoint_a{i}.pth')
        
        agent_list[i].value_net.load_state_dict(valuenet_dict)
        agent_list[i].policy_net.load_state_dict(policynet_dict)

elif (FLAG ==1):
    # load a linear controller as the start point
    for i in range(num_agent):        
        valuenet_dict = torch.load(f'checkpoints/three-phase/13bus3p/safe-ddpg/three_single copy/value_net_checkpoint_a{i}.pth')
        policynet_dict = torch.load(f'checkpoints/three-phase/13bus3p/safe-ddpg/three_single copy/policy_net_checkpoint_a{i}.pth')

        agent_list[i].value_net.load_state_dict(valuenet_dict)
        agent_list[i].policy_net.load_state_dict(policynet_dict) 
        for target_param, param in zip(agent_list[i].target_value_net.parameters(), agent_list[i].value_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(agent_list[i].target_policy_net.parameters(), agent_list[i].policy_net.parameters()):
            target_param.data.copy_(param.data)

    # training episode
    num_episodes = 500

    # trajetory length each episode
    num_steps = 30     

    # if/not plot trained policy every # episodes
    plot = False
    rewards = []
    avg_reward_list = []
    for episode in range(num_episodes):
        state = env.reset(seed = episode)
        episode_reward = 0
        last_action = np.zeros((num_agent,ph_num)) #if single phase, 1, else ,3

        for step in range(num_steps):
            action = []
            action_p = []
            for i in range(num_agent):
                # sample action according to the current policy and exploration noise                
                action_agent = np.zeros(3)
                phases = env.injection_bus[env.injection_bus_str[i]]
                id = get_id(phases)
                action_tmp = agent_list[i].policy_net.get_action(np.asarray([state[i,id]])) + np.random.normal(0, max_ac)/np.sqrt(episode+1)
                action_tmp = action_tmp.reshape(len(id),)  
                for j in range(len(phases)):
                    action_agent[id[j]]=action_tmp[j]           
                # action_p.append(action_agent)
                action_agent = np.clip(action_agent, -max_ac, max_ac) 
                action_p.append(action_agent)                
                action.append(action_agent)

            # PI policy                
            action = last_action - np.asarray(action).reshape(-1,3)             

            # execute action a_t and observe reward r_t and observe next state s_{t+1}
            next_state, reward, reward_sep, done = env.step_Preward(action, action_p)
            
            if(np.min(next_state)<0.75): #if voltage violation > 25%, episode ends.
                break
            else:
                for i in range(num_agent): 
                    
                    phases = env.injection_bus[env.injection_bus_str[i]]
                    id = get_id(phases)
                    state_buffer = state[i,id].reshape(len(phases),) 
                    action_buffer = action[i,id].reshape(len(phases),) 
                    last_action_buffer = last_action[i,id].reshape(len(phases),) 
                    next_state_buffer = next_state[i,id].reshape(len(phases),) 
                    # store transition (s_t, a_t, r_t, s_{t+1}) in R
                    replay_buffer_list[i].push(state_buffer, action_buffer, last_action_buffer,
                                               reward_sep[i], next_state_buffer, done) #_sep[i]

                    # update both critic and actor network
                    
                    if len(replay_buffer_list[i]) > batch_size:
                        agent_list[i].train_step(replay_buffer=replay_buffer_list[i], 
                                                batch_size=batch_size)

                if(done):
                    episode_reward += reward  
                    break #no break if 13bus3p
                else:
                    state = np.copy(next_state)
                    episode_reward += reward    

            last_action = np.copy(action)

        rewards.append(episode_reward)
        avg_reward = np.mean(rewards[-40:])
        if(episode%50==0):
            print("Episode * {} * Avg Reward is ==> {}".format(episode, avg_reward))
        avg_reward_list.append(avg_reward)
    for i in range(num_agent):
        pth_value = f'checkpoints/three-phase/13bus3p/safe-ddpg/three_single/value_net_checkpoint_a{injection_bus[i]}.pth'
        pth_policy = f'checkpoints/three-phase/13bus3p/safe-ddpg/three_single/policy_net_checkpoint_a{injection_bus[i]}.pth'
        
        torch.save(agent_list[i].value_net.state_dict(), pth_value)
        torch.save(agent_list[i].policy_net.state_dict(), pth_policy)

else:
    raise ValueError("Model loading optition does not exist!")


fig, axs = plt.subplots(1, num_agent, figsize=(15,3))
for i in range(num_agent):
    # plot policy
    N = 40
    s_array = np.zeros(N,)
    
    a_array_baseline = np.zeros(N,)
    a_array = np.zeros((N,ph_num))
    
    for j in range(N):
        if ph_num ==1:
            state = np.array([0.8+0.01*j])
            s_array[j] = state
            action_baseline = (np.maximum(state-1.05, 0)-np.maximum(0.95-state, 0)).reshape((1,))
        else:
            state = np.resize(np.array([0.8+0.01*j]),(3))
            s_array[j] = state[0]        
            action_baseline = (np.maximum(state[0]-1.03, 0)-np.maximum(0.97-state[0], 0)).reshape((1,))
        if ph_num == 3: 
            action = np.zeros(3)
            phases = env.injection_bus[env.injection_bus_str[i]]
            id = get_id(phases)
            action_tmp = agent_list[i].policy_net.get_action(np.asarray([state[id]])) 
            action_tmp = action_tmp.reshape(len(id),)  
            for p in range(len(phases)):
                action[id[p]]=action_tmp[p]
        else:
            action = agent_list[i].policy_net.get_action(np.asarray([state])) 
        action = np.clip(action, -max_ac, max_ac) 
        a_array_baseline[j] = -action_baseline[0]
        a_array[j] = -action
    axs[i].plot(s_array, 2*a_array_baseline, '-.', label = 'Linear')
    for k in range(ph_num):        
        axs[i].plot(s_array, a_array[:,k], label = 'safe 3 pahse')
        axs[i].legend(loc='lower left')
plt.show()

## test policy
state = env.reset()
episode_reward = 0
last_action = np.zeros((num_agent,1))
action_list=[]
state_list =[]
reward_list = []
state_list.append(state)
for step in range(100):
    action = []
    for i in range(num_agent):
        # sample action according to the current policy and exploration noise
        action_agent = np.zeros(3)
        phases = env.injection_bus[env.injection_bus_str[i]]
        id = get_id(phases)
        action_tmp = agent_list[i].policy_net.get_action(np.asarray([state[i,id]]))
        action_tmp = action_tmp.reshape(len(id),)  
        for i in range(len(phases)):
            action_agent[id[i]]=action_tmp[i]
        action_agent = np.clip(action_agent, -max_ac, max_ac) 
        action.append(action_agent)

    # PI policy    
    action = last_action - np.asarray(action)

    # execute action a_t and observe reward r_t and observe next state s_{t+1}
    next_state, reward, reward_sep, done = env.step_Preward(action, (last_action-action))
    reward_list.append(reward)
    if done:
        print("finished")
    action_list.append(last_action-action)
    state_list.append(next_state)
    last_action = np.copy(action)
    state = next_state
fig, axs = plt.subplots(1, num_agent+1, figsize=(15,3))
for i in range(num_agent):
    axs[i].plot(range(len(action_list)), np.array(state_list)[:len(action_list),i], '-.', label = 'states')
    axs[i].legend(loc='lower left')
fig1, axs1 = plt.subplots(1, num_agent+1, figsize=(15,3))
for i in range(num_agent):
    axs1[i].plot(range(len(action_list)), np.array(action_list)[:len(action_list),i], '-.', label = 'actions')
    axs1[i].legend(loc='lower left')
axs[num_agent].plot(range(len(reward_list)),reward_list)
plt.show()
