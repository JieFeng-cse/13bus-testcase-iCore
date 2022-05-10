### import collections
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

from environment_single_phase import create_56bus, VoltageCtrl_nonlinear
from env_single_phase_13bus import IEEE13bus, create_13bus
from env_single_phase_123bus import IEEE123bus, create_123bus
from env_three_phase_eu import Three_Phase_EU, create_eu_lv
from safeDDPG import ValueNetwork, SafePolicyNetwork, DDPG, ReplayBuffer, ReplayBufferPI, PolicyNetwork, SafePolicy3phase, StablePolicy3phase

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser(description='Single Phase Safe DDPG')
parser.add_argument('--env_name', default="13bus",
                    help='name of the environment to run')
parser.add_argument('--algorithm', default='safe-ddpg', help='name of algorithm')
parser.add_argument('--status', default='train')
parser.add_argument('--safe_type', default='three_single') #loss, dd
args = parser.parse_args()
seed = 10
torch.manual_seed(seed)

# plot policy
def plot_policy(agent_list, episode):
    s_array = np.zeros(30,)

    a_array_baseline = np.zeros(30,)
    a_array = np.zeros(30,)
    for i in range(30):
        state = torch.tensor([0.85+0.01*i])
        s_array[i] = state

        action_baseline = -(np.maximum(state-1.05, 0)-np.maximum(0.95-state, 0)).reshape((1,))
        action = -agent_list[3].policy_net(state.reshape(1,1))

        a_array_baseline[i] = action_baseline[0]
        a_array[i] = action[0]
        
    plt.figure() 
    plt.plot(s_array, a_array_baseline, label = 'Baseline')
    plt.plot(s_array, a_array, label = 'RL')
    plt.savefig('Policy{0}.png'.format(episode), dpi=100)

"""
Create Agent list and replay buffer
"""
vlr = 2e-4
plr = 1e-4
ph_num = 1
if args.env_name == '56bus':
    pp_net = create_56bus()
    injection_bus = np.array([18, 21, 30, 45, 53])-1  
    env = VoltageCtrl_nonlinear(pp_net, injection_bus)
    num_agent = 5
if args.env_name == '13bus':
    pp_net = create_13bus()
    injection_bus = np.array([2, 7, 9])
    env = IEEE13bus(pp_net, injection_bus)
    num_agent = 3
if args.env_name == '123bus':
    pp_net = create_123bus()
    injection_bus = np.array([10, 11, 16, 20, 33, 36, 48, 59, 66, 75, 83, 92, 104, 61])-1
    env = IEEE123bus(pp_net, injection_bus)
    num_agent = 14
    if args.algorithm == 'safe-ddpg':
        plr = 2e-4
if args.env_name == 'eu-lv':
    pp_net = create_eu_lv()
    injection_bus = np.array([25])#,70,32, 41
    env = Three_Phase_EU(pp_net, injection_bus)
    num_agent = len(injection_bus)
    ph_num=3


obs_dim = env.obs_dim
action_dim = env.action_dim
hidden_dim = 100
if args.env_name == 'eu-lv':
    type_name = 'three-phase'
else:
    type_name = 'single-phase'

agent_list = []
replay_buffer_list = []

for i in range(num_agent):
    value_net  = ValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    if args.algorithm == 'safe-ddpg' and not args.env_name=='eu-lv':
        policy_net = SafePolicyNetwork(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
        target_policy_net = SafePolicyNetwork(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    elif args.algorithm == 'safe-ddpg' and args.env_name=='eu-lv' and args.safe_type == 'loss':
        policy_net = PolicyNetwork(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
        target_policy_net = PolicyNetwork(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    elif args.algorithm == 'safe-ddpg' and args.env_name=='eu-lv' and args.safe_type == 'three_single':
        policy_net = SafePolicy3phase(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
        target_policy_net = SafePolicy3phase(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    elif args.algorithm == 'safe-ddpg' and args.env_name=='eu-lv' and args.safe_type == 'dd':
        policy_net = StablePolicy3phase(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
        target_policy_net = StablePolicy3phase(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    else:
        policy_net = PolicyNetwork(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
        target_policy_net = PolicyNetwork(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)

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

if args.status =='train':
    FLAG = 1
else:
    FLAG = 0

if (FLAG ==0): 
    # load trained policy
    for i in range(num_agent):
        if args.env_name == 'eu-lv':
            valuenet_dict = torch.load(f'checkpoints/{type_name}/{args.env_name}/{args.algorithm}/{args.safe_type}/value_net_checkpoint_a{i}.pth')
            policynet_dict = torch.load(f'checkpoints/{type_name}/{args.env_name}/{args.algorithm}/{args.safe_type}/policy_net_checkpoint_a{i}.pth')
        else:
            valuenet_dict = torch.load(f'checkpoints/{type_name}/{args.env_name}/{args.algorithm}/value_net_checkpoint_a{i}.pth')
            policynet_dict = torch.load(f'checkpoints/{type_name}/{args.env_name}/{args.algorithm}/policy_net_checkpoint_a{i}.pth')
        agent_list[i].value_net.load_state_dict(valuenet_dict)
        agent_list[i].policy_net.load_state_dict(policynet_dict)

elif (FLAG ==1):
    # training episode
    # for i in range(num_agent):
    #     if args.env_name == 'eu-lv':
    #         valuenet_dict = torch.load(f'checkpoints/{type_name}/{args.env_name}/{args.algorithm}/{args.safe_type}/value_net_checkpoint_a{i}.pth')
    #         policynet_dict = torch.load(f'checkpoints/{type_name}/{args.env_name}/{args.algorithm}/{args.safe_type}/policy_net_checkpoint_a{i}.pth')
    #     else:
    #         valuenet_dict = torch.load(f'checkpoints/{type_name}/{args.env_name}/{args.algorithm}/value_net_checkpoint_a{i}.pth')
    #         policynet_dict = torch.load(f'checkpoints/{type_name}/{args.env_name}/{args.algorithm}/policy_net_checkpoint_a{i}.pth')
    #     agent_list[i].value_net.load_state_dict(valuenet_dict)
    #     agent_list[i].policy_net.load_state_dict(policynet_dict)
    if args.algorithm == 'safe-ddpg':
        num_episodes = 200        
    else:
        num_episodes = 2000 #123 1000

    # trajetory length each episode
    num_steps = 30  
    if args.env_name =='123bus':
        num_steps = 60
    if args.env_name =='eu-lv':
        num_steps = 20
        num_episodes *= 2

    batch_size = 256

    # if/not plot trained policy every # episodes
    plot = False

    rewards = []
    avg_reward_list = []
    for episode in range(num_episodes):
        if(episode%50==0 and plot == True):
            plot_policy(agent_list, episode)

        state = env.reset(seed = episode)
        episode_reward = 0
        last_action = np.zeros((num_agent,ph_num)) #if single phase, 1, else ,3

        for step in range(num_steps):
            action = []
            action_p = []
            for i in range(num_agent):
                # sample action according to the current policy and exploration noise
                if args.env_name =='eu-lv':
                    action_agent = agent_list[i].policy_net.get_action(np.asarray([state[i]])) + np.random.normal(0, 0.005)
                    action_p.append(action_agent)
                    action_agent = np.clip(action_agent, -0.05, 0.05) 
                else:
                    action_agent = agent_list[i].policy_net.get_action(np.asarray([state[i]])) + np.random.normal(0, 1.5)/np.sqrt(episode+1)
                    action_agent = np.clip(action_agent, -0.8, 0.8) 
                    action_p.append(action_agent)                    
                action.append(action_agent)

            # PI policy    
            if args.env_name =='eu-lv':
                action = last_action - np.asarray(action).reshape(-1,3) #.reshape(-1,3) #if eu, reshape
            else:
                action = last_action - np.asarray(action)

            # execute action a_t and observe reward r_t and observe next state s_{t+1}
            next_state, reward, reward_sep, done = env.step_Preward(action, action_p)
            
            if(np.min(next_state)<0.75): #if voltage violation > 25%, episode ends.
                break
            else:
                for i in range(num_agent): # if single phase, state[i], else, state[0,i]
                    state_buffer = state[i].reshape(ph_num,) #[0.i]
                    action_buffer = action[i].reshape(ph_num,)
                    last_action_buffer = last_action[i].reshape(ph_num,)
                    next_state_buffer = next_state[i].reshape(ph_num, )

                    # store transition (s_t, a_t, r_t, s_{t+1}) in R
                    replay_buffer_list[i].push(state_buffer, action_buffer, last_action_buffer,
                                               reward_sep[i], next_state_buffer, done) #_sep[i]

                    # update both critic and actor network
                    if args.env_name =='eu-lv' and args.algorithm == 'safe-ddpg' and args.safe_type == 'loss':
                        if len(replay_buffer_list[i]) > batch_size:
                            agent_list[i].train_step_3ph(replay_buffer=replay_buffer_list[i], 
                                                    batch_size=batch_size)
                    else:
                        if len(replay_buffer_list[i]) > batch_size:
                            agent_list[i].train_step(replay_buffer=replay_buffer_list[i], 
                                                    batch_size=batch_size)

                if(done):
                    episode_reward += reward  
                    break
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
        if args.env_name == 'eu-lv':
            pth_value = f'checkpoints/{type_name}/{args.env_name}/{args.algorithm}/{args.safe_type}/value_net_checkpoint_a{i}.pth'
            pth_policy = f'checkpoints/{type_name}/{args.env_name}/{args.algorithm}/{args.safe_type}/policy_net_checkpoint_a{i}.pth'
        else:
            pth_value = f'checkpoints/{type_name}/{args.env_name}/{args.algorithm}/value_net_checkpoint_a{i}.pth'
            pth_policy = f'checkpoints/{type_name}/{args.env_name}/{args.algorithm}/policy_net_checkpoint_a{i}.pth'
        torch.save(agent_list[i].value_net.state_dict(), pth_value)
        torch.save(agent_list[i].policy_net.state_dict(), pth_policy)

else:
    raise ValueError("Model loading optition does not exist!")


# title = ['Bus 18', 'Bus 21', 'Bus 30', 'Bus 45', 'Bus 53']
# title = ['Bus 4', 'Bus 10', 'Bus 12']
# why small actions have better rewards, I have to check that
if args.status == 'train':
    check_buffer = replay_buffer_list[0]
    buffer_len = replay_buffer_list[0].__len__()
    state, action, last_action, reward, next_state, done = replay_buffer_list[0].sample(buffer_len-1)
    if ph_num==1:
        plt.scatter(action,reward)
        plt.title('bus 0')
        plt.savefig('bus0.png')
        plt.show()
    else:
        plt.scatter(action[:,0],reward)
        plt.title('bus 0')
        plt.savefig('bus0.png')
        plt.show()


# check_buffer = replay_buffer_list[1]
# buffer_len = replay_buffer_list[1].__len__()
# state, action, last_action, reward, next_state, done = replay_buffer_list[1].sample(buffer_len-1)
# plt.scatter(action,reward)
# plt.title('bus 1')
# plt.savefig('bus1.png')
# plt.show()


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
            action_baseline = (np.maximum(state[0]-1.05, 0)-np.maximum(0.95-state[0], 0)).reshape((1,))
    
        action = agent_list[i].policy_net.get_action(np.asarray([state]))
        action = np.clip(action, -0.8, 0.8) 
        a_array_baseline[j] = -action_baseline[0]
        a_array[j] = -action
    axs[i].plot(s_array, 5*a_array_baseline, '-.', label = 'Linear')
    for k in range(ph_num):        
        axs[i].plot(s_array, a_array[:,k], label = args.algorithm)
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
for step in range(60):
    action = []
    for i in range(num_agent):
        # sample action according to the current policy and exploration noise
        action_agent = agent_list[i].policy_net.get_action(np.asarray([state[i]]))
        # action_agent = (np.maximum(state[i]-1.05, 0)-np.maximum(0.95-state[i], 0)).reshape((1,))
        action_agent = np.clip(action_agent, -0.8, 0.8)
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
    # axs[i].plot(range(len(action_list)), np.array(action_list)[:,i], label = 'action')
    axs[i].legend(loc='lower left')
fig1, axs1 = plt.subplots(1, num_agent+1, figsize=(15,3))
for i in range(num_agent):
    axs1[i].plot(range(len(action_list)), np.array(action_list)[:len(action_list),i], '-.', label = 'actions')
    # axs[i].plot(range(len(action_list)), np.array(action_list)[:,i], label = 'action')
    axs1[i].legend(loc='lower left')
axs[num_agent].plot(range(len(reward_list)),reward_list)
plt.show()

# test success rate:
