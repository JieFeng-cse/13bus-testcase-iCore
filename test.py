### import collections
from cProfile import label
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
from safeDDPG import ValueNetwork, SafePolicyNetwork, DDPG, ReplayBuffer, ReplayBufferPI, PolicyNetwork

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser(description='Single Phase Safe DDPG')
parser.add_argument('--env_name', default="13bus",
                    help='name of the environment to run')
parser.add_argument('--algorithm', default='safe-ddpg', help='name of algorithm')
args = parser.parse_args()
seed = 10
torch.manual_seed(seed)
plt.rcParams['font.size'] = '20'

"""
Create Agent list and replay buffer
"""
max_ac = 0.3
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
    max_ac = 0.8

obs_dim = env.obs_dim
action_dim = env.action_dim
hidden_dim = 100


ddpg_agent_list = []
safe_ddpg_agent_list = []

for i in range(num_agent):
    safe_ddpg_value_net  = ValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)    
    safe_ddpg_policy_net = SafePolicyNetwork(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    
    ddpg_value_net  = ValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)  
    ddpg_policy_net = PolicyNetwork(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)    

    ddpg_agent = DDPG(policy_net=ddpg_policy_net, value_net=ddpg_value_net,
                 target_policy_net=ddpg_policy_net, target_value_net=ddpg_value_net)
    
    safe_ddpg_agent = DDPG(policy_net=safe_ddpg_policy_net, value_net=safe_ddpg_value_net,
                 target_policy_net=safe_ddpg_policy_net, target_value_net=safe_ddpg_value_net)    
    
    ddpg_agent_list.append(ddpg_agent)
    safe_ddpg_agent_list.append(safe_ddpg_agent)

for i in range(num_agent):
    ddpg_valuenet_dict = torch.load(f'checkpoints/single-phase/{args.env_name}/ddpg/value_net_checkpoint_a{i}.pth')
    ddpg_policynet_dict = torch.load(f'checkpoints/single-phase/{args.env_name}/ddpg/policy_net_checkpoint_a{i}.pth')
    ddpg_agent_list[i].value_net.load_state_dict(ddpg_valuenet_dict)
    ddpg_agent_list[i].policy_net.load_state_dict(ddpg_policynet_dict)

    safe_ddpg_valuenet_dict = torch.load(f'checkpoints/single-phase/{args.env_name}/safe-ddpg/value_net_checkpoint_a{i}.pth')
    safe_ddpg_policynet_dict = torch.load(f'checkpoints/single-phase/{args.env_name}/safe-ddpg/policy_net_checkpoint_a{i}.pth')
    safe_ddpg_agent_list[i].value_net.load_state_dict(safe_ddpg_valuenet_dict)
    safe_ddpg_agent_list[i].policy_net.load_state_dict(safe_ddpg_policynet_dict)
def plot_action():
    
    fig, axs = plt.subplots(1, 1, figsize=(16,12))
    for i in range(num_agent):
        # plot policy
        N = 40
        s_array = np.zeros(N,)
        
        a_array_baseline = np.zeros(N,)
        safe_ddpg_a_array = np.zeros(N,)
        
        for j in range(N):
            state = np.array([0.8+0.01*j])
            s_array[j] = state

            safe_ddpg_action = safe_ddpg_agent_list[i].policy_net.get_action(np.asarray([state]))
            safe_ddpg_action = np.clip(safe_ddpg_action, -max_ac, max_ac)
            safe_ddpg_a_array[j] = -safe_ddpg_action

        axs.plot(s_array, safe_ddpg_a_array, label = f'stable-DDPG at {injection_bus[i]}')
    for i in range(num_agent):
        # plot policy
        N = 40
        s_array = np.zeros(N,)
        
        a_array_baseline = np.zeros(N,)
        ddpg_a_array = np.zeros(N,)
        safe_ddpg_a_array = np.zeros(N,)
        
        for j in range(N):
            state = np.array([0.8+0.01*j])
            s_array[j] = state

            action_baseline = (np.maximum(state-1.03, 0)-np.maximum(0.97-state, 0)).reshape((1,))
        
            ddpg_action = ddpg_agent_list[i].policy_net.get_action(np.asarray([state]))
            ddpg_action = np.clip(ddpg_action, -max_ac, max_ac)
            a_array_baseline[j] = -action_baseline[0]
            ddpg_a_array[j] = -ddpg_action

        
        axs.plot(s_array, ddpg_a_array, '--', label = f'DDPG at {injection_bus[i]}')
    axs.plot(s_array, 2*a_array_baseline, '-.', label = 'Linear',color='r')
    axs.legend(loc='lower left', prop={"size":20})
    plt.show()
def plot_traj():
    ddpg_plt=[]
    safe_plt = []
    ddpg_a_plt=[]
    safe_a_plt = []

    state = env.reset()
    episode_reward = 0
    last_action = np.zeros((num_agent,1))
    action_list=[]
    state_list =[]
    state_list.append(state)
    for step in range(40):
        action = []
        for i in range(num_agent):
            # sample action according to the current policy and exploration noise
            action_agent = ddpg_agent_list[i].policy_net.get_action(np.asarray([state[i]]))#+np.random.normal(0, 0.05)
            action_agent = np.clip(action_agent, -max_ac, max_ac)
            action.append(action_agent)

        # PI policy    
        action = last_action - np.asarray(action)

        # execute action a_t and observe reward r_t and observe next state s_{t+1}
        next_state, reward, reward_sep, done = env.step_Preward(action, (last_action-action))
        if done:
            print("finished")
        action_list.append(action)
        state_list.append(next_state)
        last_action = np.copy(action)
        state = next_state
    fig, axs = plt.subplots(1, 2, figsize=(16,9))
    # lb = axs[0].plot(range(len(action_list)), [0.95]*len(action_list), linestyle='--', dashes=(5, 10), color='g', label='lower bound')
    # ub = axs[0].plot(range(len(action_list)), [1.05]*len(action_list), linestyle='--', dashes=(5, 10), color='r', label='upper bound')
    for i in range(num_agent):    
        dps = axs[0].plot(range(len(action_list)), np.array(state_list)[:len(action_list),i], '-.', label = f'DDPG at {injection_bus[i]}')
        dpa = axs[1].plot(range(len(action_list)), np.array(action_list)[:,i], '-.', label = f'DDPG at {injection_bus[i]}')
        ddpg_plt.append(dps)
        ddpg_a_plt.append(dpa)

    state = env.reset()
    episode_reward = 0
    last_action = np.zeros((num_agent,1))
    action_list=[]
    state_list =[]
    state_list.append(state)
    for step in range(40):
        action = []
        for i in range(num_agent):
            # sample action according to the current policy and exploration noise
            action_agent = safe_ddpg_agent_list[i].policy_net.get_action(np.asarray([state[i]]))#+np.random.normal(0, 0.05)
            # action_agent = (np.maximum(state[i]-1.05, 0)-np.maximum(0.95-state[i], 0)).reshape((1,))*2
            action_agent = np.clip(action_agent, -max_ac, max_ac)
            action.append(action_agent)

        # PI policy    
        action = last_action - np.asarray(action)

        # execute action a_t and observe reward r_t and observe next state s_{t+1}
        next_state, reward, reward_sep, done = env.step_Preward(action, (last_action-action))
        if done:
            print("finished")
        action_list.append(action)
        state_list.append(next_state)
        last_action = np.copy(action)
        state = next_state
    safe_name = []
    for i in range(num_agent):    
        safes=axs[0].plot(range(len(action_list)), np.array(state_list)[:len(action_list),i], '-', label = f'stable-DDPG at {injection_bus[i]}')
        safea=axs[1].plot(range(len(action_list)), np.array(action_list)[:,i], label = f'stable-DDPG at {injection_bus[i]}')
        safe_plt.append(safes)
        safe_name.append(f'stable-DDPG at {injection_bus[i]}')
        safe_a_plt.append(safea)
    # leg1 = plt.legend(safe_a_plt, safe_name, loc='lower left')
    axs[0].legend(loc='upper right', prop={"size":20})
    axs[1].legend(loc='lower left', prop={"size":20})
    axs[0].set_xlabel('Iteretion Steps')   
    axs[1].set_xlabel('Iteretion Steps')  
    axs[0].set_ylabel('Bus Voltage [p.u.]')   
    axs[1].set_ylabel('Reactive Power Injection [MVar]')  
    plt.show()

#test success rate, voltage violation after 40 steps
def test_suc_rate(algm, step_num=60):
    success_num = 0
    final_state_list = []
    final_step_list = []
    control_cost_list = []
    for i in range(500):
        state = env.reset(i)
        episode_reward = 0
        last_action = np.zeros((num_agent,1))
        action_list=[]
        state_list =[]
        state_list.append(state)
        control_action = []
        for step in range(step_num):
            action = []
            for i in range(num_agent):
                # sample action according to the current policy and exploration noise
                if algm == 'linear':
                    action_agent = (np.maximum(state[i]-1.03, 0)-np.maximum(0.97-state[i], 0)).reshape((1,))*5
                elif algm == 'safe-ddpg':
                    action_agent = safe_ddpg_agent_list[i].policy_net.get_action(np.asarray([state[i]]))
                elif algm == 'ddpg':
                    action_agent = ddpg_agent_list[i].policy_net.get_action(np.asarray([state[i]]))
                # 
                action_agent = np.clip(action_agent, -max_ac, max_ac)
                action.append(action_agent)

            # PI policy    
            action = last_action - np.asarray(action)
            control_action.append(np.abs(action))
            # execute action a_t and observe reward r_t and observe next state s_{t+1}
            next_state, reward, reward_sep, done = env.step_Preward(action, (last_action-action))
            if done:
                success_num += 1
                final_step_list.append(step+1)
                control_cost_list.append(np.sum(np.asarray(control_action)))
                break
            action_list.append(last_action-action)
            state_list.append(next_state)
            last_action = np.copy(action)
            state = next_state
        
        # if not done:
        #     final_step_list.append(step_num)
        final_state_list.append(next_state)
    print(success_num)
    print(np.mean(final_step_list), np.std(final_step_list))
    print('cost',np.mean(control_cost_list), np.std(control_cost_list))
    # for i in range(num_agent):
    #     states = np.array(final_state_list)
    #     n, bins, patches = axs[i].hist(states[:,i], [0.7,0.85,0.90,0.95,1.0,1.05,1.1,1.15,1.2])
    # plt.title('safe_ddpg, high voltage')
    # plt.show()
    return final_state_list
def plot_bar(num_agent):
    plt.rcParams['font.size'] = '11'
    
    plt.figure(figsize=(3.96,2.87),dpi=300)
    plt.gcf().subplots_adjust(bottom=0.17,left=0.15)
    marks=[1.0,0,0,0,0]
    bars=('Stable','5-7%','7-9%','9-10%','>10%')
    y=np.arange(len(bars))
    plt.bar(y+0.2,marks, 0.4,color='b',label='Stable-DDPG')
    state_list = test_suc_rate('ddpg')
    state_list = np.asarray(state_list)
    state_list = np.abs(state_list-1)
    marks = [0,0,0,0,0]
    marks[0]=np.sum(state_list<0.05)
    marks[1]=np.sum(state_list<0.07)-marks[0]
    marks[2]=np.sum(state_list<0.09)-marks[1]-marks[0]
    marks[3]=np.sum(state_list<0.1)-marks[1]-marks[0]-marks[2]
    marks[4]=len(state_list)*num_agent-marks[1]-marks[0]-marks[2]-marks[3]
    marks = np.array(marks)/len(state_list)/num_agent
    print(marks)
    plt.bar(y-0.2,marks,0.4,color='r',label='DDPG')
    
    plt.xticks(y,bars)
    plt.xlabel('Voltage Violation (low voltage)')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()
#11, 36, 75
def plot_traj_123():
    ddpg_plt=[]
    safe_plt = []
    ddpg_a_plt=[]
    safe_a_plt = []

    state = env.reset()
    episode_reward = 0
    last_action = np.zeros((num_agent,1))
    action_list=[]
    state_list =[]
    state_list.append(state)
    for step in range(60):
        action = []
        for i in range(num_agent):
            # sample action according to the current policy and exploration noise
            action_agent = ddpg_agent_list[i].policy_net.get_action(np.asarray([state[i]]))#+np.random.normal(0, 0.05)
            action_agent = np.clip(action_agent, -max_ac, max_ac)
            action.append(action_agent)

        # PI policy    
        action = last_action - np.asarray(action)

        # execute action a_t and observe reward r_t and observe next state s_{t+1}
        next_state, reward, reward_sep, done = env.step_Preward(action, (last_action-action))
        if done:
            print("finished")
        action_list.append(action)
        state_list.append(next_state)
        last_action = np.copy(action)
        state = next_state
    fig, axs = plt.subplots(1, 2, figsize=(16,9))
    # lb = axs[0].plot(range(len(action_list)), [0.95]*len(action_list), linestyle='--', dashes=(5, 10), color='g', label='lower bound')
    # ub = axs[0].plot(range(len(action_list)), [1.05]*len(action_list), linestyle='--', dashes=(5, 10), color='r', label='upper bound')
    for i in [1,5,9]:    
        dps = axs[0].plot(range(len(action_list)), np.array(state_list)[:len(action_list),i], '-.', label = f'DDPG at {injection_bus[i]+1}')
        dpa = axs[1].plot(range(len(action_list)), np.array(action_list)[:,i], '-.', label = f'DDPG at {injection_bus[i]+1}')
        ddpg_plt.append(dps)
        ddpg_a_plt.append(dpa)

    state = env.reset()
    episode_reward = 0
    last_action = np.zeros((num_agent,1))
    action_list=[]
    state_list =[]
    state_list.append(state)
    for step in range(60):
        action = []
        for i in range(num_agent):
            # sample action according to the current policy and exploration noise
            action_agent = safe_ddpg_agent_list[i].policy_net.get_action(np.asarray([state[i]]))#+np.random.normal(0, 0.05)
            # action_agent = (np.maximum(state[i]-1.05, 0)-np.maximum(0.95-state[i], 0)).reshape((1,))*2
            action_agent = np.clip(action_agent, -max_ac, max_ac)
            action.append(action_agent)

        # PI policy    
        action = last_action - np.asarray(action)

        # execute action a_t and observe reward r_t and observe next state s_{t+1}
        next_state, reward, reward_sep, done = env.step_Preward(action, (last_action-action))
        if done:
            print("finished")
        action_list.append(action)
        state_list.append(next_state)
        last_action = np.copy(action)
        state = next_state
    safe_name = []
    for i in [1,5,9]:    
        safes=axs[0].plot(range(len(action_list)), np.array(state_list)[:len(action_list),i], '-', label = f'stable-DDPG at {injection_bus[i]+1}')
        safea=axs[1].plot(range(len(action_list)), np.array(action_list)[:,i], label = f'stable-DDPG at {injection_bus[i]+1}')
        safe_plt.append(safes)
        safe_name.append(f'stable-DDPG at {injection_bus[i]}')
        safe_a_plt.append(safea)
    # leg1 = plt.legend(safe_a_plt, safe_name, loc='lower left')
    axs[0].legend(loc='lower left', prop={"size":20})
    axs[1].legend(loc='lower left', prop={"size":20})
    axs[0].set_xlabel('Iteretion Steps')   
    axs[1].set_xlabel('Iteretion Steps')  
    axs[0].set_ylabel('Bus Voltage [p.u.]')   
    axs[1].set_ylabel('Reactive Power Injection [MVar]')  
    plt.show()

def plot_action_selcted(selected=[1,5,9]):    
    fig, axs = plt.subplots(1, 3, figsize=(9,3))
    for indx,i in enumerate(selected):
        # plot policy
        N = 40
        s_array = np.zeros(N,)
        
        a_array_baseline = np.zeros(N,)
        safe_ddpg_a_array = np.zeros(N,)
        
        for j in range(N):
            state = np.array([0.8+0.01*j])
            s_array[j] = state

            safe_ddpg_action = safe_ddpg_agent_list[i].policy_net.get_action(np.asarray([state]))
            safe_ddpg_action = np.clip(safe_ddpg_action, -max_ac, max_ac)
            safe_ddpg_a_array[j] = -safe_ddpg_action

        axs[indx].plot(s_array, safe_ddpg_a_array, label = f'stable-DDPG')
    for indx,i in enumerate(selected):
        # plot policy
        N = 40
        s_array = np.zeros(N,)
        
        a_array_baseline = np.zeros(N,)
        ddpg_a_array = np.zeros(N,)
        safe_ddpg_a_array = np.zeros(N,)
        
        for j in range(N):
            state = np.array([0.8+0.01*j])
            s_array[j] = state

            action_baseline = (np.maximum(state-1.03, 0)-np.maximum(0.97-state, 0)).reshape((1,))*5
        
            ddpg_action = ddpg_agent_list[i].policy_net.get_action(np.asarray([state]))
            ddpg_action = np.clip(ddpg_action, -max_ac, max_ac)
            action_baseline = np.clip(action_baseline, -max_ac, max_ac)
            a_array_baseline[j] = -action_baseline[0]
            ddpg_a_array[j] = -ddpg_action

        axs[indx].set_title(f'Bus {injection_bus[i]+1}')
        axs[indx].plot(s_array, ddpg_a_array, '--', label = f'DDPG')
        axs[indx].plot(s_array, a_array_baseline, '-.', label = 'Linear',color='r')
        axs[indx].legend(loc='upper right', prop={"size":10})
    plt.show()
if __name__ == "__main__":
    # test_suc_rate('ddpg',step_num=100) #safe-ddpg
    # test_suc_rate('linear')
    # plot_action_selcted()
    # plot_bar(len(injection_bus))
    plot_traj_123()
    # plot_action_selcted()
