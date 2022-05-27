import numpy as np
from numpy import linalg as LA
import gym
import os
import random
import sys
from gym import spaces
from gym.utils import seeding
import copy
import matplotlib.pyplot as plt

from scipy.io import loadmat
import pandapower as pp
import pandapower.networks as pn
import pandas as pd 
import math

from dssdata import SystemClass
from dssdata.pfmodes import run_static_pf
from dssdata.tools import voltages
from dssdata.pfmodes import cfg_tspf

DSS_PATH = "/home/jason/Documents/research/stable-rl-three-phase/opendss_model/13bus/IEEE13Nodeckt.dss"

def create_13bus3p():
    #build the generators
    distSys = SystemClass(path=DSS_PATH, kV=[115, 4.16, 0.48])
    cfg_tspf(distSys,'0.02s')
    cmd = [
    f"New Generator.bus675_1 bus1=675.1 Phases=1 kv=4.16 kw=0 kvar=0 pf=1 model=1",
    f"New Generator.bus675_2 bus1=675.2 Phases=1 kv=4.16 kw=0 kvar=0 pf=1 model=1",
    f"New Generator.bus675_3 bus1=675.3 Phases=1 kv=4.16 kw=0 kvar=0 pf=1 model=1",
    f"New Generator.bus633_1 bus1=633.1 Phases=1 kv=4.16 kw=0 kvar=0 pf=1 model=1",
    f"New Generator.bus633_2 bus1=633.2 Phases=1 kv=4.16 kw=0 kvar=0 pf=1 model=1",
    f"New Generator.bus633_3 bus1=633.3 Phases=1 kv=4.16 kw=0 kvar=0 pf=1 model=1",
    f"New Generator.bus680_1 bus1=680.1 Phases=1 kv=4.16 kw=0 kvar=0 pf=1 model=1",
    f"New Generator.bus680_2 bus1=680.2 Phases=1 kv=4.16 kw=0 kvar=0 pf=1 model=1",
    f"New Generator.bus680_3 bus1=680.3 Phases=1 kv=4.16 kw=0 kvar=0 pf=1 model=1",
      ]
    distSys.dsscontent = distSys.dsscontent + cmd
    return distSys

class IEEE13bus3p(gym.Env):
    def __init__(self, distSys, injection_bus, v0=1, vmax=1.05, vmin=0.95):
        self.network =  distSys
        self.obs_dim = 3
        self.action_dim = 3
        self.injection_bus = injection_bus
        self.injection_bus_str = []
        self.agentnum = len(injection_bus)
        for i in range(self.agentnum):
            self.injection_bus_str.append(str(self.injection_bus[i]))
        
        self.v0 = v0 
        self.vmax = vmax
        self.vmin = vmin
        
        self.state = np.ones((self.agentnum, 3))
    def get_state(self):
        v_pu = voltages.get_from_buses(self.network, self.injection_bus_str)
        state_a = v_pu['v_pu_a'].to_numpy().reshape(-1,1)
        state_b = v_pu['v_pu_b'].to_numpy().reshape(-1,1)
        state_c = v_pu['v_pu_c'].to_numpy().reshape(-1,1)
        self.state = np.hstack([state_a, state_b, state_c]) #shape: number_of_bus*3
        return self.state
    
    def step_Preward(self, action, p_action): 
        
        done = False 
        #safe-ddpg reward
        reward = float(-0.0*LA.norm(p_action,1)-1000*LA.norm(np.clip(self.state-self.vmax, 0, np.inf),1)
                       -1000*LA.norm(np.clip(self.vmin-self.state, 0, np.inf),1))
        #why in this part originally it is not square?
        # local reward
        agent_num = len(self.injection_bus)
        reward_sep = np.zeros(agent_num, )
        #just for ddpg
        for i in range(agent_num):
            reward_sep[i] = float(-0.0*LA.norm(p_action[i])**2 -1000*LA.norm(np.clip(self.state[i]-self.vmax, 0, np.inf),1)
                           - 1000*LA.norm(np.clip(self.vmin-self.state[i], 0, np.inf),1) )     
        # state-transition dynamics
        action = action * 1000 #from kVar to MVar
        for i in range(len(self.injection_bus)):
            self.network.run_command(f"Generator.bus{self.injection_bus_str[i]}_1.kvar={action[i,0]}") 
            self.network.run_command(f"Generator.bus{self.injection_bus_str[i]}_2.kvar={action[i,1]}") 
            self.network.run_command(f"Generator.bus{self.injection_bus_str[i]}_3.kvar={action[i,2]}") 
        self.network.dss.Solution.Number(1)
        self.network.dss.Solution.Solve()
        self.state=self.get_state()
        
        if(np.min(self.state) > 0.95 and np.max(self.state)< 1.05):
            done = True
        if done:
            print('successful!')
        return self.state, reward, reward_sep, done
    
    def reset(self, seed=1): #sample different initial volateg conditions during training
        np.random.seed(seed)
        senario = np.random.choice([0,1])
        # senario = 0
        self.network.init_sys()
        if(senario == 0):
            # Low voltage
            bus675_a_kw = -1000*np.random.uniform(1, 5)
            bus675_b_kw = -1000*np.random.uniform(5, 10)
            bus675_c_kw = -1000*np.random.uniform(3, 7)
            self.network.run_command(f"Generator.bus675_1.kw={bus675_a_kw}") 
            self.network.run_command(f"Generator.bus675_2.kw={bus675_b_kw}") 
            self.network.run_command(f"Generator.bus675_3.kw={bus675_c_kw}") 

            bus633_a_kw = -1000*np.random.uniform(3, 10)
            bus633_b_kw = -1000*np.random.uniform(4, 8)
            bus633_c_kw = -1000*np.random.uniform(3, 7)
            self.network.run_command(f"Generator.bus633_1.kw={bus633_a_kw}") 
            self.network.run_command(f"Generator.bus633_2.kw={bus633_b_kw}") 
            self.network.run_command(f"Generator.bus633_3.kw={bus633_c_kw}") 

            bus680_a_kw = -1000*np.random.uniform(1.5, 4.5)
            bus680_b_kw = -1000*np.random.uniform(1.5, 5)
            bus680_c_kw = -1000*np.random.uniform(1.5, 5)
            self.network.run_command(f"Generator.bus680_1.kw={bus680_a_kw}") 
            self.network.run_command(f"Generator.bus680_2.kw={bus680_b_kw}") 
            self.network.run_command(f"Generator.bus680_3.kw={bus680_c_kw}") 
        if(senario == 1):
            # High voltage
            bus675_a_kw = 1000*np.random.uniform(2.5, 4.5)
            bus675_b_kw = 1000*np.random.uniform(3, 5)
            bus675_c_kw = 1000*np.random.uniform(2, 5)
            self.network.run_command(f"Generator.bus675_1.kw={bus675_a_kw}") 
            self.network.run_command(f"Generator.bus675_2.kw={bus675_b_kw}") 
            self.network.run_command(f"Generator.bus675_3.kw={bus675_c_kw}") 

            bus633_a_kw = 1000*np.random.uniform(3, 8)
            bus633_b_kw = 1000*np.random.uniform(3, 8)
            bus633_c_kw = 1000*np.random.uniform(3, 7)
            self.network.run_command(f"Generator.bus633_1.kw={bus633_a_kw}") 
            self.network.run_command(f"Generator.bus633_2.kw={bus633_b_kw}") 
            self.network.run_command(f"Generator.bus633_3.kw={bus633_c_kw}") 

            bus680_a_kw = 1000*np.random.uniform(1.5, 3)
            bus680_b_kw = 1000*np.random.uniform(2.5, 7)
            bus680_c_kw = 1000*np.random.uniform(3, 7)
            self.network.run_command(f"Generator.bus680_1.kw={bus680_a_kw}") 
            self.network.run_command(f"Generator.bus680_2.kw={bus680_b_kw}") 
            self.network.run_command(f"Generator.bus680_3.kw={bus680_c_kw}") 
        self.network.dss.Solution.Number(1)
        self.network.dss.Solution.Solve()      

        self.state=self.get_state()
        return self.state
    
if __name__ == "__main__":
    net = create_13bus3p()
    injection_bus = np.array([675,633,680])
    env = IEEE13bus3p(net, injection_bus)
    state_list = []
    # for i in range(100):
    #     state = env.reset(i)
    #     state_list.append(state)
    # state_list = np.array(state_list)
    # # print(state_list.shape)
    # fig, axs = plt.subplots(3, 3, figsize=(9,9))
    # for i in range(3):
    #     for j in range(len(injection_bus)):
    #         axs[j,i].hist(state_list[:,j,i])
    # plt.show()
    state = env.reset(0)
    for i in range(40):
        action = np.zeros((3,3))
        action += 0.1*i
        state, _,_,_ = env.step_Preward(action,action)
        state_list.append(state)
        # print(env.network.run_command('? Load.675a.kw'))
    state_list = np.array(state_list)
    fig, axs = plt.subplots(3, 3, figsize=(9,9))
    for i in range(3):
        for j in range(len(injection_bus)):
            axs[j,i].plot(range(40),state_list[:,j,i])
    plt.show()
