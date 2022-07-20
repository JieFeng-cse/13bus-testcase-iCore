from email import policy
# from multiprocessing.reduction import steal_handle
from turtle import forward
# from pyrsistent import T
from sqlalchemy import true
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import linalg as LA
import os
import random
import sys
import torch.optim as optim

# DPPG class
class DDPG:
    def __init__(self, policy_net, value_net,
                 target_policy_net, target_value_net,
                 value_lr=2e-4,
                 policy_lr=1e-4):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.policy_net = policy_net
        self.value_net = value_net
        self.target_policy_net = target_policy_net
        self.target_value_net = target_value_net
        
        self.value_lr = value_lr
        self.policy_lr = policy_lr
        
        self.value_optimizer = optim.Adam(value_net.parameters(),  lr=value_lr)
        self.policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)
        self.value_criterion = nn.MSELoss()

    def train_step(self, replay_buffer, batch_size,
                   gamma=0.99,
                   soft_tau=1e-2):

        state, action, last_action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        last_action = torch.FloatTensor(last_action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        next_action = action-self.target_policy_net(next_state)    
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + gamma*(1.0-done)*target_value
        
        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
 
        self.value_optimizer.step()
        
        # linear_action = (torch.maximum(state-1.03,torch.zeros_like(state))-torch.maximum(0.97-state,torch.zeros_like(state)))*1
        # policy_loss = self.value_criterion(self.policy_net(state),linear_action)
        policy_loss = self.value_net(state, last_action-self.policy_net(state)) 
        policy_loss =  -policy_loss.mean() #+ 0.1*self.value_criterion(self.policy_net(state),linear_action)
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        # print(f'value loss: {value_loss.cpu().detach().numpy():.4f}, policy_loss: {policy_loss.cpu().detach().numpy():.4f}')

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data*soft_tau
            )

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
    def train_step_3ph(self, replay_buffer, batch_size,
                   gamma=0.99,
                   soft_tau=1e-2):

        state, action, last_action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        last_action = torch.FloatTensor(last_action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        next_action = action-self.target_policy_net(next_state)    
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + gamma*(1.0-done)*target_value
        
        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
 
        self.value_optimizer.step()

        jacob_list = []
        
        for i in range(state.shape[0]):
            x = state[i]
            x.requires_grad = True
            action_jb = self.policy_net(x)
            jacob = torch.zeros(state.shape[1], state.shape[1]) 
            for j in range(3):
                output = torch.zeros(state.shape[1]).to(self.device)
                output[j]=1
                jacob[j,:]=torch.autograd.grad(action_jb, x, grad_outputs=output, retain_graph=True)[0]    
            jacob_list.append(jacob.unsqueeze(0))
        jacob_list = torch.cat(jacob_list,0).to(self.device)          
        # print(torch.mean(jacob_list,dim=0))                                                                          
        jacob_ii = -torch.sum(jacob_list[:,0,0]+jacob_list[:,1,1]+jacob_list[:,2,2])
        jacob_dif = 0.
        for i in range(3):
            for j in range(3):
                if  i==j:
                    jacob_dif -= torch.sum(torch.abs(jacob_list[:,i,j]))
                else:
                    jacob_dif += torch.sum(torch.abs(jacob_list[:,i,j]))
        
        policy_loss = self.value_net(state, last_action-self.policy_net(state)) 
        policy_loss = -policy_loss.mean()+ torch.norm(self.policy_net(torch.ones_like(state)),2) + 0.05*torch.exp(jacob_ii) + 0.05*torch.exp(jacob_dif)
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data*soft_tau
            )

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )


# monotone policy network with dead-band between [v_min, v_max]
class SafePolicyNetwork(nn.Module):
    def __init__(self, env, obs_dim, action_dim, hidden_dim, scale = 0.15, init_w=3e-3):
        super(SafePolicyNetwork, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device   = torch.device("cuda" if use_cuda else "cpu")

        self.env = env
        self.hidden_dim = hidden_dim
        self.scale = scale
        
        #define weight and bias recover matrix
        self.w_recover = torch.ones((self.hidden_dim, self.hidden_dim))
        self.w_recover = -torch.triu(self.w_recover, diagonal=0)\
        +torch.triu(self.w_recover, diagonal=2)+2*torch.eye(self.hidden_dim)
        self.w_recover=self.w_recover.to(self.device)
        
        self.b_recover = torch.ones((self.hidden_dim, self.hidden_dim))
        self.b_recover = torch.triu(self.b_recover, diagonal=0)-torch.eye(self.hidden_dim)
        self.b_recover = self.b_recover.to(self.device)
        
        self.select_w = torch.ones(1, self.hidden_dim).to(self.device)
        self.select_wneg = -torch.ones(1, self.hidden_dim).to(self.device)
        
        # initialization
        self.b = torch.rand(self.hidden_dim)
        self.b = (self.b/torch.sum(self.b))*scale
        self.b = torch.nn.Parameter(self.b, requires_grad=True)
        
        self.c = torch.rand(self.hidden_dim)
        self.c = (self.c/torch.sum(self.c))*scale
        self.c = torch.nn.Parameter(self.c, requires_grad=True)
        
        self.q = torch.nn.Parameter(torch.rand(action_dim, self.hidden_dim), requires_grad=True)
        self.z = torch.nn.Parameter(torch.rand(action_dim, self.hidden_dim), requires_grad=True)
        
    def forward(self, state):
        self.w_plus=torch.matmul(torch.square(self.q.double()), self.w_recover.double())
        
        self.w_minus=torch.matmul(-torch.square(self.q.double()), self.w_recover.double())
        
        b = self.b.data
        b = b.clamp(min=0).double()
        b = self.scale*b/torch.norm(b, 1)
        self.b.data = b
        
        #maybe this part is wrong?
        c = self.c.data
        c = c.clamp(min=0).double()
        c = self.scale*c/torch.norm(c, 1)
        self.c.data = c
        
        self.b_plus=torch.matmul(-self.b, self.b_recover.double()) - torch.tensor(self.env.vmax-0.02)
        self.b_minus=torch.matmul(-self.b, self.b_recover.double()) + torch.tensor(self.env.vmin+0.02)
        
        self.nonlinear_plus = torch.matmul(F.relu(torch.matmul(state.double(), self.select_w.double())
                                                  + self.b_plus.view(1, self.hidden_dim)),
                                           torch.transpose(self.w_plus, 0, 1))
        
        self.nonlinear_minus = torch.matmul(F.relu(torch.matmul(state.double(), self.select_wneg.double())
                                                   + self.b_minus.view(1, self.hidden_dim)),
                                            torch.transpose(self.w_minus, 0, 1))
        
        x = (self.nonlinear_plus+self.nonlinear_minus) 
        
        return x

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0] 

# class StablePolicy3phase(nn.Module):
#     def __init__(self, env, obs_dim, action_dim, hidden_dim, scale = 0.15, init_w=3e-3):
#         super(StablePolicy3phase, self).__init__()
#         use_cuda = torch.cuda.is_available()
#         self.device = torch.device("cuda" if use_cuda else "cpu")
#         self.env = env
#         self.hidden_dim = hidden_dim
#         self.scale = scale
#         assert action_dim==3, 'action dimension is not 3'
#         #in this code, we assume the shape of state is batch_size*3

#         #define weight and bias recover matrix
#         self.w_recover = torch.ones((3*self.hidden_dim,3*self.hidden_dim))
#         self.w_recover = -torch.triu(self.w_recover,diagonal=0)\
#             + torch.triu(self.w_recover,diagonal=4)+2*torch.eye(self.hidden_dim*3)\
#             + torch.triu(self.w_recover,diagonal=1)-torch.triu(self.w_recover,diagonal=3)
#         self.w_recover = self.w_recover.to(self.device)

#         self.b_recover = torch.zeros((3*self.hidden_dim,3*self.hidden_dim))
#         tmp = torch.ones((3*self.hidden_dim,3*self.hidden_dim))
#         for i in range(self.hidden_dim-1):
#             self.b_recover += torch.triu(tmp,diagonal=3*(i+1))-torch.triu(tmp,diagonal=3*(i+1)+1)
#         self.b_recover = self.b_recover.to(self.device) #cuda tensor is a new variable
#         self.select_w = torch.eye(3).repeat(1,self.hidden_dim).to(self.device)
#         self.select_wneg = -torch.eye(3).repeat(1,self.hidden_dim).to(self.device)

#         #initialization
#         self.b = torch.rand(self.hidden_dim*3)
#         self.b = (self.b/torch.sum(self.b))*scale
#         self.b = torch.nn.Parameter(self.b, requires_grad=True).to(self.device)
        
#         self.c = torch.rand(self.hidden_dim*3)
#         self.c = (self.c/torch.sum(self.c))*scale
#         self.c = torch.nn.Parameter(self.c, requires_grad=True).to(self.device)

#         self.lambda_pst = torch.nn.Parameter(torch.rand((self.hidden_dim, 6),device=self.device), requires_grad=True)
#         self.lambda_neg = torch.nn.Parameter(torch.rand((self.hidden_dim, 6),device=self.device), requires_grad=True)
#     def forward(self, state):
#         self.w_plus = torch.zeros(3, 3*self.hidden_dim).to(self.device)
#         for n in range(self.hidden_dim):
#             nth_dd = torch.zeros(3,3).to(self.device)
#             id = 0
#             for i in range(3):
#                 for j in range(i+1):
#                     e_i = torch.zeros(3,1).to(self.device)
#                     e_i[i] = 1
#                     e_j = torch.zeros(3,1).to(self.device)
#                     e_j[j] = 1
#                     if i == j:
#                         nth_dd += torch.matmul(e_i,torch.transpose(e_i,0,1))*torch.square(self.lambda_pst[n,id])
#                     else:
#                         nth_dd += torch.matmul(e_i-e_j,torch.transpose(e_i-e_j,0,1))*torch.square(self.lambda_pst[n,id])
#                     id += 1
#             self.w_plus[:,n*3:(n+1)*3]=nth_dd

#         self.w_plus = torch.matmul(self.w_plus, self.w_recover)
#         # self.w_minus = torch.matmul(-self.w_plus, self.w_recover) #3*3hidden
#         self.w_minus = -self.w_plus
#         # print(self.w_plus[:,0:3]+self.w_plus[:,3:6])
#         # exit(0)

#         b = self.b.data
#         b = b.clamp(min=0)
#         b = self.scale*b/torch.norm(b, 1)
#         self.b.data = b

#         self.b_plus=torch.matmul(-self.b, self.b_recover) - torch.tensor(self.env.vmax-0.02) #shape: 3*hidden_size
#         self.b_minus=torch.matmul(-self.b, self.b_recover) + torch.tensor(self.env.vmin+0.02)

#         self.nonlinear_plus = torch.matmul(F.relu(torch.matmul(state, self.select_w)
#                                                   + self.b_plus.view(1, self.hidden_dim*3)),
#                                            torch.transpose(self.w_plus, 0, 1))

#         self.nonlinear_minus = torch.matmul(F.relu(torch.matmul(state, self.select_wneg)
#                                                    + self.b_minus.view(1, self.hidden_dim*3)),
#                                             torch.transpose(self.w_minus, 0, 1))
#         x = (self.nonlinear_plus+self.nonlinear_minus) 
        
#         return x
    
#     def get_action(self, state):
#         state = torch.FloatTensor(state).to(self.device)
#         action = self.forward(state)
#         # x = state
#         # x.requires_grad = True
#         # action_jb = self.forward(x)
#         # jacob = torch.zeros(state.shape[1], state.shape[1]) 
#         # for j in range(3):
#         #     output = torch.zeros(1,state.shape[1]).to(self.device)
#         #     output[0,j]=1
#         #     jacob[j,:]=torch.autograd.grad(action_jb, x, grad_outputs=output, retain_graph=True)[0]  
#         # print(jacob)
#         # exit(0)
#         return action.detach().cpu().numpy()[0]

class StablePolicy3phase(nn.Module):
    def __init__(self, env, obs_dim, action_dim, hidden_dim, scale = 0.15, init_w=3e-3):
        super(StablePolicy3phase, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.env = env
        self.hidden_dim = hidden_dim
        self.scale = scale
        assert action_dim==3, 'action dimension is not 3'
        #in this code, we assume the shape of state is batch_size*3

        #initialization
        self.b = torch.rand(self.hidden_dim, 3)
        self.b = (self.b/torch.sum(self.b))*scale
        self.b = torch.nn.Parameter(self.b, requires_grad=True).to(self.device)
        
        self.c = torch.rand(self.hidden_dim, 3)
        self.c = (self.c/torch.sum(self.c))*scale
        self.c = torch.nn.Parameter(self.c, requires_grad=True).to(self.device)

        self.lambda_pst = torch.nn.Parameter(torch.rand((self.hidden_dim, 6),device=self.device), requires_grad=True)
        self.lambda_neg = torch.nn.Parameter(torch.rand((self.hidden_dim, 6),device=self.device), requires_grad=True)
        self.sigma = 1e-3*torch.eye(3).to(self.device)

    def forward(self, state):
        state = torch.clip(state-torch.tensor(self.env.vmax-0.02),0,float("Inf")) - torch.clip(torch.tensor(self.env.vmin+0.02)-state, 0, float("Inf"))
        self.w_plus = torch.zeros(self.hidden_dim, 3, 3).to(self.device)
        for n in range(self.hidden_dim):
            nth_dd = torch.zeros(3,3).to(self.device)
            id = 0
            for i in range(3):
                for j in range(i+1):
                    e_i = torch.zeros(3,1).to(self.device)
                    e_i[i] = 1
                    e_j = torch.zeros(3,1).to(self.device)
                    e_j[j] = 1
                    if i == j:
                        nth_dd += torch.matmul(e_i,torch.transpose(e_i,0,1))*torch.square(self.lambda_pst[n,id])
                    else:
                        nth_dd += torch.matmul(e_i-e_j,torch.transpose(e_i-e_j,0,1))*torch.square(self.lambda_pst[n,id])
                    id += 1
            self.w_plus[n,:,:] = nth_dd
        self.w_minus = -self.w_plus

        b = self.b.data
        b = b.clamp(min=0)
        b = self.scale*b/torch.norm(b, 1)
        self.b.data = b

        self.b_plus = -self.b #shape: 3*hidden_size
        self.b_minus = -self.b

        self.nonlinear_plus = torch.zeros_like(state)
        for i in range(self.hidden_dim):
            self.nonlinear_plus += F.relu(torch.matmul(state, torch.transpose(self.w_plus[i], 0, 1))+self.b_plus[i])

        self.nonlinear_minus = torch.zeros_like(state)
        for i in range(self.hidden_dim):
            self.nonlinear_minus -= F.relu(torch.matmul(state, torch.transpose(self.w_minus[i], 0, 1))+self.b_minus[i])
        x = (self.nonlinear_plus+self.nonlinear_minus) 
        x += torch.matmul(state, self.sigma)
        
        return x
    
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action = self.forward(state)
        # x = state
        # x.requires_grad = True
        # action_jb = self.forward(x)
        # jacob = torch.zeros(state.shape[1], state.shape[1]) 
        # for j in range(3):
        #     output = torch.zeros(1,state.shape[1]).to(self.device)
        #     output[0,j]=1
        #     jacob[j,:]=torch.autograd.grad(action_jb, x, grad_outputs=output, retain_graph=True)[0]  
        # print(jacob)
        # exit(0)
        return action.detach().cpu().numpy()[0]



class SafePolicy3phase(nn.Module):
    def __init__(self, env, obs_dim, action_dim, hidden_dim, bus_id, scale = 0.15, init_w=3e-3):
        super(SafePolicy3phase,self).__init__()
        use_cuda = torch.cuda.is_available()
        self.env = env
        self.bus_id = bus_id
        self.device   = torch.device("cuda" if use_cuda else "cpu")
        self.action_dim = action_dim
        action_dim_per_phase=1
        for phase in env.injection_bus[bus_id]:
            if phase == 'a':
                self.policy_a = SafePolicyNetwork(env, obs_dim, action_dim_per_phase, hidden_dim, scale = 0.15, init_w=3e-3)
            if phase == 'b':
                self.policy_b = SafePolicyNetwork(env, obs_dim, action_dim_per_phase, hidden_dim, scale = 0.15, init_w=3e-3)
            if phase == 'c':
                self.policy_c = SafePolicyNetwork(env, obs_dim, action_dim_per_phase, hidden_dim, scale = 0.15, init_w=3e-3)
        # self.policy_a = SafePolicyNetwork(env, obs_dim, action_dim, hidden_dim, scale = 0.15, init_w=3e-3)
        # self.policy_b = SafePolicyNetwork(env, obs_dim, action_dim, hidden_dim, scale = 0.15, init_w=3e-3)
        # self.policy_c = SafePolicyNetwork(env, obs_dim, action_dim, hidden_dim, scale = 0.15, init_w=3e-3)
    def forward(self, state):
        action_list = []
        for i,phase in enumerate(self.env.injection_bus[self.bus_id]):
            if phase == 'a':
                action = self.policy_a(state[:,i].unsqueeze(-1))
                action_list.append(action)
            if phase == 'b':
                action = self.policy_b(state[:,i].unsqueeze(-1))
                action_list.append(action)
            if phase == 'c':
                action = self.policy_c(state[:,i].unsqueeze(-1))
                action_list.append(action)
        # action_a = self.policy_a(state[:,0].unsqueeze(-1))
        # action_b = self.policy_b(state[:,1].unsqueeze(-1))
        # action_c = self.policy_c(state[:,2].unsqueeze(-1))
        # action = torch.cat((action_a,action_b,action_c),dim=1)
        action = torch.cat(action_list,dim=1)
        action += (torch.maximum(state-1.03, torch.zeros_like(state).to(self.device))-torch.maximum(0.97-state,  torch.zeros_like(state).to(self.device)))*0.01
        return action
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]


# standard ddpg policy network
class PolicyNetwork(nn.Module):
    def __init__(self, env, obs_dim, action_dim, hidden_dim, init_w=3e-3):
        super(PolicyNetwork, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device   = torch.device("cuda" if use_cuda else "cpu")

        self.env = env
        self.linear1 = nn.Linear(obs_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, action_dim)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        state.requires_grad = True
        # state = state-1
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]

# linear
class LinearPolicy(nn.Module):
    def __init__(self, env, ph_num, init_w=3e-3):
        super(LinearPolicy, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device   = torch.device("cuda" if use_cuda else "cpu")

        self.env = env
        slope =torch.ones(1, requires_grad=True).to(self.device)
        self.slope = torch.nn.Parameter(slope)
        self.ph_num = ph_num

    def forward(self, state):
        state.requires_grad = True
        # state = state-1
        # print(state.shape)
        x = (torch.maximum(state-1.03, torch.zeros_like(state).to(self.device))-torch.maximum(0.97-state, torch.zeros_like(state).to(self.device)))*torch.square(self.slope)
        return x

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]


# value network
class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        # self.bn2 = nn.BatchNorm1d(hidden_dim)
        # self.bn3 = nn.BatchNorm1d(1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.linear1(x))
        # x = self.bn1(x)
        x = F.relu(self.linear2(x))
        # x = self.bn2(x)
        x = self.linear3(x)
        # x = self.bn3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class ReplayBufferPI:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, last_action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, last_action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, last_action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, last_action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

