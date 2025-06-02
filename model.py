import torch as T
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from torch.distributions import Normal

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
    
    def generate_batches(self):
        num_states = len(self.states)
        start = np.arange(0, num_states, self.batch_size)
        indices = np.arange(num_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in start]

        return np.array(self.states), np.array(self.actions),\
        np.array(self.probs), np.array(self.vals),\
        np.array(self.rewards), np.array(self.dones), batches
    
    def store_memory(self, state, action, prob, val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, folder, lr, hidden_dim = 64):
        super().__init__()
        self.checkpoint_file = os.path.join(folder, 'actor')
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Parameter(T.zeros(output_dim))

        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.device = T.device('cuda')
        self.to(self.device)
    
    def forward(self, state):
        features = self.actor(state)
        mu = self.mu_layer(features)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self, flag):
        self.load_state_dict(T.load(self.checkpoint_file, weights_only = flag))
    
class CriticNetork(nn.Module):
    def __init__(self, input_dims, folder, lr, hidden_dim = 64):
        super().__init__()
        self.checkpoint_file = os.path.join(folder, 'critic')
        self.critic = nn.Sequential(
            nn.Linear(input_dims, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.device = T.device('cuda')
        self.to(self.device)
    
    def forward(self, state):
        return self.critic(state)
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self, flag):
        self.load_state_dict(T.load(self.checkpoint_file, weights_only = flag))
    