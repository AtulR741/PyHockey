import torch as T
import numpy as np
from model import ActorNetwork, CriticNetork, PPOMemory
from main import get_state, init, gameInit, play_step

N = 2048
device = T.device('cuda')

class Agent:
    def __init__(self, folder, input_dims, n_actions, gamma = 0.9, alpha = 0.0003, gae_lambda = 0.95,
                 policy_clip = 0.2, batch_size = 16, n_epochs = 5):
        self.actor = ActorNetwork(input_dims, n_actions, folder, alpha)
        self.critic = CriticNetork(input_dims, folder, alpha)
        self.memory = PPOMemory(batch_size)
        
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs

    def remember(self, state, action, prob, val, reward, done):
        self.memory.store_memory(state, action, prob, val, reward, done)
    
    def save_model(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
    
    def load_model(self, flag):
        self.actor.load_checkpoint(flag)
        self.critic.load_checkpoint(flag)
    
    def get_action(self, obs):
        state = T.from_numpy(obs).unsqueeze(0).float().to(device)
        dist = self.actor(state)
        action = dist.sample()
        prob = dist.log_prob(action).sum(dim=-1)
        
        value = self.critic(state)

        return action.squeeze(0).cpu().numpy(), prob.item(), value.squeeze(0).item()
    
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr, reward_arr,\
            done_arr, batches = self.memory.generate_batches()
        
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount*(reward_arr[k] + self.gamma*vals_arr[k + 1]*\
                    (1 - int(done_arr[k])) - vals_arr[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(device)

            vals_arr = T.tensor(vals_arr).to(device)

            for batch in batches:
                states = T.tensor(state_arr[batch]).to(device)
                old_probs = T.tensor(old_probs_arr[batch]).to(device)
                actions = T.tensor(action_arr[batch]).to(device)

                dist = self.actor(states)
                
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions).sum(dim=-1)
                prob_ratio = new_probs.exp() / old_probs.exp()

                weighted_probs = prob_ratio*advantage[batch]
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,\
                1 + self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + vals_arr[batch]
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        self.memory.clear_memory()

def train():
    agent = Agent('model', 6, 2)
    agent.load_model(True)
    gameInit(-1)
    n_steps, avg, bestAvg, alpha = 0, 0, 0, 0.1
    while True:
        cpu_state = get_state()
        cpu_action, cpuProb, cpuValue = agent.get_action(cpu_state)

        cpuReward, done, score = play_step(cpu_action)
        if score == 1:
            avg = alpha + (1 - alpha)*avg
        elif score == -1:
            avg = -alpha + (1 - alpha)*avg
        n_steps += 1

        # remember
        agent.remember(cpu_state, cpu_action, cpuProb, cpuValue, cpuReward, done)

        #learn
        if n_steps % N == 0:
            agent.learn()

        # episode ends
        if done:
            if avg > bestAvg:
                bestAvg = avg
                agent.save_model()
            init(score)

def test():
    agent = Agent('model', 6, 2)
    agent.load_model(True)
    gameInit(-1)
    while True:
        cpu_state = get_state()
        cpu_action, x, y = agent.get_action(cpu_state)

        x, done, point = play_step(cpu_action)

        if done:
            init(point)

if __name__ == '__main__':
    test()