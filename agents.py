import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from networks import DuelingDQN, PPOActorCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 高效经验回放池 (解决内存溢出) ---
class ReplayBuffer:
    def __init__(self, capacity, state_shape, action_dim, device):
        self.device = device
        # 使用 uint8 存储图像，节省 4 倍内存
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        self.capacity = capacity
        self.idx = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done
        
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.tensor(self.states[idxs], device=self.device), # uint8 传给 GPU
            torch.tensor(self.actions[idxs], device=self.device),
            torch.tensor(self.rewards[idxs], device=self.device),
            torch.tensor(self.next_states[idxs], device=self.device), # uint8 传给 GPU
            torch.tensor(self.dones[idxs], device=self.device)
        )

# --- Agent 1: Dueling Double DQN ---
class DuelingDDQNAgent:
    def __init__(self, state_shape, action_dim, lr=1e-4, gamma=0.99, buffer_size=100000):
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99998  # 稍微减缓衰减
        
        self.online_net = DuelingDQN(state_shape, action_dim).to(device)
        self.target_net = DuelingDQN(state_shape, action_dim).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        # 使用自定义的高效 Buffer
        self.memory = ReplayBuffer(buffer_size, state_shape, action_dim, device)
        self.batch_size = 64 # 增大 Batch Size 以利用 GPU

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            # state 是 (H, W, C) 或 (C, H, W)，确保增加 batch 维度
            # 注意：传入 uint8，网络内部会处理归一化
            state_tensor = torch.tensor(np.array(state), device=device, dtype=torch.uint8).unsqueeze(0)
            q_values = self.online_net(state_tensor)
            return q_values.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update(self):
        if self.memory.size < self.batch_size * 5: # 等待一定数据量再开始
            return None
        
        # 采样已经在 GPU 上的 Tensor
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Double DQN Logic
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        curr_q_values = self.online_net(states).gather(1, actions)
        
        loss = nn.MSELoss()(curr_q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

# --- Agent 2: PPO ---
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_coef=0.2, ent_coef=0.0):
        self.network = PPOActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.gamma = gamma
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.gae_lambda = 0.95

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action, log_prob, _, value = self.network.get_action_and_value(state)
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]

    def update(self, rollouts):
        states = torch.FloatTensor(np.array(rollouts['states'])).to(device)
        actions = torch.FloatTensor(np.array(rollouts['actions'])).to(device)
        log_probs = torch.FloatTensor(np.array(rollouts['log_probs'])).to(device)
        rewards = torch.FloatTensor(np.array(rollouts['rewards'])).to(device)
        dones = torch.FloatTensor(np.array(rollouts['dones'])).to(device)
        values = torch.FloatTensor(np.array(rollouts['values'])).flatten().to(device)

        # GAE Calculation (全是 Tensor 操作)
        with torch.no_grad():
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    nextnonterminal = 1.0 - dones[t]
                    # 这里简化处理，实际上应该取 next_value
                    nextvalues = values[t] 
                else:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = values[t+1]
                
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + values

        # PPO Update Loop
        b_inds = np.arange(len(states))
        batch_size = 64
        clipfracs = []
        
        for _ in range(10): 
            np.random.shuffle(b_inds)
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.network.get_action_and_value(states[mb_inds], actions[mb_inds])
                logratio = newlogprob - log_probs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = advantages[mb_inds]
                # Normalize advantages
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                v_loss = 0.5 * ((newvalue.view(-1) - returns[mb_inds]) ** 2).mean()

                loss = pg_loss + v_loss * 0.5 - entropy.mean() * self.ent_coef

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return loss.item()