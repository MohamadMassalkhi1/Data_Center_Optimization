import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

# Data Center Environment
class DataCenterEnv:
    """Simulates a data center with 10 GPUs, optimizing cooling for energy, water, and temperature."""
    def __init__(self):
        self.num_servers = 10
        self.max_power = 700  # Watts per GPU
        self.temp_threshold = 80  # Max safe temperature (°C)
        self.current_temp = 40.0
        self.cooling_rates = [5.0, 8.0, 12.0]  # Cooling effect (air, liquid, immersion)
        self.energy_costs = [0.08, 0.05, 0.03]  # $/kWh
        self.water_usages = [0.1, 1.0, 0.01]  # Liters/kWh

    def reset(self):
        """Reset temperature to 40°C and return initial state."""
        self.current_temp = 40.0
        return self.get_state(0.5)

    def get_state(self, load):
        """Convert temperature and load to a discrete state (0-8)."""
        temp_bin = 0 if self.current_temp < 60 else 1 if self.current_temp < 70 else 2
        load_bin = 0 if load < 0.33 else 1 if load < 0.66 else 2
        return temp_bin * 3 + load_bin

    def step(self, action, load):
        """Apply cooling action, update temperature, and calculate reward."""
        heat = self.num_servers * self.max_power * load * 0.005
        cooling = self.cooling_rates[action] * (1 + np.random.normal(0, 0.1))
        self.current_temp = max(20, min(100, self.current_temp + heat - cooling))
        
        energy_cost = self.energy_costs[action] * self.num_servers * self.max_power * load / 1000
        water_usage = self.water_usages[action] * self.num_servers * self.max_power * load / 1000
        reward = -energy_cost * 2 - water_usage
        if self.current_temp > self.temp_threshold:
            reward -= 100
        
        done = self.current_temp > 90
        return self.get_state(load), reward, done, water_usage

# Replay Buffer for Experience Storage
class ReplayBuffer:
    """Stores past experiences for training the AI."""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# Neural Network for DQN
class DQN(nn.Module):
    """Neural network to predict the best cooling action."""
    def __init__(self, num_states=9, num_actions=3):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(num_states, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions)
        )
    
    def forward(self, x):
        return self.network(x)

# DQN Agent
class DQNAgent:
    """AI agent that learns to control cooling using reinforcement learning."""
    def __init__(self, num_states=9, num_actions=3, lr=0.001, gamma=0.99, epsilon=1.0):
        self.num_actions = num_actions
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = epsilon  # Exploration rate
        self.batch_size = 32
        
        self.policy_net = DQN(num_states, num_actions)
        self.target_net = DQN(num_states, num_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer()

    def state_to_tensor(self, state):
        """Convert state to one-hot tensor."""
        one_hot = torch.zeros(9)
        one_hot[state] = 1.0
        return one_hot.unsqueeze(0)

    def choose_action(self, state):
        """Select an action (cooling method) based on exploration or learned policy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        with torch.no_grad():
            q_values = self.policy_net(self.state_to_tensor(state))
        return q_values.argmax().item()

    def update(self):
        """Train the neural network using past experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        state_tensors = torch.cat([self.state_to_tensor(s) for s in states])
        next_state_tensors = torch.cat([self.state_to_tensor(s) for s in next_states])
        actions = torch.tensor(actions).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        q_values = self.policy_net(state_tensors).gather(1, actions).squeeze()
        with torch.no_grad():
            next_q = self.target_net(next_state_tensors).max(1)[0]
            targets = rewards + self.gamma * next_q * (1 - dones)
        
        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        """Reduce exploration over time."""
        self.epsilon = max(0.1, self.epsilon * 0.999)

# Training Function
def train_agent(episodes=500, steps=100):
    """Train the AI to optimize cooling."""
    env = DataCenterEnv()
    agent = DQNAgent()
    rewards = []
    water_usages = []
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        total_water = 0
        for _ in range(steps):
            load = random.uniform(0, 1)
            action = agent.choose_action(state)
            next_state, reward, done, water_usage = env.step(action, load)
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            agent.decay_epsilon()
            state = next_state
            total_reward += reward
            total_water += water_usage
            if done:
                break
        rewards.append(total_reward)
        water_usages.append(total_water)
        if ep % 50 == 0:
            print(f"Episode {ep}: Reward = {total_reward:.2f}, Water Usage = {total_water:.2f} liters")
    
    torch.save(agent.policy_net.state_dict(), 'cooling_dqn.pth')
    return agent, rewards, water_usages

# Evaluation Function
def evaluate_agent(agent, steps=100):
    """Test the trained AI on the data center environment."""
    env = DataCenterEnv()
    state = env.reset()
    total_reward = 0
    temps = [env.current_temp]
    water_usages = []
    actions = []
    
    for _ in range(steps):
        load = random.uniform(0, 1)
        action = agent.choose_action(state)
        next_state, reward, done, water_usage = env.step(action, load)
        state = next_state
        total_reward += reward
        temps.append(env.current_temp)
        water_usages.append(water_usage)
        actions.append(action)
        if done:
            break
    
    print(f"Evaluation: Reward = {total_reward:.2f}, Total Water = {sum(water_usages):.2f} liters")
    print("Action Counts (Air=0, Liquid=1, Immersion=2):", np.bincount(actions, minlength=3))
    return temps, water_usages

# Run Training and Evaluation
if __name__ == "__main__":
    print("Training AI to optimize data center cooling...")
    agent, rewards, water_usages = train_agent(episodes=500)
    
    # Plot training results
    plt.figure(figsize=(8, 4))
    plt.plot(rewards)
    plt.title("Training Progress: Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.savefig("training_rewards.png")
    plt.show()
    
    plt.figure(figsize=(8, 4))
    plt.plot(water_usages)
    plt.title("Training Progress: Water Usage")
    plt.xlabel("Episode")
    plt.ylabel("Water Usage (Liters)")
    plt.grid(True)
    plt.savefig("training_water_usage.png")
    plt.show()
    
    # Evaluate the trained agent
    print("\nTesting the trained AI...")
    temps, water_usages = evaluate_agent(agent)
    
    # Plot evaluation results
    plt.figure(figsize=(8, 4))
    plt.plot(temps, label="Temperature")
    plt.axhline(y=80, color='r', linestyle='--', label="Max Safe Temp")
    plt.title("Temperature During Testing")
    plt.xlabel("Time Step")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)
    plt.savefig("eval_temps.png")
    plt.show()
    
    plt.figure(figsize=(8, 4))
    plt.plot(water_usages)
    plt.title("Water Usage During Testing")
    plt.xlabel("Time Step")
    plt.ylabel("Water Usage (Liters)")
    plt.grid(True)
    plt.savefig("eval_water_usage.png")
    plt.show()