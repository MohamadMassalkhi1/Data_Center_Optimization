import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, render_template, request
from collections import deque
import random
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
import os

# Ensure static folder exists
if not os.path.exists('static'):
    os.makedirs('static')

# Data Center Environment
class DataCenterEnv:
    """Simulates a data center, optimizing cooling for energy, water, and temperature."""
    def __init__(self, num_servers=10, max_power=700):
        self.num_servers = num_servers
        self.max_power = max_power  # Watts per GPU
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

# Replay Buffer
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
        self.gamma = gamma
        self.epsilon = epsilon
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
        """Select an action based on exploration or learned policy."""
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
def train_agent(num_servers=10, max_power=700, episodes=500, steps=100):
    """Train the AI with custom number of GPUs and GPU power."""
    env = DataCenterEnv(num_servers, max_power)
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
def evaluate_agent(agent, num_servers=10, max_power=700, steps=100):
    """Evaluate the trained AI with custom parameters."""
    env = DataCenterEnv(num_servers, max_power)
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
    
    # Generate plots
    plt.figure(figsize=(8, 4))
    plt.plot(temps, label="Temperature")
    plt.axhline(y=80, color='r', linestyle='--', label="Max Safe Temp (80°C)")
    plt.title("Temperature Control (Heat Management)")
    plt.xlabel("Time Step")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)
    plt.savefig("static/eval_temps.png")
    plt.close()
    
    plt.figure(figsize=(8, 4))
    plt.plot(water_usages)
    plt.title("Water Usage Optimization")
    plt.xlabel("Time Step")
    plt.ylabel("Water Usage (Liters)")
    plt.grid(True)
    plt.savefig("static/eval_water_usage.png")
    plt.close()
    
    action_counts = np.bincount(actions, minlength=3)
    return total_reward, sum(water_usages), action_counts.tolist()

# Flask App
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the main page with input form and results."""
    results = None
    if request.method == 'POST':
        try:
            # Get user inputs
            num_servers = int(request.form.get('num_servers', 10))
            gpu_type = request.form.get('gpu_type', 'H100')
            
            # Map GPU type to max_power
            gpu_power_map = {'H100': 700, 'A100': 400, 'V100': 300}
            max_power = gpu_power_map.get(gpu_type, 700)  # Default to H100
            
            # Validate inputs
            if num_servers < 1:
                return render_template('index.html', results={'error': 'Number of GPUs must be at least 1.'})
            
            # Train and evaluate
            agent, _, _ = train_agent(num_servers, max_power)
            total_reward, total_water, action_counts = evaluate_agent(agent, num_servers, max_power)
            
            results = {
                'num_servers': num_servers,
                'gpu_type': gpu_type,
                'reward': round(total_reward, 2),
                'water': round(total_water, 2),
                'actions': {
                    'Air (High Energy, Low Water)': action_counts[0],
                    'Liquid (Medium Energy, High Water)': action_counts[1],
                    'Immersion (Low Energy, Low Water)': action_counts[2]
                },
                'temp_plot': 'static/eval_temps.png',
                'water_plot': 'static/eval_water_usage.png',
                'cache_buster': str(random.randint(1, 1000000))  # Simple cache-busting
            }
        except Exception as e:
            results = {'error': f'Error during training/evaluation: {str(e)}'}
    
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)