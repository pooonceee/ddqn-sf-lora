# agents/dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import itertools
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # Expanded slightly for the harder 4x4 math
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    # UPDATED: State=33, Action=1296
    def __init__(self, state_size=35, action_size=1296):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # --- NEW: Action Mapping Logic ---
        # Creates a list of all possible combinations of SF 7-12 for 4 antennas
        sf_range = [7, 8, 9, 10, 11, 12]
        self.action_mapping = list(itertools.product(sf_range, repeat=4))
        
        # --- Models ---
        self.model = QNetwork(state_size, action_size)
        self.target_model = QNetwork(state_size, action_size)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def get_action_format(self, action_index):
        """Converts the integer index (0-1295) to the antenna dict format"""
        sfs = self.action_mapping[action_index]
        return {
            "sf_ant1": sfs[0],
            "sf_ant2": sfs[1],
            "sf_ant3": sfs[2],
            "sf_ant4": sfs[3]
        }

    def update_target_model(self):
        """Copies weights from the learning model to the stable target model"""
        self.target_model.load_state_dict(self.model.state_dict())
                
    def get_action_format(self, action_index):
        sf1, sf2, sf3, sf4 = self.action_mapping[action_index]
        return {"sf_ant1": sf1, "sf_ant2": sf2, "sf_ant3": sf3, "sf_ant4": sf4}

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) 
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values[0]).item()
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def save(self, name):
        """Saves the model weights"""
        torch.save(self.model.state_dict(), name)
        print(f"--- Model saved as {name} ---")

    def load(self, name):
        """Loads the model weights"""
        self.model.load_state_dict(torch.load(name))
        self.update_target_model()
        print(f"--- Model {name} loaded ---")
        
    def replay(self, batch_size):
        # 1. Sample a random batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            # Convert numpy arrays to PyTorch tensors
            state_t = torch.FloatTensor(state).unsqueeze(0)
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
            reward_t = torch.tensor([reward], dtype=torch.float32)
            
            # 2. Get current Q values for the state
            target_q = self.model(state_t).detach()
            
            if done:
                target_value = reward_t
            else:
                # --- DOUBLE DQN LOGIC ---
                # A. Use 'online' model to pick the best ACTION for next_state
                next_actions = self.model(next_state_t).detach()
                best_next_action = torch.argmax(next_actions, dim=1)
                
                # B. Use 'target' model to get the Q-VALUE of that action
                target_next_q = self.target_model(next_state_t).detach()
                target_value = reward_t + self.gamma * target_next_q[0][best_next_action]
            
            # 3. Update the specific action's Q-value in the target
            target_q[0][action] = target_value
            
            # 4. Perform a single gradient descent step
            self.optimizer.zero_grad()
            output = self.model(state_t)
            loss = nn.MSELoss()(output, target_q)
            loss.backward()
            self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay