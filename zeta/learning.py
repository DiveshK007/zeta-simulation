"""Adaptive learning module for ZETA."""

from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class ExperienceBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: np.ndarray, 
            reward: float, next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample random batch of experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)

class ActionValueNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class AdaptiveLearningModule:
    def __init__(self, state_dim: int, action_dim: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = ActionValueNetwork(state_dim, action_dim).to(self.device)
        self.target_net = ActionValueNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Training parameters
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.criterion = nn.MSELoss()
        self.batch_size = 64
        self.gamma = 0.99  # Discount factor
        self.tau = 0.001   # Target network update rate
        
        # Experience replay
        self.experience_buffer = ExperienceBuffer()
        
        # Metrics
        self.metrics = {
            'losses': [],
            'rewards': [],
            'episodes': 0
        }
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
        """Select action using epsilon-greedy policy."""
        if random.random() < epsilon:
            return np.random.randn(self.policy_net.network[-1].out_features)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_net(state_tensor).cpu().numpy()[0]
    
    def update_model(self, state: np.ndarray, action: np.ndarray, 
                    reward: float, next_state: np.ndarray, done: bool):
        """Update model with new experience."""
        # Store experience
        self.experience_buffer.add(state, action, reward, next_state, done)
        
        # Train if enough samples
        if len(self.experience_buffer) >= self.batch_size:
            self.train_step()
        
        # Update metrics
        self.metrics['rewards'].append(reward)
        if done:
            self.metrics['episodes'] += 1
    
    def train_step(self):
        """Perform one training step."""
        # Sample batch
        batch = self.experience_buffer.sample(self.batch_size)
        state_batch = torch.FloatTensor([x[0] for x in batch]).to(self.device)
        action_batch = torch.FloatTensor([x[1] for x in batch]).to(self.device)
        reward_batch = torch.FloatTensor([x[2] for x in batch]).to(self.device)
        next_state_batch = torch.FloatTensor([x[3] for x in batch]).to(self.device)
        done_batch = torch.FloatTensor([x[4] for x in batch]).to(self.device)
        
        # Compute Q values
        current_q_values = self.policy_net(state_batch)
        next_q_values = self.target_net(next_state_batch).detach()
        
        # Compute expected Q values
        expected_q_values = reward_batch + self.gamma * (1 - done_batch) * next_q_values.max(1)[0]
        
        # Compute loss
        loss = self.criterion(current_q_values, expected_q_values.unsqueeze(1))
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        for target_param, policy_param in zip(
            self.target_net.parameters(), 
            self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )
        
        # Update metrics
        self.metrics['losses'].append(loss.item())
    
    def get_metrics(self) -> Dict:
        """Get current training metrics."""
        return {
            'avg_loss': np.mean(self.metrics['losses'][-100:]) if self.metrics['losses'] else 0,
            'avg_reward': np.mean(self.metrics['rewards'][-100:]) if self.metrics['rewards'] else 0,
            'total_episodes': self.metrics['episodes']
        }
    
    def save_model(self, path: str):
        """Save model state."""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics
        }, path)
    
    def load_model(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.metrics = checkpoint['metrics']
