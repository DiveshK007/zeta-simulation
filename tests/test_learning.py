"""Unit tests for ZETA learning module."""

import unittest
import numpy as np
import torch
from zeta.learning import AdaptiveLearningModule

class TestLearning(unittest.TestCase):
    def setUp(self):
        self.learning = AdaptiveLearningModule(
            state_dim=4,
            action_dim=2
        )
    
    def test_action_selection(self):
        """Test action selection logic."""
        state = np.random.randn(4)
        
        # Test deterministic action
        action = self.learning.select_action(state, epsilon=0.0)
        self.assertEqual(action.shape, (2,))
        
        # Test random action
        action = self.learning.select_action(state, epsilon=1.0)
        self.assertEqual(action.shape, (2,))
    
    def test_experience_buffer(self):
        """Test experience replay buffer."""
        state = np.random.randn(4)
        action = np.random.randn(2)
        next_state = np.random.randn(4)
        
        # Add experience
        self.learning.update_model(
            state=state,
            action=action,
            reward=1.0,
            next_state=next_state,
            done=False
        )
        
        # Check buffer
        self.assertEqual(len(self.learning.experience_buffer), 1)
    
    def test_model_saving_loading(self):
        """Test model persistence."""
        import tempfile
        import os
        
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            path = tmp.name
        
        try:
            # Save model
            self.learning.save_model(path)
            
            # Create new instance
            new_learning = AdaptiveLearningModule(
                state_dim=4,
                action_dim=2
            )
            
            # Load model
            new_learning.load_model(path)
            
            # Compare weights
            for p1, p2 in zip(
                self.learning.policy_net.parameters(),
                new_learning.policy_net.parameters()
            ):
                self.assertTrue(torch.all(torch.eq(p1, p2)))
        
        finally:
            # Cleanup
            if os.path.exists(path):
                os.remove(path)
    
    def test_metrics_tracking(self):
        """Test metrics collection."""
        # Generate some experiences
        for _ in range(10):
            state = np.random.randn(4)
            action = np.random.randn(2)
            next_state = np.random.randn(4)
            
            self.learning.update_model(
                state=state,
                action=action,
                reward=1.0,
                next_state=next_state,
                done=False
            )
        
        # Check metrics
        metrics = self.learning.get_metrics()
        self.assertIn('avg_reward', metrics)
        self.assertIn('avg_loss', metrics)
        self.assertIn('total_episodes', metrics)
