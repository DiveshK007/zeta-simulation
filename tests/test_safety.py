"""Unit tests for ZETA safety module."""

import unittest
import numpy as np
from zeta.safety import SafetyMonitor, CollisionDetector

class TestSafety(unittest.TestCase):
    def setUp(self):
        self.safety = SafetyMonitor()
        self.collision = CollisionDetector()
    
    def test_collision_detection(self):
        """Test collision detection boundaries."""
        # Test point within bounds
        self.assertTrue(
            self.collision.check(np.array([0.0, 0.0, 0.5]))
        )
        
        # Test point outside bounds
        self.assertFalse(
            self.collision.check(np.array([2.0, 0.0, 0.5]))
        )
    
    def test_motion_safety(self):
        """Test trajectory safety checks."""
        # Safe trajectory
        safe_traj = [
            np.array([0.0, 0.0, 0.5]),
            np.array([0.1, 0.0, 0.5]),
            np.array([0.2, 0.0, 0.5])
        ]
        safe, _ = self.safety.check_motion_safety(safe_traj)
        self.assertTrue(safe)
        
        # Unsafe trajectory (exceeds workspace)
        unsafe_traj = [
            np.array([0.0, 0.0, 0.5]),
            np.array([2.0, 0.0, 0.5])
        ]
        safe, reason = self.safety.check_motion_safety(unsafe_traj)
        self.assertFalse(safe)
        self.assertIn("workspace bounds", reason)
