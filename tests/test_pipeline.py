"""Unit tests for ZETA pipeline."""

import unittest
import numpy as np
import torch
from zeta.pipeline import ZETAPipeline, PerceptionModule, PlanningModule
from zeta.world_model import ObjectState

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = ZETAPipeline()
    
    def test_perception_module(self):
        """Test perception module processing."""
        perception = PerceptionModule()
        
        # Create dummy image
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Test processing
        output = perception.process(image)
        
        self.assertIsNotNone(output)
        self.assertIsInstance(output.objects, dict)
        self.assertGreaterEqual(output.confidence, 0.0)
        self.assertLessEqual(output.confidence, 1.0)
    
    def test_planning_module(self):
        """Test planning module."""
        planning = PlanningModule()
        
        # Create test objects
        objects = {
            'cube1': ObjectState(
                position=np.array([0.0, 0.0, 0.0]),
                orientation=np.array([0.0, 0.0, 0.0, 1.0])
            )
        }
        
        # Test plan generation
        plan = planning.generate(
            instruction="pick up the cube",
            percepts=objects
        )
        
        self.assertIsInstance(plan, list)
        if plan:  # If plan is generated
            self.assertTrue(all(hasattr(step, 'action') for step in plan))
    
    def test_complete_pipeline(self):
        """Test complete pipeline execution."""
        # Create test inputs
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        instruction = "pick up the red cube"
        
        # Run pipeline
        success = self.pipeline.process(instruction, image)
        
        # Check result type
        self.assertIsInstance(success, bool)

class TestErrorHandling(unittest.TestCase):
    """Test error handling in pipeline."""
    
    def setUp(self):
        self.pipeline = ZETAPipeline()
    
    def test_invalid_instruction(self):
        """Test handling of invalid instruction."""
        with self.assertRaises(ValueError):
            self.pipeline.process("", np.zeros((224, 224, 3)))
    
    def test_invalid_image(self):
        """Test handling of invalid image."""
        with self.assertRaises(ValueError):
            self.pipeline.process("pick up cube", None)
    
    def test_recovery_from_failure(self):
        """Test system recovery from execution failure."""
        # Simulate failed execution
        self.pipeline.modules['execution'].robot = None  # No robot available
        
        # Should handle gracefully
        success = self.pipeline.process(
            "pick up cube",
            np.zeros((224, 224, 3))
        )
        
        self.assertFalse(success)  # Should fail gracefully
