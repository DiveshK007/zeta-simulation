"""Unit tests for ZETA world model."""

import unittest
import numpy as np
from zeta.world_model import WorldModel, ObjectState, SceneGraph

class TestWorldModel(unittest.TestCase):
    def setUp(self):
        self.world_model = WorldModel()
    
    def test_scene_graph(self):
        """Test scene graph updates and queries."""
        # Add test objects
        obj1 = ObjectState(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0])
        )
        obj2 = ObjectState(
            position=np.array([0.0, 0.0, 0.2]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0])
        )
        
        observations = {
            'cube1': obj1,
            'cube2': obj2
        }
        
        # Update world model
        self.world_model.update(observations)
        
        # Check object states
        self.assertEqual(
            len(self.world_model.object_states),
            2
        )
        
        # Check object states
        state1 = self.world_model.get_object_state('cube1')
        state2 = self.world_model.get_object_state('cube2')
        print(f"State 1 pos: {state1.position}")
        print(f"State 2 pos: {state2.position}")
        
        # Check spatial relations
        relations = self.world_model.get_spatial_relations('cube2')
        print(f"Relations for cube2: {relations}")
        self.assertTrue('on_top_of:cube1' in relations)
