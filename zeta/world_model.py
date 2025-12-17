"""World modeling and scene understanding for ZETA."""

from typing import Dict, List, Optional, Set
import numpy as np
from dataclasses import dataclass

@dataclass
class ObjectState:
    position: np.ndarray
    orientation: np.ndarray
    velocity: Optional[np.ndarray] = None
    category: Optional[str] = None
    attributes: Dict[str, any] = None

class SceneGraph:
    def __init__(self):
        self.objects: Dict[str, ObjectState] = {}
        self.relations: Dict[str, Set[str]] = {}
        
    def add_object(self, obj_id: str, state: ObjectState):
        """Add or update object in the scene."""
        self.objects[obj_id] = state
        if obj_id not in self.relations:
            self.relations[obj_id] = set()
    
    def add_relation(self, obj1: str, obj2: str, relation: str):
        """Add spatial/semantic relation between objects."""
        if obj1 not in self.objects or obj2 not in self.objects:
            raise ValueError("Objects must exist in scene")
        
        if obj1 not in self.relations:
            self.relations[obj1] = set()
        self.relations[obj1].add(f"{relation}:{obj2}")
    
    def update(self, observations: Dict[str, ObjectState]):
        """Update scene graph with new observations."""
        for obj_id, state in observations.items():
            self.add_object(obj_id, state)
        
        # Update spatial relations
        self.update_spatial_relations()
    
    def update_spatial_relations(self):
        """Update spatial relations between objects."""
        for obj1_id, obj1_state in self.objects.items():
            for obj2_id, obj2_state in self.objects.items():
                if obj1_id != obj2_id:
                    # Check spatial relations from obj2's perspective relative to obj1
                    rel = self.compute_spatial_relation(
                        obj1_state.position,
                        obj2_state.position
                    )
                    if rel:
                        # obj2 has the relation to obj1
                        self.add_relation(obj2_id, obj1_id, rel)
    
    def compute_spatial_relation(self, pos1: np.ndarray, pos2: np.ndarray) -> Optional[str]:
        """Compute spatial relation between two positions."""
        diff = pos2 - pos1
        
        # We care more about vertical relations
        if abs(diff[2]) > 0.1:  # Vertical difference
            return "on_top_of" if diff[2] > 0 else "below"
        
        # For horizontal relations, check if objects are close
        horizontal_dist = np.linalg.norm(diff[:2])
        if horizontal_dist < 0.2:  # Within 20cm horizontally
            return "next_to"
        
        return None

class SpatialRelationTracker:
    def __init__(self):
        self.relations: Dict[str, List[str]] = {}
        self.history: List[Dict[str, List[str]]] = []
    
    def update(self, scene_graph: SceneGraph):
        """Update spatial relations from scene graph."""
        current_relations = {}
        
        for obj_id, relations in scene_graph.relations.items():
            current_relations[obj_id] = list(relations)
        
        self.relations = current_relations
        self.history.append(current_relations.copy())
        
        # Keep only last 100 frames of history
        if len(self.history) > 100:
            self.history.pop(0)
    
    def get_relation_history(self, obj_id: str) -> List[List[str]]:
        """Get history of relations for an object."""
        return [frame.get(obj_id, []) for frame in self.history]

class WorldModel:
    def __init__(self):
        self.scene_graph = SceneGraph()
        self.spatial_tracker = SpatialRelationTracker()
        self.object_states: Dict[str, ObjectState] = {}
    
    def update(self, observations: Dict[str, ObjectState]):
        """Update world model with new observations."""
        # Update scene graph
        self.scene_graph.update(observations)
        
        # Update spatial relations
        self.spatial_tracker.update(self.scene_graph)
        
        # Update object states
        self.object_states.update(observations)
    
    def get_object_state(self, obj_id: str) -> Optional[ObjectState]:
        """Get current state of an object."""
        return self.object_states.get(obj_id)
    
    def get_spatial_relations(self, obj_id: str) -> List[str]:
        """Get current spatial relations for an object."""
        return self.spatial_tracker.relations.get(obj_id, [])
