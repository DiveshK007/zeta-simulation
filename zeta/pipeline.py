"""Modular pipeline system for ZETA."""

from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass

from .world_model import WorldModel, ObjectState
from .safety import SafetyMonitor

@dataclass
class PerceptionOutput:
    objects: Dict[str, ObjectState]
    scene_embedding: np.ndarray
    confidence: float

@dataclass
class PlanStep:
    action: str
    params: Dict[str, any]
    constraints: Dict[str, any]

class PerceptionModule:
    def __init__(self):
        self.object_detector = None  # YOLOv8 detector
        self.scene_embedder = None   # CLIP embedder
        
    def process(self, image: np.ndarray) -> PerceptionOutput:
        """Process camera image and return scene understanding."""
        # Object detection
        objects = {}
        if self.object_detector:
            detections = self.object_detector(image)
            for det in detections:
                obj_id = f"obj_{len(objects)}"
                objects[obj_id] = ObjectState(
                    position=det.position,
                    orientation=det.orientation,
                    category=det.class_name
                )
        
        # Scene embedding
        scene_embedding = None
        if self.scene_embedder:
            scene_embedding = self.scene_embedder(image)
        
        return PerceptionOutput(
            objects=objects,
            scene_embedding=scene_embedding,
            confidence=0.9  # TODO: Calculate actual confidence
        )

class PlanningModule:
    def __init__(self):
        self.world_model = WorldModel()
        self.safety_monitor = SafetyMonitor()
    
    def generate(self, instruction: str, percepts: PerceptionOutput) -> List[PlanStep]:
        """Generate execution plan from instruction and perception."""
        # Update world model
        self.world_model.update(percepts.objects)
        
        # TODO: Use LLM to generate plan
        plan = []
        
        # Validate plan safety
        for step in plan:
            if not self.validate_step_safety(step):
                raise ValueError(f"Unsafe plan step: {step}")
        
        return plan
    
    def validate_step_safety(self, step: PlanStep) -> bool:
        """Check if plan step is safe."""
        if step.action == "move_to":
            trajectory = step.params.get("trajectory")
            if trajectory:
                safe, reason = self.safety_monitor.check_motion_safety(trajectory)
                if not safe:
                    print(f"Unsafe motion: {reason}")
                    return False
        return True

class ExecutionModule:
    def __init__(self):
        self.robot = None  # Robot interface
        self.world_model = WorldModel()
        self.safety_monitor = SafetyMonitor()
    
    def run(self, plan: List[PlanStep]) -> bool:
        """Execute plan steps while monitoring safety."""
        for step in plan:
            # Pre-execution safety check
            if not self.safety_monitor.check_motion_safety(self.get_step_trajectory(step)):
                return False
            
            # Execute step
            success = self.execute_step(step)
            if not success:
                return False
            
            # Update world model
            self.update_world_state()
        
        return True
    
    def execute_step(self, step: PlanStep) -> bool:
        """Execute single plan step."""
        if not self.robot:
            return False
            
        try:
            if step.action == "move_to":
                return self.robot.move_to(**step.params)
            elif step.action == "grasp":
                return self.robot.grasp_object(**step.params)
            elif step.action == "place":
                return self.robot.place_object(**step.params)
            else:
                print(f"Unknown action: {step.action}")
                return False
        except Exception as e:
            print(f"Execution error: {e}")
            return False
    
    def get_step_trajectory(self, step: PlanStep) -> Optional[List[np.ndarray]]:
        """Get predicted trajectory for step."""
        if step.action == "move_to":
            return step.params.get("trajectory")
        return None
    
    def update_world_state(self):
        """Update world model after execution."""
        if self.robot:
            # Get current scene state
            objects = self.robot.get_scene_state()
            self.world_model.update(objects)

class MonitoringModule:
    def __init__(self):
        self.metrics = {}
        self.error_history = []
    
    def track(self, execution_result: bool):
        """Track execution metrics and errors."""
        self.metrics['success'] = execution_result
        
        if not execution_result:
            self.error_history.append({
                'timestamp': np.datetime64('now'),
                'type': 'execution_failure'
            })
    
    def get_metrics(self) -> Dict[str, any]:
        """Get current metrics."""
        return self.metrics

class ZETAPipeline:
    def __init__(self):
        self.modules = {
            'perception': PerceptionModule(),
            'planning': PlanningModule(),
            'execution': ExecutionModule(),
            'monitoring': MonitoringModule()
        }
        
    def process(self, instruction: str, image: np.ndarray) -> bool:
        """Run complete pipeline."""
        try:
            # Perception
            percepts = self.modules['perception'].process(image)
            
            # Planning
            plan = self.modules['planning'].generate(instruction, percepts)
            
            # Execution
            success = self.modules['execution'].run(plan)
            
            # Monitoring
            self.modules['monitoring'].track(success)
            
            return success
            
        except Exception as e:
            print(f"Pipeline error: {e}")
            self.modules['monitoring'].track(False)
            return False
