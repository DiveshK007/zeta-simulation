"""Safety monitoring and constraints module for ZETA."""

import numpy as np
from typing import Dict, List, Tuple, Optional

class CollisionDetector:
    def __init__(self):
        self.safety_margin = 0.1  # 10cm safety margin
        self.workspace_bounds = {
            'x': (-1.0, 1.0),
            'y': (-1.0, 1.0),
            'z': (0.0, 1.5)
        }
    
    def check(self, point: np.ndarray) -> bool:
        """Check if point is within safe workspace bounds."""
        x, y, z = point
        bounds = self.workspace_bounds
        return (bounds['x'][0] <= x <= bounds['x'][1] and
                bounds['y'][0] <= y <= bounds['y'][1] and
                bounds['z'][0] <= z <= bounds['z'][1])

class SafetyMonitor:
    def __init__(self):
        self.safety_boundaries: Dict[str, List[float]] = {}
        self.collision_detector = CollisionDetector()
        self.max_velocity = 1.0  # m/s
        self.max_acceleration = 2.0  # m/s²
        
    def check_motion_safety(self, trajectory: List[np.ndarray]) -> Tuple[bool, str]:
        """Check if trajectory is safe."""
        if not trajectory:
            return False, "Empty trajectory"
            
        # Check workspace bounds
        if not all(self.collision_detector.check(p) for p in trajectory):
            return False, "Trajectory exceeds workspace bounds"
            
        # Check velocity and acceleration
        for i in range(len(trajectory) - 1):
            velocity = np.linalg.norm(trajectory[i+1] - trajectory[i])
            if velocity > self.max_velocity:
                return False, f"Velocity limit exceeded: {velocity:.2f} m/s"
                
            if i < len(trajectory) - 2:
                acc = np.linalg.norm(trajectory[i+2] - 2*trajectory[i+1] + trajectory[i])
                if acc > self.max_acceleration:
                    return False, f"Acceleration limit exceeded: {acc:.2f} m/s²"
        
        return True, "Trajectory is safe"
    
    def set_safety_boundary(self, object_id: str, boundary: List[float]):
        """Set safety boundary for an object."""
        self.safety_boundaries[object_id] = boundary
    
    def check_object_safety(self, object_id: str, position: np.ndarray) -> bool:
        """Check if object is within its safety boundary."""
        if object_id not in self.safety_boundaries:
            return True
        
        boundary = self.safety_boundaries[object_id]
        return all(b[0] <= p <= b[1] for p, b in zip(position, boundary))

class EmergencyStop:
    def __init__(self):
        self.is_stopped = False
        self.stop_reason = None
    
    def trigger(self, reason: str):
        """Trigger emergency stop."""
        self.is_stopped = True
        self.stop_reason = reason
        print(f"EMERGENCY STOP: {reason}")
    
    def reset(self):
        """Reset emergency stop."""
        self.is_stopped = False
        self.stop_reason = None
