"""ZETA framework configuration."""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ZETAConfig:
    # Core Settings
    enable_zeta_scoring: bool = True
    enable_visual_grounding: bool = True
    enable_affordance: bool = True
    enable_feedback_loop: bool = True
    use_constraint_fallback: bool = True
    
    # Robot Settings
    robot_type: str = "panda"  # franka panda robot
    max_velocity: float = 1.0  # m/s
    max_acceleration: float = 2.0  # m/s²
    safety_margin: float = 0.1  # meters
    
    # Learning Settings
    learning_rate: float = 0.001
    batch_size: int = 64
    buffer_size: int = 10000
    gamma: float = 0.99
    
    # Pipeline Settings
    perception_confidence_threshold: float = 0.8
    planning_timeout: float = 5.0  # seconds
    execution_timeout: float = 30.0  # seconds
    
    # Safety Settings
    workspace_bounds: Dict[str, tuple] = None
    
    def __post_init__(self):
        if self.workspace_bounds is None:
            self.workspace_bounds = {
                'x': (-1.0, 1.0),
                'y': (-1.0, 1.0),
                'z': (0.0, 1.5)
            }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ZETAConfig':
        """Create config from dictionary."""
        return cls(**{
            k: v for k, v in config_dict.items() 
            if k in cls.__dataclass_fields__
        })

# Default configuration
DEFAULT_CONFIG = ZETAConfig()
