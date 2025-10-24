"""
DevStream Protocol Components.

This package provides the core protocol enforcement components for DevStream:

- ProtocolStateManager: Atomic state persistence and session management
- EnforcementGate: Blocking enforcement gate with user interaction
- StepValidator: Step completion validation with memory search
- TaskFirstHandler: Mandatory STEP 1 enforcement for task creation
- TaskStateSync: Automatic state synchronization with hook integration
"""

__version__ = "1.0.0"
__author__ = "DevStream Team"

# Export key classes for easy import
from .protocol_state_manager import ProtocolStep, ProtocolStateManager
from .enforcement_gate import EnforcementGate
from .step_validator import StepValidator
from .task_first_handler import TaskFirstHandler
from .task_state_sync import TaskStateSync
from .interactive_step_validator import InteractiveStepValidator
from .implementation_plan_generator import ImplementationPlanGenerator

__all__ = [
    'ProtocolStep',
    'ProtocolStateManager',
    'EnforcementGate',
    'StepValidator',
    'TaskFirstHandler',
    'TaskStateSync',
    'InteractiveStepValidator',
    'ImplementationPlanGenerator'
]