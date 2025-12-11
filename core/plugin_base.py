"""
Plugin base classes and interfaces for the adaptive engineer system.
Defines the contract that all plugins must follow.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class StateVariable:
    """
    Represents a state variable that can be tracked and modified by plugins.

    Attributes:
        name: Variable name (e.g., 'cpu_utilization', 'threat_score')
        value: Current value
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        metadata: Additional metadata about the variable
    """

    name: str
    value: float
    min_value: float = 0.0
    max_value: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update(self, new_value: float) -> None:
        """Update the variable value within bounds."""
        self.value = max(self.min_value, min(self.max_value, new_value))

    def normalize(self) -> float:
        """Return value normalized to 0-1 range."""
        if self.max_value == self.min_value:
            return 0.5
        return (self.value - self.min_value) / (
            self.max_value - self.min_value
        )


class PluginBase(ABC):
    """
    Abstract base class for all plugins in the adaptive engineer system.

    Each plugin provides:
    - State management: Define and track domain-specific state variables
    - Memory operations: Store and retrieve domain-specific memories
    - Communication: Send and receive domain-specific signals
    - Actions: Execute domain-specific behaviors
    """

    def __init__(
        self, plugin_id: str, config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the plugin.

        Args:
            plugin_id: Unique identifier for this plugin instance
            config: Optional configuration dictionary
        """
        self.plugin_id = plugin_id
        self.config = config or {}
        self.enabled = True
        self.state_variables: Dict[str, StateVariable] = {}

    @abstractmethod
    def get_plugin_type(self) -> str:
        """
        Return the type of this plugin (e.g., 'IT', 'Security', 'ArtificialLife').

        Returns:
            String identifying the plugin type
        """
        pass

    @abstractmethod
    def initialize(self, node: Any) -> None:
        """
        Initialize the plugin with a reference to its host node.

        Args:
            node: The AliveLoopNode or agent that hosts this plugin
        """
        pass

    @abstractmethod
    def get_state_schema(self) -> Dict[str, StateVariable]:
        """
        Define the state variables this plugin manages.

        Returns:
            Dictionary mapping variable names to StateVariable definitions
        """
        pass

    @abstractmethod
    def update_state(self, delta_time: float) -> None:
        """
        Update plugin state variables based on elapsed time.

        Args:
            delta_time: Time elapsed since last update
        """
        pass

    @abstractmethod
    def process_signal(self, signal: Any) -> Optional[Any]:
        """
        Process an incoming signal relevant to this plugin.

        Args:
            signal: Incoming signal to process

        Returns:
            Optional response signal
        """
        pass

    @abstractmethod
    def get_actions(self) -> List[str]:
        """
        Return list of action types this plugin can perform.

        Returns:
            List of action type names
        """
        pass

    @abstractmethod
    def execute_action(self, action_type: str, params: Dict[str, Any]) -> bool:
        """
        Execute a plugin-specific action.

        Args:
            action_type: Type of action to execute
            params: Action parameters

        Returns:
            True if action executed successfully, False otherwise
        """
        pass

    def enable(self) -> None:
        """Enable this plugin."""
        self.enabled = True

    def disable(self) -> None:
        """Disable this plugin."""
        self.enabled = False

    def is_enabled(self) -> bool:
        """Check if plugin is enabled."""
        return self.enabled

    def get_state(self) -> Dict[str, float]:
        """
        Get current values of all state variables.

        Returns:
            Dictionary mapping variable names to current values
        """
        return {name: var.value for name, var in self.state_variables.items()}

    def get_state_variable(self, name: str) -> Optional[StateVariable]:
        """
        Get a specific state variable.

        Args:
            name: Variable name

        Returns:
            StateVariable or None if not found
        """
        return self.state_variables.get(name)

    def update_state_variable(self, name: str, value: float) -> bool:
        """
        Update a state variable value.

        Args:
            name: Variable name
            value: New value

        Returns:
            True if updated successfully, False if variable not found
        """
        if name in self.state_variables:
            self.state_variables[name].update(value)
            return True
        return False
