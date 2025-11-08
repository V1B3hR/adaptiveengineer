"""
Artificial Life plugin wrapping existing AL behaviors in the plugin system.
"""

import logging
from typing import Any, Dict, List, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.plugin_base import PluginBase, StateVariable


logger = logging.getLogger('artificial_life_plugin')


class ArtificialLifePlugin(PluginBase):
    """
    Plugin for Artificial Life behaviors.
    
    Wraps existing emotional, social, and adaptive behaviors:
    - Universal state (energy, health, emotion, trust)
    - Self-organization and emergence
    - Social learning and cooperation
    """
    
    def __init__(self, plugin_id: str = "artificial_life", config: Optional[Dict[str, Any]] = None):
        """Initialize Artificial Life plugin."""
        super().__init__(plugin_id, config)
        self.node = None
        
    def get_plugin_type(self) -> str:
        """Return plugin type."""
        return "ArtificialLife"
    
    def initialize(self, node: Any) -> None:
        """
        Initialize plugin with host node.
        
        Args:
            node: Host AliveLoopNode instance
        """
        self.node = node
        
        # Initialize state variables that expose existing node state
        self.state_variables = self.get_state_schema()
        
        logger.info(f"Artificial Life plugin initialized for node {getattr(node, 'node_id', 'unknown')}")
    
    def get_state_schema(self) -> Dict[str, StateVariable]:
        """
        Define Artificial Life state variables (wrapping existing node attributes).
        
        Returns:
            Dictionary of state variable definitions
        """
        return {
            # Universal state variables
            'energy': StateVariable(
                name='energy',
                value=0.0,  # Will sync with node
                min_value=0.0,
                max_value=20.0,
                metadata={'unit': 'units', 'description': 'Available energy for actions'}
            ),
            'health': StateVariable(
                name='health',
                value=1.0,  # Derived from composite emotional health
                min_value=0.0,
                max_value=1.0,
                metadata={'unit': 'ratio', 'description': 'Overall health/wellbeing'}
            ),
            'trust': StateVariable(
                name='trust',
                value=0.5,  # Will sync with node trust attribute
                min_value=0.0,
                max_value=1.0,
                metadata={'unit': 'ratio', 'description': 'General trustworthiness level'}
            ),
            
            # Emotional state
            'anxiety': StateVariable(
                name='anxiety',
                value=0.0,
                min_value=0.0,
                max_value=10.0,
                metadata={'unit': 'level', 'description': 'Anxiety level'}
            ),
            'calm': StateVariable(
                name='calm',
                value=1.0,
                min_value=0.0,
                max_value=5.0,
                metadata={'unit': 'level', 'description': 'Calmness level'}
            ),
            'joy': StateVariable(
                name='joy',
                value=0.0,
                min_value=0.0,
                max_value=5.0,
                metadata={'unit': 'level', 'description': 'Joy level'}
            ),
            
            # Self-organization indicators
            'emergency_mode': StateVariable(
                name='emergency_mode',
                value=0.0,  # 0=normal, 1=emergency
                min_value=0.0,
                max_value=1.0,
                metadata={'unit': 'boolean', 'description': 'Emergency self-organization active'}
            ),
            'adaptation_level': StateVariable(
                name='adaptation_level',
                value=0.5,
                min_value=0.0,
                max_value=1.0,
                metadata={'unit': 'ratio', 'description': 'Current adaptation/organization level'}
            )
        }
    
    def update_state(self, delta_time: float) -> None:
        """
        Sync state variables with node's actual AL state.
        
        Args:
            delta_time: Time elapsed since last update
        """
        if not self.node:
            return
        
        # Sync universal state variables
        self.update_state_variable('energy', getattr(self.node, 'energy', 0.0))
        self.update_state_variable('trust', getattr(self.node, 'trust', 0.5))
        
        # Sync emotional state
        self.update_state_variable('anxiety', getattr(self.node, 'anxiety', 0.0))
        self.update_state_variable('calm', getattr(self.node, 'calm', 1.0))
        self.update_state_variable('joy', getattr(self.node, 'joy', 0.0))
        
        # Calculate health from emotional state
        if hasattr(self.node, 'calculate_composite_emotional_health'):
            health = self.node.calculate_composite_emotional_health()
            self.update_state_variable('health', health)
        else:
            # Simple health calculation
            anxiety_factor = 1.0 - min(1.0, getattr(self.node, 'anxiety', 0.0) / 10.0)
            energy_factor = min(1.0, getattr(self.node, 'energy', 0.0) / 10.0)
            health = (anxiety_factor + energy_factor) / 2.0
            self.update_state_variable('health', health)
        
        # Track emergency/self-organization mode
        emergency = 1.0 if getattr(self.node, 'emergency_mode', False) else 0.0
        self.update_state_variable('emergency_mode', emergency)
        
        # Calculate adaptation level based on various factors
        adaptation = self._calculate_adaptation_level()
        self.update_state_variable('adaptation_level', adaptation)
    
    def _calculate_adaptation_level(self) -> float:
        """
        Calculate how well the node is adapting/self-organizing.
        
        Higher values indicate better adaptation to current conditions.
        """
        if not self.node:
            return 0.5
        
        # Factors that indicate good adaptation
        factors = []
        
        # Energy management
        energy = getattr(self.node, 'energy', 10.0)
        if 5.0 <= energy <= 15.0:
            factors.append(1.0)  # Optimal energy range
        elif energy < 2.0:
            factors.append(0.0)  # Critical energy
        else:
            factors.append(0.5)
        
        # Emotional stability
        anxiety = getattr(self.node, 'anxiety', 0.0)
        if anxiety < 3.0:
            factors.append(1.0)  # Low anxiety
        elif anxiety > 8.0:
            factors.append(0.0)  # High anxiety
        else:
            factors.append(0.5)
        
        # Social integration
        trust_network = getattr(self.node, 'trust_network', {})
        if len(trust_network) > 0:
            avg_trust = sum(trust_network.values()) / len(trust_network)
            factors.append(avg_trust)
        else:
            factors.append(0.3)  # Isolated
        
        # Emergency response capability
        if hasattr(self.node, 'emergency_mode'):
            if getattr(self.node, 'emergency_mode', False):
                # In emergency but still functioning
                factors.append(0.7)
            else:
                # Normal operation
                factors.append(1.0)
        
        return sum(factors) / len(factors) if factors else 0.5
    
    def process_signal(self, signal: Any) -> Optional[Any]:
        """
        Process AL-related signals (delegates to node's existing methods).
        
        Args:
            signal: Incoming signal
            
        Returns:
            Optional response signal
        """
        # AL signals are handled by the node itself
        # This plugin just exposes the state
        return None
    
    def get_actions(self) -> List[str]:
        """Return available AL actions."""
        return [
            'share_memory',
            'request_help',
            'offer_support',
            'adapt_to_stress',
            'self_organize'
        ]
    
    def execute_action(self, action_type: str, params: Dict[str, Any]) -> bool:
        """
        Execute an AL action (delegates to node's existing methods).
        
        Args:
            action_type: Type of action
            params: Action parameters
            
        Returns:
            True if action succeeded
        """
        if not self.node:
            return False
        
        if action_type == 'share_memory':
            # Trigger memory sharing
            nodes = params.get('target_nodes', [])
            if nodes and hasattr(self.node, 'share_valuable_memory'):
                self.node.share_valuable_memory(nodes)
                return True
        
        elif action_type == 'request_help':
            # Request help from network
            if hasattr(self.node, 'try_send_help_signal'):
                return self.node.try_send_help_signal()
        
        elif action_type == 'offer_support':
            # Offer support to others
            target = params.get('target_node')
            amount = params.get('support_amount', 1.0)
            if target and hasattr(self.node, 'receive_help'):
                target.receive_help(amount)
                return True
        
        elif action_type == 'adapt_to_stress':
            # Trigger adaptive response to stress
            if hasattr(self.node, 'adaptive_energy_allocation'):
                self.node.adaptive_energy_allocation()
                return True
        
        elif action_type == 'self_organize':
            # Trigger self-organization behavior
            return self._trigger_self_organization()
        
        return False
    
    def _trigger_self_organization(self) -> bool:
        """
        Trigger self-organization behaviors based on local rules.
        
        This implements the "Emergency & Self-Organization" requirement.
        """
        if not self.node:
            return False
        
        logger.info(f"Triggering self-organization for node {getattr(self.node, 'node_id', 'unknown')}")
        
        # Rule 1: Low energy → seek help and conserve
        energy = getattr(self.node, 'energy', 10.0)
        if energy < 3.0:
            if hasattr(self.node, 'activate_emergency_energy_conservation'):
                self.node.activate_emergency_energy_conservation()
            if hasattr(self.node, 'try_send_help_signal'):
                self.node.try_send_help_signal()
        
        # Rule 2: High anxiety → seek support
        anxiety = getattr(self.node, 'anxiety', 0.0)
        if anxiety > 7.0:
            if hasattr(self.node, 'check_anxiety_overwhelm'):
                overwhelmed = self.node.check_anxiety_overwhelm()
                if overwhelmed and hasattr(self.node, 'try_send_help_signal'):
                    self.node.try_send_help_signal()
        
        # Rule 3: Detect threats → increase defense
        if hasattr(self.node, 'detect_energy_attack'):
            if self.node.detect_energy_attack():
                if hasattr(self.node, 'threat_assessment_level'):
                    self.node.threat_assessment_level = min(3, self.node.threat_assessment_level + 1)
        
        # Rule 4: Stable state → share knowledge
        if energy > 8.0 and anxiety < 4.0:
            # In good state, contribute to collective
            if hasattr(self.node, 'collective_contribution'):
                self.node.collective_contribution += 0.1
        
        return True
    
    def get_al_summary(self) -> Dict[str, Any]:
        """Get a summary of Artificial Life metrics."""
        return {
            'energy': self.state_variables['energy'].value,
            'health': self.state_variables['health'].value,
            'trust': self.state_variables['trust'].value,
            'anxiety': self.state_variables['anxiety'].value,
            'calm': self.state_variables['calm'].value,
            'joy': self.state_variables['joy'].value,
            'emergency_mode': bool(self.state_variables['emergency_mode'].value),
            'adaptation_level': self.state_variables['adaptation_level'].value,
            'is_self_organizing': self.state_variables['adaptation_level'].value > 0.6
        }
