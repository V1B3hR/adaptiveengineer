"""
IT Operations plugin for monitoring and managing service health and resources.
"""

import logging
from typing import Any, Dict, List, Optional
from collections import deque

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.plugin_base import PluginBase, StateVariable


logger = logging.getLogger('it_operations_plugin')


class ITOperationsPlugin(PluginBase):
    """
    Plugin for IT Operations monitoring and management.
    
    Tracks:
    - Service health (uptime, error rates)
    - Resource utilization (CPU, memory, bandwidth)
    - Performance metrics
    - Incident tracking
    """
    
    def __init__(self, plugin_id: str = "it_ops", config: Optional[Dict[str, Any]] = None):
        """Initialize IT Operations plugin."""
        super().__init__(plugin_id, config)
        self.node = None
        self.incident_history = deque(maxlen=100)
        self.performance_history = deque(maxlen=50)
        
    def get_plugin_type(self) -> str:
        """Return plugin type."""
        return "IT"
    
    def initialize(self, node: Any) -> None:
        """
        Initialize plugin with host node.
        
        Args:
            node: Host AliveLoopNode instance
        """
        self.node = node
        
        # Initialize state variables based on schema
        self.state_variables = self.get_state_schema()
        
        logger.info(f"IT Operations plugin initialized for node {getattr(node, 'node_id', 'unknown')}")
    
    def get_state_schema(self) -> Dict[str, StateVariable]:
        """
        Define IT Operations state variables.
        
        Returns:
            Dictionary of state variable definitions
        """
        return {
            # Service health metrics
            'service_uptime': StateVariable(
                name='service_uptime',
                value=1.0,  # Start at 100% uptime
                min_value=0.0,
                max_value=1.0,
                metadata={'unit': 'percentage', 'description': 'Service availability ratio'}
            ),
            'error_rate': StateVariable(
                name='error_rate',
                value=0.0,  # Start with no errors
                min_value=0.0,
                max_value=1.0,
                metadata={'unit': 'percentage', 'description': 'Error rate ratio'}
            ),
            
            # Resource utilization
            'cpu_utilization': StateVariable(
                name='cpu_utilization',
                value=0.3,  # Start at 30% utilization
                min_value=0.0,
                max_value=1.0,
                metadata={'unit': 'percentage', 'description': 'CPU usage ratio'}
            ),
            'memory_utilization': StateVariable(
                name='memory_utilization',
                value=0.4,  # Start at 40% utilization
                min_value=0.0,
                max_value=1.0,
                metadata={'unit': 'percentage', 'description': 'Memory usage ratio'}
            ),
            'bandwidth_utilization': StateVariable(
                name='bandwidth_utilization',
                value=0.2,  # Start at 20% utilization
                min_value=0.0,
                max_value=1.0,
                metadata={'unit': 'percentage', 'description': 'Network bandwidth usage ratio'}
            ),
            
            # Incident tracking
            'incident_count': StateVariable(
                name='incident_count',
                value=0.0,
                min_value=0.0,
                max_value=100.0,
                metadata={'unit': 'count', 'description': 'Active incident count'}
            ),
            'remediation_success_rate': StateVariable(
                name='remediation_success_rate',
                value=0.9,  # Start with 90% success rate
                min_value=0.0,
                max_value=1.0,
                metadata={'unit': 'percentage', 'description': 'Incident remediation success ratio'}
            )
        }
    
    def update_state(self, delta_time: float) -> None:
        """
        Update IT Operations state based on node conditions.
        
        Args:
            delta_time: Time elapsed since last update
        """
        if not self.node:
            return
        
        # Update resource utilization based on node activity
        # Higher energy consumption = higher resource usage
        energy_level = getattr(self.node, 'energy', 10.0)
        max_energy = 20.0  # Assumed max energy
        
        # CPU utilization increases with activity
        base_cpu = 0.2
        activity_cpu = (1.0 - energy_level / max_energy) * 0.5
        self.update_state_variable('cpu_utilization', base_cpu + activity_cpu)
        
        # Memory utilization based on memory size
        memory_count = len(getattr(self.node, 'memory', []))
        max_memory = getattr(self.node, 'max_memory_size', 1000)
        memory_util = min(1.0, memory_count / max_memory)
        self.update_state_variable('memory_utilization', memory_util)
        
        # Bandwidth utilization based on communication activity
        comm_count = getattr(self.node, 'communications_this_step', 0)
        max_comm = getattr(self.node, 'max_communications_per_step', 5)
        bandwidth_util = min(1.0, comm_count / max(1, max_comm))
        self.update_state_variable('bandwidth_utilization', bandwidth_util)
        
        # Service uptime decreases with high anxiety or low energy
        anxiety = getattr(self.node, 'anxiety', 0.0)
        if anxiety > 8.0 or energy_level < 2.0:
            current_uptime = self.state_variables['service_uptime'].value
            self.update_state_variable('service_uptime', current_uptime * 0.95)
        else:
            # Gradually recover uptime
            current_uptime = self.state_variables['service_uptime'].value
            self.update_state_variable('service_uptime', min(1.0, current_uptime + 0.01))
        
        # Error rate increases under stress
        if anxiety > 5.0:
            error_increase = (anxiety - 5.0) / 10.0
            current_error = self.state_variables['error_rate'].value
            self.update_state_variable('error_rate', min(1.0, current_error + error_increase * 0.1))
        else:
            # Error rate decreases during calm
            current_error = self.state_variables['error_rate'].value
            self.update_state_variable('error_rate', max(0.0, current_error - 0.01))
        
        # Record performance metrics
        self.performance_history.append({
            'timestamp': getattr(self.node, '_time', 0),
            'cpu': self.state_variables['cpu_utilization'].value,
            'memory': self.state_variables['memory_utilization'].value,
            'uptime': self.state_variables['service_uptime'].value
        })
    
    def process_signal(self, signal: Any) -> Optional[Any]:
        """
        Process IT Operations related signals.
        
        Args:
            signal: Incoming signal
            
        Returns:
            Optional response signal
        """
        if not hasattr(signal, 'signal_type'):
            return None
        
        signal_type = signal.signal_type
        
        if signal_type == 'service_health_query':
            # Respond with service health status
            return self._create_health_response()
        elif signal_type == 'incident_report':
            # Record incident
            self._record_incident(signal.content)
            return None
        elif signal_type == 'remediation_request':
            # Attempt remediation
            return self._attempt_remediation(signal.content)
        
        return None
    
    def get_actions(self) -> List[str]:
        """Return available IT Operations actions."""
        return [
            'restart_service',
            'scale_resources',
            'trigger_alert',
            'run_diagnostics',
            'apply_patch'
        ]
    
    def execute_action(self, action_type: str, params: Dict[str, Any]) -> bool:
        """
        Execute an IT Operations action.
        
        Args:
            action_type: Type of action
            params: Action parameters
            
        Returns:
            True if action succeeded
        """
        if action_type == 'restart_service':
            return self._restart_service()
        elif action_type == 'scale_resources':
            return self._scale_resources(params.get('scale_factor', 1.0))
        elif action_type == 'trigger_alert':
            return self._trigger_alert(params.get('alert_level', 'warning'))
        elif action_type == 'run_diagnostics':
            return self._run_diagnostics()
        elif action_type == 'apply_patch':
            return self._apply_patch(params.get('patch_id'))
        
        logger.warning(f"Unknown action type: {action_type}")
        return False
    
    def _create_health_response(self) -> Any:
        """Create a service health response signal."""
        # Import here to avoid circular dependency
        try:
            from adaptiveengineer import SocialSignal
            return SocialSignal(
                content={
                    'uptime': self.state_variables['service_uptime'].value,
                    'error_rate': self.state_variables['error_rate'].value,
                    'cpu_util': self.state_variables['cpu_utilization'].value,
                    'memory_util': self.state_variables['memory_utilization'].value
                },
                signal_type='service_health_response',
                urgency=0.3,
                source_id=getattr(self.node, 'node_id', 0)
            )
        except ImportError:
            return None
    
    def _record_incident(self, incident_data: Any) -> None:
        """Record an incident."""
        self.incident_history.append({
            'timestamp': getattr(self.node, '_time', 0),
            'data': incident_data
        })
        
        # Increment incident count
        current_count = self.state_variables['incident_count'].value
        self.update_state_variable('incident_count', current_count + 1)
        
        logger.info(f"Incident recorded for node {getattr(self.node, 'node_id', 'unknown')}")
    
    def _attempt_remediation(self, remediation_request: Any) -> Any:
        """Attempt to remediate an incident."""
        # Simulate remediation success based on current state
        success_rate = self.state_variables['remediation_success_rate'].value
        
        import random
        success = random.random() < success_rate
        
        if success:
            # Decrease incident count
            current_count = self.state_variables['incident_count'].value
            self.update_state_variable('incident_count', max(0, current_count - 1))
            
            logger.info(f"Remediation successful for node {getattr(self.node, 'node_id', 'unknown')}")
        
        # Return remediation result
        try:
            from adaptiveengineer import SocialSignal
            return SocialSignal(
                content={'success': success, 'incident': remediation_request},
                signal_type='remediation_response',
                urgency=0.5,
                source_id=getattr(self.node, 'node_id', 0)
            )
        except ImportError:
            return None
    
    def _restart_service(self) -> bool:
        """Restart service (simulated)."""
        logger.info(f"Restarting service for node {getattr(self.node, 'node_id', 'unknown')}")
        
        # Reset error rate and improve uptime
        self.update_state_variable('error_rate', 0.0)
        self.update_state_variable('service_uptime', 1.0)
        
        return True
    
    def _scale_resources(self, scale_factor: float) -> bool:
        """Scale resources up or down."""
        logger.info(f"Scaling resources by {scale_factor}x for node {getattr(self.node, 'node_id', 'unknown')}")
        
        # Adjust resource utilization
        cpu = self.state_variables['cpu_utilization'].value / scale_factor
        memory = self.state_variables['memory_utilization'].value / scale_factor
        
        self.update_state_variable('cpu_utilization', cpu)
        self.update_state_variable('memory_utilization', memory)
        
        return True
    
    def _trigger_alert(self, alert_level: str) -> bool:
        """Trigger an alert."""
        logger.warning(f"Alert triggered ({alert_level}) for node {getattr(self.node, 'node_id', 'unknown')}")
        return True
    
    def _run_diagnostics(self) -> bool:
        """Run system diagnostics."""
        logger.info(f"Running diagnostics for node {getattr(self.node, 'node_id', 'unknown')}")
        
        # Check for issues
        issues = []
        if self.state_variables['cpu_utilization'].value > 0.8:
            issues.append('high_cpu')
        if self.state_variables['memory_utilization'].value > 0.8:
            issues.append('high_memory')
        if self.state_variables['error_rate'].value > 0.1:
            issues.append('high_errors')
        
        logger.info(f"Diagnostics found {len(issues)} issues: {issues}")
        return True
    
    def _apply_patch(self, patch_id: Optional[str]) -> bool:
        """Apply a software patch."""
        if not patch_id:
            return False
        
        logger.info(f"Applying patch {patch_id} for node {getattr(self.node, 'node_id', 'unknown')}")
        
        # Improve remediation success rate slightly
        current_rate = self.state_variables['remediation_success_rate'].value
        self.update_state_variable('remediation_success_rate', min(1.0, current_rate + 0.05))
        
        return True
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of IT health metrics."""
        return {
            'service_uptime': self.state_variables['service_uptime'].value,
            'error_rate': self.state_variables['error_rate'].value,
            'cpu_utilization': self.state_variables['cpu_utilization'].value,
            'memory_utilization': self.state_variables['memory_utilization'].value,
            'bandwidth_utilization': self.state_variables['bandwidth_utilization'].value,
            'incident_count': self.state_variables['incident_count'].value,
            'remediation_success_rate': self.state_variables['remediation_success_rate'].value,
            'recent_incidents': len(self.incident_history)
        }
