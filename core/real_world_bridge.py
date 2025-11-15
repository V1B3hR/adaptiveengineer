"""
Real-World Integration Bridge for Phase 4

Connects the fully autonomous polity to the real world. Translates the collective's
decisions into concrete commands for real-world systems (Ansible playbooks, AWS API
calls, Kubernetes operations, etc.).

Provides safe execution with rollback capabilities and comprehensive auditing.
"""

import logging
import time
import json
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from collections import deque

logger = logging.getLogger(__name__)


class IntegrationSystem(Enum):
    """Supported external systems"""
    ANSIBLE = "ansible"
    AWS = "aws"
    KUBERNETES = "kubernetes"
    TERRAFORM = "terraform"
    DOCKER = "docker"
    JENKINS = "jenkins"
    DATADOG = "datadog"
    PAGERDUTY = "pagerduty"
    SLACK = "slack"
    CUSTOM = "custom"


class CommandStatus(Enum):
    """Status of a real-world command"""
    PENDING = "pending"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class RiskLevel(Enum):
    """Risk level of a command"""
    LOW = "low"          # Read-only, no system changes
    MEDIUM = "medium"    # Non-critical changes with easy rollback
    HIGH = "high"        # Critical changes affecting services
    CRITICAL = "critical" # Major system changes, requires approval


@dataclass
class RealWorldCommand:
    """A command to be executed in the real world"""
    command_id: str
    timestamp: float
    system: IntegrationSystem
    operation: str  # e.g., "patch_server", "scale_service", "deploy_config"
    parameters: Dict[str, Any]
    agent_decision_id: Optional[str] = None
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    
    # Safety
    risk_level: RiskLevel = RiskLevel.MEDIUM
    dry_run: bool = False
    requires_approval: bool = False
    approved_by: Optional[str] = None
    
    # Execution
    status: CommandStatus = CommandStatus.PENDING
    executed_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Results
    success: Optional[bool] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # Rollback
    rollback_available: bool = False
    rollback_command: Optional[Dict[str, Any]] = None
    rolled_back: bool = False


@dataclass
class IntegrationAdapter:
    """Adapter for a specific external system"""
    system: IntegrationSystem
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    executor: Optional[Callable] = None  # Function to execute commands
    dry_run_mode: bool = False
    
    # Statistics
    commands_executed: int = 0
    commands_succeeded: int = 0
    commands_failed: int = 0
    
    def get_success_rate(self) -> float:
        """Get command success rate"""
        if self.commands_executed == 0:
            return 0.0
        return self.commands_succeeded / self.commands_executed


class RealWorldBridge:
    """
    Bridge between autonomous agent collective and real-world systems.
    
    Translates high-level agent decisions into concrete system commands,
    with safety checks, approval workflows, and rollback capabilities.
    """
    
    def __init__(self,
                 require_approval_for_high_risk: bool = True,
                 dry_run_mode: bool = False,
                 constitutional_framework: Optional[object] = None):
        """
        Initialize Real-World Bridge.
        
        Args:
            require_approval_for_high_risk: Whether high-risk commands need approval
            dry_run_mode: If True, commands are simulated, not executed
            constitutional_framework: Optional ConstitutionalFramework for compliance
        """
        self.require_approval_for_high_risk = require_approval_for_high_risk
        self.dry_run_mode = dry_run_mode
        self.constitutional_framework = constitutional_framework
        
        # Integration adapters
        self.adapters: Dict[IntegrationSystem, IntegrationAdapter] = {}
        
        # Commands
        self.commands: Dict[str, RealWorldCommand] = {}
        self.pending_approval: List[str] = []
        self.command_history: deque = deque(maxlen=1000)
        
        # Decision mapping
        self.decision_to_commands: Dict[str, List[str]] = {}
        
        # Statistics
        self.total_commands_created = 0
        self.total_commands_executed = 0
        self.total_commands_succeeded = 0
        self.total_commands_failed = 0
        self.total_rollbacks = 0
        
        logger.info(f"Real-World Bridge initialized (dry_run={dry_run_mode})")
    
    def register_adapter(self,
                        system: IntegrationSystem,
                        config: Dict[str, Any],
                        executor: Optional[Callable] = None,
                        dry_run_mode: Optional[bool] = None) -> IntegrationAdapter:
        """
        Register an integration adapter for an external system.
        
        Args:
            system: External system type
            config: Configuration for the adapter
            executor: Function to execute commands (optional)
            dry_run_mode: Override global dry_run setting for this adapter
        
        Returns:
            Registered IntegrationAdapter
        """
        adapter = IntegrationAdapter(
            system=system,
            config=config,
            executor=executor,
            dry_run_mode=dry_run_mode if dry_run_mode is not None else self.dry_run_mode
        )
        
        self.adapters[system] = adapter
        
        logger.info(f"Integration adapter registered: {system.value} "
                   f"(dry_run={adapter.dry_run_mode})")
        
        return adapter
    
    def translate_decision(self,
                          decision_id: str,
                          decision_type: str,
                          decision_data: Dict[str, Any],
                          agent_id: Optional[str] = None,
                          task_id: Optional[str] = None) -> List[RealWorldCommand]:
        """
        Translate an agent decision into real-world commands.
        
        Args:
            decision_id: Unique decision identifier
            decision_type: Type of decision
            decision_data: Decision data and parameters
            agent_id: ID of agent that made decision
            task_id: ID of task this decision relates to
        
        Returns:
            List of RealWorldCommand objects
        """
        logger.info(f"Translating decision {decision_id} (type={decision_type})")
        
        commands = []
        
        # Translate based on decision type
        if decision_type == "patch_server":
            commands.extend(self._translate_patch_server(decision_id, decision_data, 
                                                         agent_id, task_id))
        
        elif decision_type == "scale_service":
            commands.extend(self._translate_scale_service(decision_id, decision_data,
                                                          agent_id, task_id))
        
        elif decision_type == "deploy_configuration":
            commands.extend(self._translate_deploy_config(decision_id, decision_data,
                                                          agent_id, task_id))
        
        elif decision_type == "restart_service":
            commands.extend(self._translate_restart_service(decision_id, decision_data,
                                                            agent_id, task_id))
        
        elif decision_type == "update_firewall":
            commands.extend(self._translate_update_firewall(decision_id, decision_data,
                                                            agent_id, task_id))
        
        elif decision_type == "deploy_container":
            commands.extend(self._translate_deploy_container(decision_id, decision_data,
                                                             agent_id, task_id))
        
        else:
            logger.warning(f"Unknown decision type: {decision_type}")
        
        # Track decision-to-command mapping
        if commands:
            self.decision_to_commands[decision_id] = [cmd.command_id for cmd in commands]
        
        logger.info(f"Decision {decision_id} translated to {len(commands)} command(s)")
        
        return commands
    
    def _translate_patch_server(self,
                               decision_id: str,
                               data: Dict[str, Any],
                               agent_id: Optional[str],
                               task_id: Optional[str]) -> List[RealWorldCommand]:
        """Translate patch server decision to Ansible command"""
        server_id = data.get('server_id', 'unknown')
        patch_name = data.get('patch_name', 'security-updates')
        
        command_id = f"cmd_{int(time.time()*1000)}_{hashlib.md5(decision_id.encode()).hexdigest()[:8]}"
        
        command = RealWorldCommand(
            command_id=command_id,
            timestamp=time.time(),
            system=IntegrationSystem.ANSIBLE,
            operation="apply_patch",
            parameters={
                'playbook': 'patch_server.yml',
                'hosts': server_id,
                'extra_vars': {
                    'patch_name': patch_name,
                    'backup': True
                }
            },
            agent_decision_id=decision_id,
            agent_id=agent_id,
            task_id=task_id,
            risk_level=RiskLevel.HIGH,
            requires_approval=True,
            rollback_available=True,
            rollback_command={
                'playbook': 'rollback_patch.yml',
                'hosts': server_id
            }
        )
        
        self._register_command(command)
        return [command]
    
    def _translate_scale_service(self,
                                decision_id: str,
                                data: Dict[str, Any],
                                agent_id: Optional[str],
                                task_id: Optional[str]) -> List[RealWorldCommand]:
        """Translate scale service decision to Kubernetes command"""
        service_name = data.get('service_name', 'unknown')
        replicas = data.get('replicas', 3)
        
        command_id = f"cmd_{int(time.time()*1000)}_{hashlib.md5(decision_id.encode()).hexdigest()[:8]}"
        
        command = RealWorldCommand(
            command_id=command_id,
            timestamp=time.time(),
            system=IntegrationSystem.KUBERNETES,
            operation="scale_deployment",
            parameters={
                'deployment': service_name,
                'replicas': replicas,
                'namespace': data.get('namespace', 'default')
            },
            agent_decision_id=decision_id,
            agent_id=agent_id,
            task_id=task_id,
            risk_level=RiskLevel.MEDIUM,
            requires_approval=False,
            rollback_available=True,
            rollback_command={
                'deployment': service_name,
                'replicas': data.get('original_replicas', 1)
            }
        )
        
        self._register_command(command)
        return [command]
    
    def _translate_deploy_config(self,
                                decision_id: str,
                                data: Dict[str, Any],
                                agent_id: Optional[str],
                                task_id: Optional[str]) -> List[RealWorldCommand]:
        """Translate deploy configuration decision to Terraform command"""
        config_name = data.get('config_name', 'unknown')
        
        command_id = f"cmd_{int(time.time()*1000)}_{hashlib.md5(decision_id.encode()).hexdigest()[:8]}"
        
        command = RealWorldCommand(
            command_id=command_id,
            timestamp=time.time(),
            system=IntegrationSystem.TERRAFORM,
            operation="apply_config",
            parameters={
                'config': config_name,
                'variables': data.get('variables', {}),
                'auto_approve': False
            },
            agent_decision_id=decision_id,
            agent_id=agent_id,
            task_id=task_id,
            risk_level=RiskLevel.CRITICAL,
            requires_approval=True,
            rollback_available=True,
            rollback_command={
                'config': config_name,
                'action': 'destroy'
            }
        )
        
        self._register_command(command)
        return [command]
    
    def _translate_restart_service(self,
                                  decision_id: str,
                                  data: Dict[str, Any],
                                  agent_id: Optional[str],
                                  task_id: Optional[str]) -> List[RealWorldCommand]:
        """Translate restart service decision to Docker command"""
        service_name = data.get('service_name', 'unknown')
        
        command_id = f"cmd_{int(time.time()*1000)}_{hashlib.md5(decision_id.encode()).hexdigest()[:8]}"
        
        command = RealWorldCommand(
            command_id=command_id,
            timestamp=time.time(),
            system=IntegrationSystem.DOCKER,
            operation="restart_container",
            parameters={
                'container': service_name,
                'timeout': data.get('timeout', 10)
            },
            agent_decision_id=decision_id,
            agent_id=agent_id,
            task_id=task_id,
            risk_level=RiskLevel.MEDIUM,
            requires_approval=False,
            rollback_available=False
        )
        
        self._register_command(command)
        return [command]
    
    def _translate_update_firewall(self,
                                  decision_id: str,
                                  data: Dict[str, Any],
                                  agent_id: Optional[str],
                                  task_id: Optional[str]) -> List[RealWorldCommand]:
        """Translate update firewall decision to AWS command"""
        rule_id = data.get('rule_id', 'unknown')
        
        command_id = f"cmd_{int(time.time()*1000)}_{hashlib.md5(decision_id.encode()).hexdigest()[:8]}"
        
        command = RealWorldCommand(
            command_id=command_id,
            timestamp=time.time(),
            system=IntegrationSystem.AWS,
            operation="update_security_group",
            parameters={
                'group_id': data.get('security_group_id'),
                'rule': {
                    'id': rule_id,
                    'protocol': data.get('protocol', 'tcp'),
                    'port_range': data.get('port_range'),
                    'source': data.get('source'),
                    'action': data.get('action', 'allow')
                }
            },
            agent_decision_id=decision_id,
            agent_id=agent_id,
            task_id=task_id,
            risk_level=RiskLevel.HIGH,
            requires_approval=True,
            rollback_available=True,
            rollback_command={
                'group_id': data.get('security_group_id'),
                'action': 'revert',
                'rule_id': rule_id
            }
        )
        
        self._register_command(command)
        return [command]
    
    def _translate_deploy_container(self,
                                   decision_id: str,
                                   data: Dict[str, Any],
                                   agent_id: Optional[str],
                                   task_id: Optional[str]) -> List[RealWorldCommand]:
        """Translate deploy container decision to Kubernetes command"""
        container_image = data.get('image', 'unknown')
        
        command_id = f"cmd_{int(time.time()*1000)}_{hashlib.md5(decision_id.encode()).hexdigest()[:8]}"
        
        command = RealWorldCommand(
            command_id=command_id,
            timestamp=time.time(),
            system=IntegrationSystem.KUBERNETES,
            operation="deploy_pod",
            parameters={
                'name': data.get('name', 'unnamed-pod'),
                'image': container_image,
                'namespace': data.get('namespace', 'default'),
                'replicas': data.get('replicas', 1),
                'env_vars': data.get('env_vars', {})
            },
            agent_decision_id=decision_id,
            agent_id=agent_id,
            task_id=task_id,
            risk_level=RiskLevel.MEDIUM,
            requires_approval=False,
            rollback_available=True,
            rollback_command={
                'action': 'delete',
                'name': data.get('name', 'unnamed-pod')
            }
        )
        
        self._register_command(command)
        return [command]
    
    def _register_command(self, command: RealWorldCommand):
        """Register a command in the system"""
        self.commands[command.command_id] = command
        self.total_commands_created += 1
        
        # Check if requires approval
        if command.requires_approval and self.require_approval_for_high_risk:
            self.pending_approval.append(command.command_id)
            logger.info(f"Command {command.command_id} requires approval "
                       f"(risk={command.risk_level.value})")
        
        # Check constitutional compliance if framework available
        if self.constitutional_framework:
            self._check_constitutional_compliance(command)
    
    def _check_constitutional_compliance(self, command: RealWorldCommand):
        """Check command against constitutional framework"""
        # Build action context for evaluation
        context = {
            'affects_services': command.operation in ['restart_service', 'scale_service'],
            'affects_critical_components': command.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL],
            'is_destructive': command.operation in ['destroy', 'delete', 'terminate'],
            'has_backup': command.rollback_available,
            'resource_usage': 0.5 if command.risk_level == RiskLevel.HIGH else 0.3
        }
        
        evaluation = self.constitutional_framework.evaluate_action(
            action_id=command.command_id,
            action_description=f"{command.system.value}:{command.operation}",
            action_context=context,
            agent_id=command.agent_id
        )
        
        if not evaluation.is_compliant:
            logger.warning(f"Command {command.command_id} violates constitutional framework")
            command.status = CommandStatus.CANCELLED
            command.error = "Constitutional framework violation"
    
    def approve_command(self, command_id: str, approver: str) -> bool:
        """
        Approve a pending command.
        
        Args:
            command_id: ID of command to approve
            approver: ID of approver
        
        Returns:
            True if approved successfully
        """
        command = self.commands.get(command_id)
        if not command:
            logger.warning(f"Cannot approve unknown command {command_id}")
            return False
        
        if command_id not in self.pending_approval:
            logger.warning(f"Command {command_id} not pending approval")
            return False
        
        command.status = CommandStatus.APPROVED
        command.approved_by = approver
        self.pending_approval.remove(command_id)
        
        logger.info(f"Command {command_id} approved by {approver}")
        
        return True
    
    def execute_command(self, command_id: str) -> bool:
        """
        Execute a real-world command.
        
        Args:
            command_id: ID of command to execute
        
        Returns:
            True if execution started successfully
        """
        command = self.commands.get(command_id)
        if not command:
            logger.warning(f"Cannot execute unknown command {command_id}")
            return False
        
        # Check if requires approval
        if command.requires_approval and command.status != CommandStatus.APPROVED:
            logger.warning(f"Command {command_id} requires approval before execution")
            return False
        
        # Check if adapter exists
        adapter = self.adapters.get(command.system)
        if not adapter:
            logger.error(f"No adapter registered for {command.system.value}")
            command.status = CommandStatus.FAILED
            command.error = f"No adapter for {command.system.value}"
            return False
        
        if not adapter.enabled:
            logger.warning(f"Adapter {command.system.value} is disabled")
            command.status = CommandStatus.FAILED
            command.error = f"Adapter {command.system.value} disabled"
            return False
        
        # Execute command
        command.status = CommandStatus.EXECUTING
        command.executed_at = time.time()
        
        logger.info(f"Executing command {command_id} ({command.system.value}:{command.operation})")
        
        try:
            if adapter.executor:
                # Use custom executor
                result = adapter.executor(command)
                command.success = result.get('success', False)
                command.result = result
            elif adapter.dry_run_mode or command.dry_run:
                # Dry run mode - simulate success
                logger.info(f"DRY RUN: Would execute {command.operation} with {command.parameters}")
                command.success = True
                command.result = {'simulated': True}
            else:
                # Default execution (would call actual APIs in production)
                logger.info(f"Would execute: {command.system.value} {command.operation}")
                command.success = True
                command.result = {'executed': True}
            
            # Update status
            command.status = CommandStatus.COMPLETED if command.success else CommandStatus.FAILED
            command.completed_at = time.time()
            
            # Update statistics
            adapter.commands_executed += 1
            self.total_commands_executed += 1
            
            if command.success:
                adapter.commands_succeeded += 1
                self.total_commands_succeeded += 1
                logger.info(f"Command {command_id} completed successfully")
            else:
                adapter.commands_failed += 1
                self.total_commands_failed += 1
                logger.error(f"Command {command_id} failed: {command.error}")
            
            # Add to history
            self.command_history.append(command_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing command {command_id}: {e}")
            command.status = CommandStatus.FAILED
            command.error = str(e)
            command.completed_at = time.time()
            adapter.commands_failed += 1
            self.total_commands_failed += 1
            return False
    
    def rollback_command(self, command_id: str) -> bool:
        """
        Rollback a completed command.
        
        Args:
            command_id: ID of command to rollback
        
        Returns:
            True if rollback successful
        """
        command = self.commands.get(command_id)
        if not command:
            logger.warning(f"Cannot rollback unknown command {command_id}")
            return False
        
        if not command.rollback_available:
            logger.warning(f"Command {command_id} does not support rollback")
            return False
        
        if command.rolled_back:
            logger.warning(f"Command {command_id} already rolled back")
            return False
        
        if command.status != CommandStatus.COMPLETED:
            logger.warning(f"Command {command_id} not in completed status")
            return False
        
        # Create rollback command
        rollback_id = f"{command_id}_rollback"
        rollback_cmd = RealWorldCommand(
            command_id=rollback_id,
            timestamp=time.time(),
            system=command.system,
            operation=f"rollback_{command.operation}",
            parameters=command.rollback_command or {},
            agent_decision_id=command.agent_decision_id,
            agent_id=command.agent_id,
            task_id=command.task_id,
            risk_level=command.risk_level,
            dry_run=command.dry_run
        )
        
        self._register_command(rollback_cmd)
        
        # Execute rollback
        success = self.execute_command(rollback_id)
        
        if success:
            command.rolled_back = True
            command.status = CommandStatus.ROLLED_BACK
            self.total_rollbacks += 1
            logger.info(f"Command {command_id} rolled back successfully")
        
        return success
    
    def get_statistics(self) -> Dict:
        """Get bridge statistics"""
        adapter_stats = {
            system.value: {
                'enabled': adapter.enabled,
                'commands_executed': adapter.commands_executed,
                'success_rate': adapter.get_success_rate()
            }
            for system, adapter in self.adapters.items()
        }
        
        return {
            'total_commands_created': self.total_commands_created,
            'total_commands_executed': self.total_commands_executed,
            'total_commands_succeeded': self.total_commands_succeeded,
            'total_commands_failed': self.total_commands_failed,
            'total_rollbacks': self.total_rollbacks,
            'success_rate': self.total_commands_succeeded / max(1, self.total_commands_executed),
            'pending_approval': len(self.pending_approval),
            'adapters': adapter_stats,
            'dry_run_mode': self.dry_run_mode
        }
