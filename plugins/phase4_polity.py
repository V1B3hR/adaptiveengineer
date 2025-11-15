"""
Phase 4 Polity Plugin: Integration of all governance features

Integrates:
- Reputation Ledger & Meritocratic Progression
- Contract Net Protocol for collaborative tasking
- Council of Professors governance
- Constitutional Framework enforcement
- Real-World Integration Bridge

Provides a unified interface for Phase 4 "Sentient Polity" features.
"""

import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass

from core.plugin_base import PluginBase
from core.reputation_ledger import ReputationLedger, PrivilegeTier
from core.contract_net import ContractNetProtocol, TaskRequirements
from core.governance import CouncilOfProfessors
from core.constitutional_framework import ConstitutionalFramework
from core.real_world_bridge import RealWorldBridge, IntegrationSystem

logger = logging.getLogger(__name__)


class Phase4PolityPlugin(PluginBase):
    """
    Plugin integrating Phase 4 Polity & Governance features.
    
    This plugin provides the complete "Sentient Polity" system with:
    - Byzantine-resilient reputation and privilege tiers
    - Market-based collaborative tasking
    - Professor governance and oversight
    - Constitutional compliance checking
    - Real-world system integration
    """
    
    # Class-level shared instances (singleton pattern for global systems)
    _reputation_ledger: Optional[ReputationLedger] = None
    _contract_net: Optional[ContractNetProtocol] = None
    _council: Optional[CouncilOfProfessors] = None
    _constitution: Optional[ConstitutionalFramework] = None
    _bridge: Optional[RealWorldBridge] = None
    _initialized = False
    
    def __init__(self):
        """Initialize Phase 4 Polity Plugin"""
        super().__init__()
        self.plugin_name = "Phase4Polity"
        
        # Per-node tracking
        self.node_reputation: Dict[int, float] = {}
        self.node_privilege_tier: Dict[int, int] = {}
        self.node_active_tasks: Dict[int, Set[str]] = {}
        self.node_completed_tasks: Dict[int, int] = {}
    
    @classmethod
    def initialize_global_systems(cls,
                                  knowledge_graph: Optional[object] = None,
                                  evolution_engine: Optional[object] = None,
                                  critical_services: Optional[Set[str]] = None,
                                  system_components: Optional[Set[str]] = None,
                                  dry_run_mode: bool = True):
        """
        Initialize global singleton systems (called once).
        
        Args:
            knowledge_graph: Optional KnowledgeGraph for integration
            evolution_engine: Optional EvolutionEngine for integration
            critical_services: Set of mission-critical services
            system_components: Set of critical system components
            dry_run_mode: Whether to run bridge in dry-run mode
        """
        if cls._initialized:
            logger.info("Phase 4 global systems already initialized")
            return
        
        logger.info("Initializing Phase 4 global systems...")
        
        # Initialize Reputation Ledger
        cls._reputation_ledger = ReputationLedger(
            byzantine_tolerance=0.33
        )
        logger.info("  ✓ Reputation Ledger initialized")
        
        # Initialize Contract Net Protocol
        cls._contract_net = ContractNetProtocol(
            default_bidding_time=60.0,
            reputation_ledger=cls._reputation_ledger
        )
        logger.info("  ✓ Contract Net Protocol initialized")
        
        # Initialize Council of Professors
        cls._council = CouncilOfProfessors(
            knowledge_graph=knowledge_graph,
            evolution_engine=evolution_engine
        )
        logger.info("  ✓ Council of Professors assembled")
        
        # Initialize Constitutional Framework
        cls._constitution = ConstitutionalFramework(
            critical_services=critical_services,
            system_components=system_components
        )
        logger.info("  ✓ Constitutional Framework established")
        
        # Initialize Real-World Bridge
        cls._bridge = RealWorldBridge(
            require_approval_for_high_risk=True,
            dry_run_mode=dry_run_mode,
            constitutional_framework=cls._constitution
        )
        logger.info("  ✓ Real-World Integration Bridge ready")
        
        cls._initialized = True
        logger.info("Phase 4 global systems initialization complete")
    
    def initialize(self, node):
        """Initialize plugin for a specific node"""
        if not self._initialized:
            # Initialize with defaults if not already initialized
            self.initialize_global_systems()
        
        # Register node in reputation ledger
        agent_id = f"agent_{node.node_id}"
        self._reputation_ledger.register_agent(agent_id)
        
        # Initialize per-node tracking
        self.node_reputation[node.node_id] = 0.0
        self.node_privilege_tier[node.node_id] = 0  # NEWBORN
        self.node_active_tasks[node.node_id] = set()
        self.node_completed_tasks[node.node_id] = 0
        
        logger.info(f"Phase 4 Polity plugin initialized for node {node.node_id}")
    
    def get_state_variables(self, node) -> Dict:
        """Get state variables exposed by this plugin"""
        agent_id = f"agent_{node.node_id}"
        reputation = self._reputation_ledger.get_reputation(agent_id)
        
        # Get task statistics
        active_tasks = len(self.node_active_tasks.get(node.node_id, set()))
        agent_tasks = self._contract_net.get_agent_tasks(agent_id)
        success_rate = self._contract_net.get_agent_success_rate(agent_id)
        
        return {
            # Reputation & Privilege
            'reputation': reputation.total_reputation if reputation else 0.0,
            'privilege_tier': reputation.privilege_tier.value if reputation else 0,
            'privilege_tier_name': reputation.privilege_tier.name if reputation else "NEWBORN",
            'tasks_completed': reputation.tasks_completed if reputation else 0,
            'tasks_failed': reputation.tasks_failed if reputation else 0,
            'task_success_rate': reputation.get_success_rate() if reputation else 0.0,
            'squads_led': reputation.squads_led if reputation else 0,
            'strategies_proposed': reputation.strategies_proposed if reputation else 0,
            
            # Contract Net
            'active_contracts': active_tasks,
            'available_tasks': len(self._contract_net.get_available_tasks(agent_id)),
            'contract_success_rate': success_rate,
            
            # Capabilities
            'can_join_tasks': reputation.has_capability('join_task') if reputation else False,
            'can_lead_squads': reputation.has_capability('lead_squad') if reputation else False,
            'can_propose_strategies': reputation.has_capability('propose_strategy') if reputation else False,
            
            # System status
            'constitutional_compliance': 'active',
            'governance_active': True
        }
    
    def get_available_actions(self, node) -> List[str]:
        """Get actions available to this node"""
        agent_id = f"agent_{node.node_id}"
        reputation = self._reputation_ledger.get_reputation(agent_id)
        
        # Base actions available to all
        actions = [
            'check_reputation',
            'view_available_tasks',
            'get_governance_status'
        ]
        
        # Actions based on privilege tier
        if reputation:
            if reputation.has_capability('join_task'):
                actions.extend(['bid_on_task', 'report_task_progress'])
            
            if reputation.has_capability('lead_squad'):
                actions.extend(['propose_task', 'lead_squad', 'request_resources'])
            
            if reputation.has_capability('propose_strategy'):
                actions.extend(['propose_strategy_to_council'])
        
        return actions
    
    def update(self, delta_time: float):
        """Update plugin state (called each cycle)"""
        # Periodic updates can go here
        # For now, systems are reactive rather than time-based
        pass
    
    # =========================================================================
    # REPUTATION & PRIVILEGE ACTIONS
    # =========================================================================
    
    def award_reputation(self,
                        node,
                        amount: float,
                        action_type: str,
                        validator_id: Optional[str] = None) -> bool:
        """
        Award reputation to a node.
        
        Args:
            node: AliveLoopNode receiving reputation
            amount: Reputation amount (positive or negative)
            action_type: Type of action earning reputation
            validator_id: Optional validator ID
        
        Returns:
            True if reputation awarded
        """
        agent_id = f"agent_{node.node_id}"
        
        success = self._reputation_ledger.record_reputation_change(
            agent_id=agent_id,
            amount=amount,
            action_type=action_type,
            validator_id=validator_id
        )
        
        if success:
            # Update local tracking
            reputation = self._reputation_ledger.get_reputation(agent_id)
            self.node_reputation[node.node_id] = reputation.total_reputation
            self.node_privilege_tier[node.node_id] = reputation.privilege_tier.value
        
        return success
    
    def check_capability(self, node, capability: str) -> bool:
        """Check if node has a specific capability"""
        agent_id = f"agent_{node.node_id}"
        return self._reputation_ledger.has_capability(agent_id, capability)
    
    # =========================================================================
    # CONTRACT NET ACTIONS
    # =========================================================================
    
    def propose_task(self,
                    node,
                    description: str,
                    requirements: Dict[str, Any],
                    reward: float,
                    deadline: Optional[float] = None) -> Optional[str]:
        """
        Propose a new task (requires VETERAN tier).
        
        Args:
            node: AliveLoopNode proposing task
            description: Task description
            requirements: Task requirements dict
            reward: Reputation reward
            deadline: Optional deadline timestamp
        
        Returns:
            Task ID if successful
        """
        agent_id = f"agent_{node.node_id}"
        
        # Check capability
        if not self.check_capability(node, 'lead_squad'):
            logger.warning(f"Agent {agent_id} cannot propose tasks (insufficient privilege)")
            return None
        
        # Create TaskRequirements
        task_reqs = TaskRequirements(
            required_skills=set(requirements.get('skills', [])),
            min_reputation=requirements.get('min_reputation', 0.0),
            min_privilege_tier=requirements.get('min_tier', 0),
            max_distance=requirements.get('max_distance'),
            min_energy=requirements.get('min_energy', 10.0)
        )
        
        # Propose task
        task = self._contract_net.propose_task(
            description=description,
            requirements=task_reqs,
            reward=reward,
            deadline=deadline,
            created_by=agent_id
        )
        
        return task.task_id
    
    def bid_on_task(self,
                   node,
                   task_id: str,
                   bid_amount: float,
                   estimated_time: float,
                   confidence: float,
                   skills: Set[str]) -> Optional[str]:
        """
        Submit bid on a task (requires TRUSTED_PEER tier).
        
        Args:
            node: AliveLoopNode bidding
            task_id: Task to bid on
            bid_amount: Requested reputation
            estimated_time: Estimated completion time
            confidence: Confidence level (0-1)
            skills: Agent skills
        
        Returns:
            Bid ID if successful
        """
        agent_id = f"agent_{node.node_id}"
        
        # Check capability
        if not self.check_capability(node, 'join_task'):
            logger.warning(f"Agent {agent_id} cannot bid on tasks (insufficient privilege)")
            return None
        
        # Get reputation
        reputation = self._reputation_ledger.get_reputation(agent_id)
        if not reputation:
            return None
        
        # Submit bid
        bid = self._contract_net.submit_bid(
            task_id=task_id,
            bidder_id=agent_id,
            bid_amount=bid_amount,
            estimated_time=estimated_time,
            confidence=confidence,
            bidder_skills=skills,
            bidder_reputation=reputation.total_reputation,
            bidder_energy=node.energy if hasattr(node, 'energy') else 100.0
        )
        
        return bid.bid_id if bid else None
    
    def complete_task(self, node, task_id: str) -> bool:
        """
        Mark task as completed.
        
        Args:
            node: AliveLoopNode completing task
            task_id: Task to complete
        
        Returns:
            True if successful
        """
        agent_id = f"agent_{node.node_id}"
        
        success = self._contract_net.complete_task(task_id, agent_id)
        
        if success:
            # Update local tracking
            if node.node_id in self.node_active_tasks:
                self.node_active_tasks[node.node_id].discard(task_id)
            self.node_completed_tasks[node.node_id] = \
                self.node_completed_tasks.get(node.node_id, 0) + 1
        
        return success
    
    # =========================================================================
    # GOVERNANCE ACTIONS
    # =========================================================================
    
    def report_failure_for_analysis(self,
                                    failure_id: str,
                                    agent_id: Optional[str] = None,
                                    failure_type: str = "unknown",
                                    context: Optional[Dict] = None):
        """Report a failure to the Systemic Pathologist for analysis"""
        analysis = self._council.pathologist.analyze_failure(
            failure_id=failure_id,
            agent_id=agent_id,
            failure_type=failure_type,
            context=context
        )
        
        # Publish lesson learned
        self._council.pathologist.publish_lesson_learned(failure_id)
        
        return analysis
    
    def report_vulnerability(self,
                           vuln_id: str,
                           vulnerability_type: str,
                           description: str,
                           severity: float,
                           affected_components: List[str]):
        """Report a vulnerability to the Strategic Immunologist"""
        return self._council.immunologist.monitor_vulnerability(
            vuln_id=vuln_id,
            vulnerability_type=vulnerability_type,
            description=description,
            severity=severity,
            affected_components=affected_components
        )
    
    def assess_genetic_health(self,
                             population_size: int,
                             genetic_diversity: float,
                             avg_fitness: float,
                             stagnation_level: float,
                             mutation_rate: float):
        """Request genetic health assessment from Evolutionary Biologist"""
        return self._council.biologist.assess_genetic_health(
            population_size=population_size,
            genetic_diversity=genetic_diversity,
            avg_fitness=avg_fitness,
            stagnation_level=stagnation_level,
            mutation_rate=mutation_rate
        )
    
    def propose_strategy_to_council(self,
                                   node,
                                   strategy_description: str,
                                   strategy_details: Dict) -> bool:
        """
        Propose a strategy to Collective Cognition Engine (requires EMERITUS tier).
        
        Args:
            node: AliveLoopNode proposing strategy
            strategy_description: Description of strategy
            strategy_details: Strategy implementation details
        
        Returns:
            True if proposal accepted
        """
        agent_id = f"agent_{node.node_id}"
        
        # Check capability
        if not self.check_capability(node, 'propose_strategy'):
            logger.warning(f"Agent {agent_id} cannot propose strategies (requires EMERITUS tier)")
            return False
        
        # Award reputation for proposing
        self.award_reputation(node, 5.0, "strategy_proposal")
        
        logger.info(f"Strategy proposed by {agent_id}: {strategy_description}")
        # In production, this would integrate with Collective Cognition Engine
        
        return True
    
    # =========================================================================
    # CONSTITUTIONAL & BRIDGE ACTIONS
    # =========================================================================
    
    def evaluate_action_constitutionally(self,
                                        action_id: str,
                                        action_description: str,
                                        action_context: Dict[str, Any],
                                        agent_id: Optional[str] = None):
        """Evaluate an action against constitutional framework"""
        return self._constitution.evaluate_action(
            action_id=action_id,
            action_description=action_description,
            action_context=action_context,
            agent_id=agent_id
        )
    
    def translate_and_execute(self,
                             decision_id: str,
                             decision_type: str,
                             decision_data: Dict[str, Any],
                             agent_id: Optional[str] = None) -> List[str]:
        """
        Translate agent decision to real-world commands and execute.
        
        Args:
            decision_id: Decision identifier
            decision_type: Type of decision
            decision_data: Decision parameters
            agent_id: ID of agent making decision
        
        Returns:
            List of command IDs created
        """
        # Translate decision to commands
        commands = self._bridge.translate_decision(
            decision_id=decision_id,
            decision_type=decision_type,
            decision_data=decision_data,
            agent_id=agent_id
        )
        
        command_ids = []
        for command in commands:
            # Execute if approved or doesn't require approval
            if not command.requires_approval:
                self._bridge.execute_command(command.command_id)
            
            command_ids.append(command.command_id)
        
        return command_ids
    
    def approve_command(self, command_id: str, approver: str) -> bool:
        """Approve a pending high-risk command"""
        return self._bridge.approve_command(command_id, approver)
    
    def execute_approved_command(self, command_id: str) -> bool:
        """Execute an approved command"""
        return self._bridge.execute_command(command_id)
    
    # =========================================================================
    # STATISTICS & QUERIES
    # =========================================================================
    
    def get_reputation_statistics(self) -> Dict:
        """Get reputation ledger statistics"""
        return self._reputation_ledger.get_statistics()
    
    def get_contract_statistics(self) -> Dict:
        """Get contract net statistics"""
        return self._contract_net.get_statistics()
    
    def get_governance_statistics(self) -> Dict:
        """Get Council of Professors statistics"""
        return self._council.get_statistics()
    
    def get_constitutional_statistics(self) -> Dict:
        """Get constitutional framework statistics"""
        return self._constitution.get_statistics()
    
    def get_bridge_statistics(self) -> Dict:
        """Get real-world bridge statistics"""
        return self._bridge.get_statistics()
    
    def get_summary(self) -> str:
        """Get comprehensive summary of Phase 4 systems"""
        rep_stats = self.get_reputation_statistics()
        contract_stats = self.get_contract_statistics()
        gov_stats = self.get_governance_statistics()
        const_stats = self.get_constitutional_statistics()
        bridge_stats = self.get_bridge_statistics()
        
        summary = []
        summary.append("\n" + "="*70)
        summary.append("PHASE 4: SENTIENT POLITY & GOVERNANCE LAYER - STATUS")
        summary.append("="*70)
        
        summary.append("\n1. REPUTATION LEDGER & MERITOCRATIC PROGRESSION")
        summary.append(f"   Total Agents: {rep_stats['total_agents']}")
        summary.append(f"   Avg Reputation: {rep_stats['avg_reputation']:.1f}")
        summary.append("   Tier Distribution:")
        for tier, count in rep_stats['tier_distribution'].items():
            summary.append(f"     - {tier}: {count} agents")
        
        summary.append("\n2. CONTRACT NET PROTOCOL")
        summary.append(f"   Tasks Announced: {contract_stats['total_tasks_announced']}")
        summary.append(f"   Tasks Completed: {contract_stats['total_tasks_completed']}")
        summary.append(f"   Completion Rate: {contract_stats['completion_rate']:.1%}")
        summary.append(f"   Active Contracts: {contract_stats['active_tasks']}")
        
        summary.append("\n3. COUNCIL OF PROFESSORS")
        summary.append("   Systemic Pathologist:")
        summary.append(f"     - Failures Analyzed: {gov_stats['pathologist']['total_failures_analyzed']}")
        summary.append(f"     - Lessons Published: {gov_stats['pathologist']['total_lessons_published']}")
        summary.append("   Strategic Immunologist:")
        summary.append(f"     - Vulnerabilities Tracked: {gov_stats['immunologist']['total_vulnerabilities_tracked']}")
        summary.append(f"     - Antibodies Injected: {gov_stats['immunologist']['total_antibodies_injected']}")
        summary.append("   Evolutionary Biologist:")
        summary.append(f"     - Assessments: {gov_stats['biologist']['total_assessments']}")
        summary.append(f"     - Interventions: {gov_stats['biologist']['total_interventions']}")
        
        summary.append("\n4. CONSTITUTIONAL FRAMEWORK")
        summary.append(f"   Actions Evaluated: {const_stats['total_actions_evaluated']}")
        summary.append(f"   Compliance Rate: {const_stats['compliance_rate']:.1%}")
        summary.append(f"   Violations Prevented: {const_stats['total_violations_prevented']}")
        summary.append("   Violations by Law:")
        for law, count in const_stats['violations_by_law'].items():
            summary.append(f"     - {law}: {count}")
        
        summary.append("\n5. REAL-WORLD INTEGRATION BRIDGE")
        summary.append(f"   Commands Created: {bridge_stats['total_commands_created']}")
        summary.append(f"   Commands Executed: {bridge_stats['total_commands_executed']}")
        summary.append(f"   Success Rate: {bridge_stats['success_rate']:.1%}")
        summary.append(f"   Pending Approval: {bridge_stats['pending_approval']}")
        summary.append(f"   Dry Run Mode: {bridge_stats['dry_run_mode']}")
        
        summary.append("\n" + "="*70)
        
        return "\n".join(summary)
