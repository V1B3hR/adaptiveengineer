"""
Reputation Ledger System for Phase 4: The Sentient Polity

Implements a Byzantine-resilient reputation system with meritocratic progression
through privilege tiers. Agents gain reputation through successful task completion,
reliable information sharing, and contributing to incident resolutions.

This creates a digital polity where reputation unlocks tangible capabilities.
"""

import logging
import time
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class PrivilegeTier(Enum):
    """Privilege tiers for agents based on reputation"""

    NEWBORN = 0  # Basic sensory and movement. Cannot propose actions.
    TRUSTED_PEER = (
        1  # Can join collaborative tasks, signals considered in decisions
    )
    VETERAN = (
        2  # Can lead squads, request higher resources, use advanced sensors
    )
    EMERITUS = 3  # Can propose strategies to Collective Cognition Engine


@dataclass
class ReputationRecord:
    """A single reputation transaction/record"""

    agent_id: str
    timestamp: float
    action_type: (
        str  # e.g., "task_completion", "incident_resolution", "reliable_info"
    )
    reputation_change: float
    validator_id: Optional[str] = None
    signature: Optional[str] = None

    def generate_signature(self, secret_key: Optional[str] = None) -> str:
        """Generate cryptographic signature for this record"""
        data = f"{self.agent_id}:{self.timestamp}:{self.action_type}:{self.reputation_change}:{self.validator_id}"
        # In production, this would use proper cryptographic signing
        # For now, we use a simple hash
        if secret_key:
            data = f"{data}:{secret_key}"
        return hashlib.sha256(data.encode()).hexdigest()

    def verify_signature(self, secret_key: Optional[str] = None) -> bool:
        """Verify the signature of this record"""
        if not self.signature:
            return False
        expected = self.generate_signature(secret_key)
        return self.signature == expected


@dataclass
class AgentReputation:
    """Reputation state for a single agent"""

    agent_id: str
    total_reputation: float = 0.0
    privilege_tier: PrivilegeTier = PrivilegeTier.NEWBORN
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    # Statistics
    tasks_completed: int = 0
    tasks_failed: int = 0
    incidents_resolved: int = 0
    reliable_info_provided: int = 0
    squads_led: int = 0
    strategies_proposed: int = 0

    # Recent activity (for Byzantine detection)
    recent_records: deque = field(default_factory=lambda: deque(maxlen=100))

    def add_reputation(self, amount: float, action_type: str):
        """Add reputation and update statistics"""
        self.total_reputation += amount
        self.last_updated = time.time()

        # Update statistics
        if action_type == "task_completion":
            if amount > 0:
                self.tasks_completed += 1
            else:
                self.tasks_failed += 1
        elif action_type == "incident_resolution":
            self.incidents_resolved += 1
        elif action_type == "reliable_info":
            self.reliable_info_provided += 1
        elif action_type == "squad_leadership":
            self.squads_led += 1
        elif action_type == "strategy_proposal":
            self.strategies_proposed += 1

        # Update privilege tier
        self._update_tier()

    def _update_tier(self):
        """Update privilege tier based on reputation"""
        if self.total_reputation >= 1000.0:
            self.privilege_tier = PrivilegeTier.EMERITUS
        elif self.total_reputation >= 500.0:
            self.privilege_tier = PrivilegeTier.VETERAN
        elif self.total_reputation >= 100.0:
            self.privilege_tier = PrivilegeTier.TRUSTED_PEER
        else:
            self.privilege_tier = PrivilegeTier.NEWBORN

    def get_success_rate(self) -> float:
        """Calculate task success rate"""
        total = self.tasks_completed + self.tasks_failed
        if total == 0:
            return 0.0
        return self.tasks_completed / total

    def has_capability(self, capability: str) -> bool:
        """Check if agent has a specific capability based on tier"""
        tier_capabilities = {
            PrivilegeTier.NEWBORN: {"sense", "move"},
            PrivilegeTier.TRUSTED_PEER: {"sense", "move", "join_task", "vote"},
            PrivilegeTier.VETERAN: {
                "sense",
                "move",
                "join_task",
                "vote",
                "lead_squad",
                "request_resources",
                "advanced_sensors",
            },
            PrivilegeTier.EMERITUS: {
                "sense",
                "move",
                "join_task",
                "vote",
                "lead_squad",
                "request_resources",
                "advanced_sensors",
                "propose_strategy",
            },
        }
        return capability in tier_capabilities.get(self.privilege_tier, set())


class ReputationLedger:
    """
    Byzantine-resilient reputation ledger for the digital polity.

    Maintains a tamper-resistant record of agent reputations and provides
    meritocratic progression through privilege tiers.
    """

    def __init__(
        self,
        byzantine_tolerance: float = 0.33,
        tier_thresholds: Optional[Dict[PrivilegeTier, float]] = None,
        secret_key: Optional[str] = None,
    ):
        """
        Initialize reputation ledger.

        Args:
            byzantine_tolerance: Max fraction of validators that can be malicious (default 0.33)
            tier_thresholds: Custom reputation thresholds for tiers
            secret_key: Secret key for cryptographic signing (in production)
        """
        self.byzantine_tolerance = byzantine_tolerance
        self.secret_key = secret_key

        # Agent reputations
        self.agents: Dict[str, AgentReputation] = {}

        # Ledger records (immutable history)
        self.records: List[ReputationRecord] = []

        # Validators (trusted nodes that can validate reputation changes)
        self.validators: Set[str] = set()

        # Tier thresholds (can be customized)
        self.tier_thresholds = tier_thresholds or {
            PrivilegeTier.NEWBORN: 0.0,
            PrivilegeTier.TRUSTED_PEER: 100.0,
            PrivilegeTier.VETERAN: 500.0,
            PrivilegeTier.EMERITUS: 1000.0,
        }

        # Byzantine detection
        self.suspicious_validators: Set[str] = set()
        self.validator_validation_counts: Dict[str, int] = {}

        logger.info(
            f"Reputation ledger initialized (byzantine_tolerance={byzantine_tolerance})"
        )

    def register_agent(self, agent_id: str) -> AgentReputation:
        """Register a new agent with initial reputation"""
        if agent_id in self.agents:
            logger.warning(f"Agent {agent_id} already registered")
            return self.agents[agent_id]

        reputation = AgentReputation(agent_id=agent_id)
        self.agents[agent_id] = reputation

        logger.info(
            f"Agent {agent_id} registered with tier {reputation.privilege_tier.name}"
        )
        return reputation

    def add_validator(self, validator_id: str):
        """Add a trusted validator to the network"""
        self.validators.add(validator_id)
        self.validator_validation_counts[validator_id] = 0
        logger.info(f"Validator {validator_id} added to reputation ledger")

    def remove_validator(self, validator_id: str):
        """Remove a validator (e.g., if detected as Byzantine)"""
        if validator_id in self.validators:
            self.validators.remove(validator_id)
            logger.warning(
                f"Validator {validator_id} removed from reputation ledger"
            )

    def record_reputation_change(
        self,
        agent_id: str,
        amount: float,
        action_type: str,
        validator_id: Optional[str] = None,
        requires_consensus: bool = False,
    ) -> bool:
        """
        Record a reputation change for an agent.

        Args:
            agent_id: ID of the agent
            amount: Reputation change amount (positive or negative)
            action_type: Type of action that earned/lost reputation
            validator_id: ID of validator confirming this change
            requires_consensus: Whether to require Byzantine-resilient consensus

        Returns:
            True if reputation change was recorded
        """
        # Ensure agent exists
        if agent_id not in self.agents:
            self.register_agent(agent_id)

        # Validate validator (Byzantine detection)
        if validator_id:
            if validator_id not in self.validators:
                logger.warning(
                    f"Unknown validator {validator_id}, cannot validate"
                )
                return False

            if validator_id in self.suspicious_validators:
                logger.warning(
                    f"Suspicious validator {validator_id}, rejecting"
                )
                return False

        # Create record
        record = ReputationRecord(
            agent_id=agent_id,
            timestamp=time.time(),
            action_type=action_type,
            reputation_change=amount,
            validator_id=validator_id,
        )

        # Sign record
        record.signature = record.generate_signature(self.secret_key)

        # Byzantine-resilient consensus check (if required)
        if requires_consensus and len(self.validators) > 0:
            required_validations = (
                int(len(self.validators) * (1 - self.byzantine_tolerance)) + 1
            )
            # In a real system, we'd wait for multiple validators to confirm
            # For now, we just check that we have a validator
            if not validator_id:
                logger.warning(f"Consensus required but no validator provided")
                return False

        # Add to ledger
        self.records.append(record)

        # Update agent reputation
        agent = self.agents[agent_id]
        agent.add_reputation(amount, action_type)
        agent.recent_records.append(record)

        # Update validator stats
        if validator_id:
            self.validator_validation_counts[validator_id] = (
                self.validator_validation_counts.get(validator_id, 0) + 1
            )

        logger.info(
            f"Reputation change: {agent_id} {amount:+.1f} ({action_type}) "
            f"-> {agent.total_reputation:.1f} (tier={agent.privilege_tier.name})"
        )

        return True

    def get_reputation(self, agent_id: str) -> Optional[AgentReputation]:
        """Get reputation for an agent"""
        return self.agents.get(agent_id)

    def get_tier(self, agent_id: str) -> PrivilegeTier:
        """Get privilege tier for an agent"""
        agent = self.agents.get(agent_id)
        return agent.privilege_tier if agent else PrivilegeTier.NEWBORN

    def has_capability(self, agent_id: str, capability: str) -> bool:
        """Check if agent has a specific capability"""
        agent = self.agents.get(agent_id)
        if not agent:
            return False
        return agent.has_capability(capability)

    def get_agents_by_tier(self, tier: PrivilegeTier) -> List[str]:
        """Get all agents at a specific tier"""
        return [
            agent_id
            for agent_id, agent in self.agents.items()
            if agent.privilege_tier == tier
        ]

    def get_top_agents(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get top agents by reputation"""
        sorted_agents = sorted(
            self.agents.items(),
            key=lambda x: x[1].total_reputation,
            reverse=True,
        )
        return [
            (agent_id, agent.total_reputation)
            for agent_id, agent in sorted_agents[:limit]
        ]

    def detect_byzantine_behavior(self, agent_id: str) -> bool:
        """
        Detect potentially Byzantine (malicious) behavior.

        Looks for:
        - Rapid reputation changes
        - Suspicious validation patterns
        - Inconsistent records

        Returns:
            True if Byzantine behavior detected
        """
        agent = self.agents.get(agent_id)
        if not agent:
            return False

        # Check for rapid reputation changes (possible gaming)
        recent_records = list(agent.recent_records)[-10:]  # Last 10 records
        if len(recent_records) >= 10:
            time_span = (
                recent_records[-1].timestamp - recent_records[0].timestamp
            )
            if time_span < 60.0:  # 10 changes in less than 1 minute
                logger.warning(
                    f"Rapid reputation changes detected for {agent_id}"
                )
                return True

            # Check for too many validators from same source
            validator_counts = {}
            for record in recent_records:
                if record.validator_id:
                    validator_counts[record.validator_id] = (
                        validator_counts.get(record.validator_id, 0) + 1
                    )

            for validator_id, count in validator_counts.items():
                if count > 7:  # More than 70% from one validator
                    logger.warning(
                        f"Suspicious validator pattern for {agent_id} "
                        f"(validator {validator_id})"
                    )
                    self.suspicious_validators.add(validator_id)
                    return True

        return False

    def verify_ledger_integrity(self) -> bool:
        """
        Verify integrity of the entire ledger.

        Checks all signatures and consistency.

        Returns:
            True if ledger is valid
        """
        logger.info(
            f"Verifying ledger integrity ({len(self.records)} records)..."
        )

        for i, record in enumerate(self.records):
            # Verify signature
            if not record.verify_signature(self.secret_key):
                logger.error(
                    f"Invalid signature at record {i}: {record.agent_id}"
                )
                return False

        # Verify reputation totals match records
        computed_reputations = {}
        for record in self.records:
            computed_reputations[record.agent_id] = (
                computed_reputations.get(record.agent_id, 0.0)
                + record.reputation_change
            )

        for agent_id, computed_rep in computed_reputations.items():
            agent = self.agents.get(agent_id)
            if agent and abs(agent.total_reputation - computed_rep) > 0.01:
                logger.error(
                    f"Reputation mismatch for {agent_id}: "
                    f"{agent.total_reputation} vs {computed_rep}"
                )
                return False

        logger.info("Ledger integrity verified ✓")
        return True

    def get_statistics(self) -> Dict:
        """Get ledger statistics"""
        tier_distribution = {}
        for tier in PrivilegeTier:
            tier_distribution[tier.name] = len(self.get_agents_by_tier(tier))

        return {
            "total_agents": len(self.agents),
            "total_records": len(self.records),
            "total_validators": len(self.validators),
            "suspicious_validators": len(self.suspicious_validators),
            "tier_distribution": tier_distribution,
            "top_agents": self.get_top_agents(5),
            "avg_reputation": (
                sum(a.total_reputation for a in self.agents.values())
                / len(self.agents)
                if self.agents
                else 0.0
            ),
        }

    def export_ledger(self) -> Dict:
        """Export ledger to JSON-serializable format"""
        return {
            "agents": {
                agent_id: {
                    "total_reputation": agent.total_reputation,
                    "privilege_tier": agent.privilege_tier.name,
                    "tasks_completed": agent.tasks_completed,
                    "tasks_failed": agent.tasks_failed,
                    "incidents_resolved": agent.incidents_resolved,
                    "created_at": agent.created_at,
                    "last_updated": agent.last_updated,
                }
                for agent_id, agent in self.agents.items()
            },
            "records": [
                {
                    "agent_id": record.agent_id,
                    "timestamp": record.timestamp,
                    "action_type": record.action_type,
                    "reputation_change": record.reputation_change,
                    "validator_id": record.validator_id,
                    "signature": record.signature,
                }
                for record in self.records
            ],
            "statistics": self.get_statistics(),
        }
