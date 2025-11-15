"""
Contract Net Protocol for Phase 4: Strategic Negotiation & Collaborative Tasking

Implements a market-based tasking system where agents can:
- Announce complex tasks
- Bid on contracts based on skills, location, and energy
- Award contracts to best bidders
- Report progress
- Handle failures

This creates a dynamic, decentralized marketplace for problem-solving.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Callable

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a task in the contract net"""
    ANNOUNCED = "announced"           # Task announced, accepting bids
    BIDDING_CLOSED = "bidding_closed" # No longer accepting bids
    AWARDED = "awarded"               # Contract awarded to an agent
    IN_PROGRESS = "in_progress"       # Agent working on task
    COMPLETED = "completed"           # Task successfully completed
    FAILED = "failed"                 # Task failed
    CANCELLED = "cancelled"           # Task cancelled


class MessageType(Enum):
    """Message types in the contract net protocol"""
    PROPOSE_TASK = "propose_task"
    BID = "bid"
    AWARD = "award"
    REJECT = "reject"
    REPORT_PROGRESS = "report_progress"
    DECLARE_FAILURE = "declare_failure"
    REQUEST_HELP = "request_help"


@dataclass
class TaskRequirements:
    """Requirements for a task"""
    required_skills: Set[str] = field(default_factory=set)
    min_reputation: float = 0.0
    min_privilege_tier: int = 0  # PrivilegeTier value
    max_distance: Optional[float] = None  # Maximum distance from task location
    min_energy: float = 10.0
    required_resources: Dict[str, float] = field(default_factory=dict)
    
    def is_qualified(self, 
                    skills: Set[str],
                    reputation: float,
                    privilege_tier: int,
                    distance: Optional[float] = None,
                    energy: float = 0.0,
                    resources: Optional[Dict[str, float]] = None) -> bool:
        """Check if agent meets task requirements"""
        # Check skills
        if not self.required_skills.issubset(skills):
            return False
        
        # Check reputation
        if reputation < self.min_reputation:
            return False
        
        # Check privilege tier
        if privilege_tier < self.min_privilege_tier:
            return False
        
        # Check distance
        if self.max_distance is not None and distance is not None:
            if distance > self.max_distance:
                return False
        
        # Check energy
        if energy < self.min_energy:
            return False
        
        # Check resources
        if resources:
            for resource, amount in self.required_resources.items():
                if resources.get(resource, 0.0) < amount:
                    return False
        
        return True


@dataclass
class Task:
    """A task in the contract net"""
    task_id: str
    description: str
    requirements: TaskRequirements
    reward: float  # Reputation reward for completion
    deadline: Optional[float] = None
    location: Optional[Tuple[float, float]] = None
    created_at: float = field(default_factory=time.time)
    created_by: Optional[str] = None
    
    # Task lifecycle
    status: TaskStatus = TaskStatus.ANNOUNCED
    awarded_to: Optional[str] = None
    awarded_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Progress tracking
    progress: float = 0.0  # 0.0 to 1.0
    progress_updates: List[Dict] = field(default_factory=list)
    
    # Bidding
    bids: List['Bid'] = field(default_factory=list)
    bidding_deadline: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if task deadline has passed"""
        if not self.deadline:
            return False
        return time.time() > self.deadline
    
    def is_bidding_open(self) -> bool:
        """Check if bidding is still open"""
        if self.status != TaskStatus.ANNOUNCED:
            return False
        if self.bidding_deadline and time.time() > self.bidding_deadline:
            return False
        return not self.is_expired()
    
    def get_best_bid(self) -> Optional['Bid']:
        """Get the best bid (highest score)"""
        if not self.bids:
            return None
        return max(self.bids, key=lambda b: b.bid_score)


@dataclass
class Bid:
    """A bid on a task"""
    bid_id: str
    task_id: str
    bidder_id: str
    bid_amount: float  # How much reputation bidder wants
    estimated_time: float  # Estimated completion time
    confidence: float  # Confidence in completing (0.0-1.0)
    proposed_approach: Optional[str] = None
    submitted_at: float = field(default_factory=time.time)
    
    # Bidder qualifications
    bidder_skills: Set[str] = field(default_factory=set)
    bidder_reputation: float = 0.0
    bidder_energy: float = 0.0
    bidder_distance: Optional[float] = None
    
    # Bid score (computed by contract net)
    bid_score: float = 0.0
    
    def calculate_score(self,
                       task_reward: float,
                       skill_weight: float = 0.3,
                       cost_weight: float = 0.2,
                       time_weight: float = 0.2,
                       confidence_weight: float = 0.3) -> float:
        """
        Calculate bid score based on multiple factors.
        
        Higher score = better bid
        """
        # Cost factor (lower bid amount is better)
        cost_factor = max(0.0, 1.0 - (self.bid_amount / task_reward)) if task_reward > 0 else 0.0
        
        # Time factor (shorter time is better)
        # Normalize to 0-1 range assuming max reasonable time is 3600 seconds
        time_factor = max(0.0, 1.0 - (self.estimated_time / 3600.0))
        
        # Confidence factor
        confidence_factor = self.confidence
        
        # Skill factor (more skills is better)
        # Normalize to 0-1 range assuming max useful skills is 10
        skill_factor = min(1.0, len(self.bidder_skills) / 10.0)
        
        # Weighted score
        score = (skill_weight * skill_factor +
                cost_weight * cost_factor +
                time_weight * time_factor +
                confidence_weight * confidence_factor)
        
        self.bid_score = score
        return score


class ContractNetProtocol:
    """
    Contract Net Protocol implementation for decentralized task allocation.
    
    Supports:
    - Task announcement and bidding
    - Market-based contract awarding
    - Progress tracking
    - Failure handling
    """
    
    def __init__(self,
                 default_bidding_time: float = 60.0,
                 reputation_ledger: Optional[object] = None):
        """
        Initialize contract net protocol.
        
        Args:
            default_bidding_time: Default time window for bidding (seconds)
            reputation_ledger: Optional ReputationLedger for integration
        """
        self.default_bidding_time = default_bidding_time
        self.reputation_ledger = reputation_ledger
        
        # Tasks
        self.tasks: Dict[str, Task] = {}
        self.active_tasks: Set[str] = set()
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        
        # Bids
        self.bids: Dict[str, Bid] = {}
        
        # Agent tracking
        self.agent_active_tasks: Dict[str, Set[str]] = {}  # agent_id -> set of task_ids
        self.agent_completed_tasks: Dict[str, int] = {}
        self.agent_failed_tasks: Dict[str, int] = {}
        
        # Statistics
        self.total_tasks_announced = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        self.total_bids_received = 0
        self.total_contracts_awarded = 0
        
        logger.info(f"Contract net protocol initialized (bidding_time={default_bidding_time}s)")
    
    def propose_task(self,
                    description: str,
                    requirements: TaskRequirements,
                    reward: float,
                    deadline: Optional[float] = None,
                    bidding_time: Optional[float] = None,
                    location: Optional[Tuple[float, float]] = None,
                    created_by: Optional[str] = None) -> Task:
        """
        Propose a new task (PROPOSE_TASK message).
        
        Args:
            description: Description of the task
            requirements: Task requirements
            reward: Reputation reward for completion
            deadline: Task deadline (timestamp)
            bidding_time: Time window for bidding (seconds)
            location: Task location (x, y)
            created_by: ID of agent/system proposing task
        
        Returns:
            Created Task
        """
        task_id = f"task_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
        
        bidding_time = bidding_time or self.default_bidding_time
        bidding_deadline = time.time() + bidding_time
        
        task = Task(
            task_id=task_id,
            description=description,
            requirements=requirements,
            reward=reward,
            deadline=deadline,
            bidding_deadline=bidding_deadline,
            location=location,
            created_by=created_by
        )
        
        self.tasks[task_id] = task
        self.active_tasks.add(task_id)
        self.total_tasks_announced += 1
        
        logger.info(f"Task proposed: {task_id} - {description} "
                   f"(reward={reward}, bidding_time={bidding_time}s)")
        
        return task
    
    def submit_bid(self,
                  task_id: str,
                  bidder_id: str,
                  bid_amount: float,
                  estimated_time: float,
                  confidence: float,
                  bidder_skills: Set[str],
                  bidder_reputation: float,
                  bidder_energy: float,
                  bidder_distance: Optional[float] = None,
                  proposed_approach: Optional[str] = None) -> Optional[Bid]:
        """
        Submit a bid on a task (BID message).
        
        Args:
            task_id: ID of task to bid on
            bidder_id: ID of bidding agent
            bid_amount: Requested reputation reward
            estimated_time: Estimated completion time (seconds)
            confidence: Confidence in completion (0.0-1.0)
            bidder_skills: Skills of bidder
            bidder_reputation: Current reputation of bidder
            bidder_energy: Available energy of bidder
            bidder_distance: Distance to task location
            proposed_approach: Optional description of approach
        
        Returns:
            Created Bid if successful, None if rejected
        """
        task = self.tasks.get(task_id)
        if not task:
            logger.warning(f"Cannot bid on unknown task {task_id}")
            return None
        
        if not task.is_bidding_open():
            logger.warning(f"Bidding closed for task {task_id}")
            return None
        
        # Check if agent meets requirements
        # Get privilege tier from reputation ledger if available
        privilege_tier = 0
        if self.reputation_ledger:
            agent_rep = self.reputation_ledger.get_reputation(bidder_id)
            if agent_rep:
                privilege_tier = agent_rep.privilege_tier.value
        
        if not task.requirements.is_qualified(
            skills=bidder_skills,
            reputation=bidder_reputation,
            privilege_tier=privilege_tier,
            distance=bidder_distance,
            energy=bidder_energy
        ):
            logger.warning(f"Agent {bidder_id} does not meet requirements for task {task_id}")
            return None
        
        # Create bid
        bid_id = f"bid_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
        bid = Bid(
            bid_id=bid_id,
            task_id=task_id,
            bidder_id=bidder_id,
            bid_amount=bid_amount,
            estimated_time=estimated_time,
            confidence=confidence,
            proposed_approach=proposed_approach,
            bidder_skills=bidder_skills,
            bidder_reputation=bidder_reputation,
            bidder_energy=bidder_energy,
            bidder_distance=bidder_distance
        )
        
        # Calculate bid score
        bid.calculate_score(task.reward)
        
        # Add bid to task
        task.bids.append(bid)
        self.bids[bid_id] = bid
        self.total_bids_received += 1
        
        logger.info(f"Bid submitted: {bidder_id} on {task_id} "
                   f"(amount={bid_amount}, time={estimated_time}s, score={bid.bid_score:.2f})")
        
        return bid
    
    def award_contract(self,
                      task_id: str,
                      awarded_to: Optional[str] = None,
                      auto_select_best: bool = True) -> Optional[str]:
        """
        Award contract to an agent (AWARD message).
        
        Args:
            task_id: ID of task to award
            awarded_to: ID of agent to award to (if None and auto_select_best, selects best bid)
            auto_select_best: Automatically select best bid if awarded_to is None
        
        Returns:
            ID of agent awarded, or None if award failed
        """
        task = self.tasks.get(task_id)
        if not task:
            logger.warning(f"Cannot award unknown task {task_id}")
            return None
        
        if task.status != TaskStatus.ANNOUNCED:
            logger.warning(f"Cannot award task {task_id} with status {task.status}")
            return None
        
        if not task.bids:
            logger.warning(f"No bids received for task {task_id}")
            task.status = TaskStatus.CANCELLED
            return None
        
        # Determine winner
        if awarded_to is None and auto_select_best:
            best_bid = task.get_best_bid()
            if best_bid:
                awarded_to = best_bid.bidder_id
        
        if not awarded_to:
            logger.warning(f"Could not determine winner for task {task_id}")
            return None
        
        # Award contract
        task.status = TaskStatus.AWARDED
        task.awarded_to = awarded_to
        task.awarded_at = time.time()
        
        # Track agent tasks
        if awarded_to not in self.agent_active_tasks:
            self.agent_active_tasks[awarded_to] = set()
        self.agent_active_tasks[awarded_to].add(task_id)
        
        self.total_contracts_awarded += 1
        
        logger.info(f"Contract awarded: task {task_id} to agent {awarded_to}")
        
        # Notify rejected bidders
        for bid in task.bids:
            if bid.bidder_id != awarded_to:
                logger.debug(f"Bid rejected: {bid.bidder_id} on {task_id}")
        
        return awarded_to
    
    def start_task(self, task_id: str, agent_id: str) -> bool:
        """
        Mark task as in progress.
        
        Args:
            task_id: ID of task
            agent_id: ID of agent starting task
        
        Returns:
            True if successful
        """
        task = self.tasks.get(task_id)
        if not task:
            logger.warning(f"Cannot start unknown task {task_id}")
            return False
        
        if task.awarded_to != agent_id:
            logger.warning(f"Task {task_id} not awarded to {agent_id}")
            return False
        
        if task.status != TaskStatus.AWARDED:
            logger.warning(f"Task {task_id} has status {task.status}, cannot start")
            return False
        
        task.status = TaskStatus.IN_PROGRESS
        logger.info(f"Task {task_id} started by {agent_id}")
        return True
    
    def report_progress(self,
                       task_id: str,
                       agent_id: str,
                       progress: float,
                       message: Optional[str] = None) -> bool:
        """
        Report progress on a task (REPORT_PROGRESS message).
        
        Args:
            task_id: ID of task
            agent_id: ID of agent reporting
            progress: Progress value (0.0-1.0)
            message: Optional progress message
        
        Returns:
            True if successful
        """
        task = self.tasks.get(task_id)
        if not task:
            logger.warning(f"Cannot report progress on unknown task {task_id}")
            return False
        
        if task.awarded_to != agent_id:
            logger.warning(f"Task {task_id} not awarded to {agent_id}")
            return False
        
        progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
        task.progress = progress
        
        update = {
            'timestamp': time.time(),
            'progress': progress,
            'message': message
        }
        task.progress_updates.append(update)
        
        logger.info(f"Progress reported: {task_id} by {agent_id} - {progress:.1%}"
                   f"{': ' + message if message else ''}")
        
        return True
    
    def complete_task(self,
                     task_id: str,
                     agent_id: str,
                     result: Optional[Dict] = None) -> bool:
        """
        Mark task as completed.
        
        Args:
            task_id: ID of task
            agent_id: ID of agent completing task
            result: Optional result data
        
        Returns:
            True if successful
        """
        task = self.tasks.get(task_id)
        if not task:
            logger.warning(f"Cannot complete unknown task {task_id}")
            return False
        
        if task.awarded_to != agent_id:
            logger.warning(f"Task {task_id} not awarded to {agent_id}")
            return False
        
        task.status = TaskStatus.COMPLETED
        task.completed_at = time.time()
        task.progress = 1.0
        
        # Track completion
        self.completed_tasks.add(task_id)
        if task_id in self.active_tasks:
            self.active_tasks.remove(task_id)
        if agent_id in self.agent_active_tasks:
            self.agent_active_tasks[agent_id].discard(task_id)
        
        self.agent_completed_tasks[agent_id] = \
            self.agent_completed_tasks.get(agent_id, 0) + 1
        self.total_tasks_completed += 1
        
        # Award reputation (if ledger available)
        if self.reputation_ledger:
            self.reputation_ledger.record_reputation_change(
                agent_id=agent_id,
                amount=task.reward,
                action_type="task_completion"
            )
        
        completion_time = task.completed_at - task.awarded_at if task.awarded_at else 0
        logger.info(f"Task completed: {task_id} by {agent_id} "
                   f"(time={completion_time:.1f}s, reward={task.reward})")
        
        return True
    
    def declare_failure(self,
                       task_id: str,
                       agent_id: str,
                       reason: str) -> bool:
        """
        Declare task failure (DECLARE_FAILURE message).
        
        Args:
            task_id: ID of task
            agent_id: ID of agent declaring failure
            reason: Reason for failure
        
        Returns:
            True if successful
        """
        task = self.tasks.get(task_id)
        if not task:
            logger.warning(f"Cannot declare failure on unknown task {task_id}")
            return False
        
        if task.awarded_to != agent_id:
            logger.warning(f"Task {task_id} not awarded to {agent_id}")
            return False
        
        task.status = TaskStatus.FAILED
        task.completed_at = time.time()
        
        # Track failure
        self.failed_tasks.add(task_id)
        if task_id in self.active_tasks:
            self.active_tasks.remove(task_id)
        if agent_id in self.agent_active_tasks:
            self.agent_active_tasks[agent_id].discard(task_id)
        
        self.agent_failed_tasks[agent_id] = \
            self.agent_failed_tasks.get(agent_id, 0) + 1
        self.total_tasks_failed += 1
        
        # Penalize reputation (if ledger available)
        if self.reputation_ledger:
            penalty = -task.reward * 0.5  # 50% of reward as penalty
            self.reputation_ledger.record_reputation_change(
                agent_id=agent_id,
                amount=penalty,
                action_type="task_completion"  # Same type for learning
            )
        
        logger.warning(f"Task failed: {task_id} by {agent_id} - {reason}")
        
        return True
    
    def request_help(self,
                    task_id: str,
                    agent_id: str,
                    help_needed: str) -> bool:
        """
        Request help on a task (REQUEST_HELP message).
        
        Args:
            task_id: ID of task
            agent_id: ID of agent requesting help
            help_needed: Description of help needed
        
        Returns:
            True if successful
        """
        task = self.tasks.get(task_id)
        if not task:
            logger.warning(f"Cannot request help on unknown task {task_id}")
            return False
        
        if task.awarded_to != agent_id:
            logger.warning(f"Task {task_id} not awarded to {agent_id}")
            return False
        
        logger.info(f"Help requested: {task_id} by {agent_id} - {help_needed}")
        
        # In a real system, this would broadcast to other agents
        # For now, we just log it
        return True
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def get_active_tasks(self) -> List[Task]:
        """Get all active tasks"""
        return [self.tasks[task_id] for task_id in self.active_tasks 
                if task_id in self.tasks]
    
    def get_available_tasks(self, agent_id: str) -> List[Task]:
        """Get tasks available for bidding"""
        return [task for task in self.tasks.values()
                if task.is_bidding_open()]
    
    def get_agent_tasks(self, agent_id: str) -> List[Task]:
        """Get tasks currently assigned to an agent"""
        task_ids = self.agent_active_tasks.get(agent_id, set())
        return [self.tasks[task_id] for task_id in task_ids 
                if task_id in self.tasks]
    
    def get_agent_success_rate(self, agent_id: str) -> float:
        """Get task success rate for an agent"""
        completed = self.agent_completed_tasks.get(agent_id, 0)
        failed = self.agent_failed_tasks.get(agent_id, 0)
        total = completed + failed
        if total == 0:
            return 0.0
        return completed / total
    
    def get_statistics(self) -> Dict:
        """Get contract net statistics"""
        return {
            'total_tasks_announced': self.total_tasks_announced,
            'total_tasks_completed': self.total_tasks_completed,
            'total_tasks_failed': self.total_tasks_failed,
            'total_bids_received': self.total_bids_received,
            'total_contracts_awarded': self.total_contracts_awarded,
            'active_tasks': len(self.active_tasks),
            'completion_rate': self.total_tasks_completed / max(1, self.total_tasks_announced),
            'avg_bids_per_task': self.total_bids_received / max(1, self.total_tasks_announced)
        }
