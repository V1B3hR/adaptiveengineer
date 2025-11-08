"""
Byzantine-resilient consensus mechanism for distributed decision making.

This module implements consensus algorithms that can tolerate Byzantine faults
(malicious or compromised nodes) for incident root cause analysis, attack
validation, and collective response coordination.
"""

import time
import hashlib
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class ConsensusType(str, Enum):
    """Types of decisions requiring consensus."""
    ROOT_CAUSE = "root_cause"
    ATTACK_VALIDATION = "attack_validation"
    COLLECTIVE_RESPONSE = "collective_response"
    TRUST_ASSESSMENT = "trust_assessment"
    THRESHOLD_UPDATE = "threshold_update"


class VoteType(str, Enum):
    """Vote types."""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


@dataclass
class Vote:
    """A single vote in consensus process."""
    voter_id: int
    vote_type: VoteType
    confidence: float  # 0.0 to 1.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    signature: Optional[str] = None
    
    def compute_signature(self, secret: str = "") -> str:
        """Compute cryptographic signature for vote."""
        data = f"{self.voter_id}:{self.vote_type}:{self.confidence}:{self.timestamp}:{secret}"
        return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class ConsensusProposal:
    """A proposal requiring consensus."""
    proposal_id: str
    consensus_type: ConsensusType
    proposer_id: int
    subject: Dict[str, Any]  # What we're deciding about
    timestamp: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    
    # Voting state
    votes: List[Vote] = field(default_factory=list)
    quorum_size: int = 0
    required_approval_ratio: float = 0.67  # Byzantine fault tolerance threshold
    
    # Results
    consensus_reached: bool = False
    consensus_result: Optional[VoteType] = None
    confidence: float = 0.0
    finalized_at: Optional[float] = None


class ByzantineDetector:
    """
    Detects Byzantine (malicious or faulty) behavior in voting.
    
    Uses statistical analysis and behavioral patterns to identify
    nodes that may be compromised or unreliable.
    """
    
    def __init__(self, suspicion_threshold: float = 0.3):
        """
        Initialize Byzantine detector.
        
        Args:
            suspicion_threshold: Threshold for marking node as suspicious
        """
        self.suspicion_threshold = suspicion_threshold
        self.node_behavior: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
            'total_votes': 0,
            'minority_votes': 0,  # Votes against consensus
            'confidence_sum': 0.0,
            'suspicious_patterns': 0,
            'last_vote_time': 0.0
        })
    
    def record_vote(self, node_id: int, vote: Vote, was_minority: bool) -> None:
        """Record vote for behavioral analysis."""
        behavior = self.node_behavior[node_id]
        behavior['total_votes'] += 1
        behavior['confidence_sum'] += vote.confidence
        behavior['last_vote_time'] = vote.timestamp
        
        if was_minority:
            behavior['minority_votes'] += 1
    
    def detect_suspicious_patterns(
        self,
        votes: List[Vote],
        consensus_result: VoteType
    ) -> Set[int]:
        """
        Detect suspicious voting patterns.
        
        Args:
            votes: All votes cast
            consensus_result: The consensus that was reached
            
        Returns:
            Set of suspicious node IDs
        """
        suspicious = set()
        
        for vote in votes:
            # Pattern 1: Consistently voting against consensus
            behavior = self.node_behavior[vote.voter_id]
            if behavior['total_votes'] > 5:
                minority_ratio = behavior['minority_votes'] / behavior['total_votes']
                if minority_ratio > 0.7:
                    suspicious.add(vote.voter_id)
                    behavior['suspicious_patterns'] += 1
            
            # Pattern 2: Extreme confidence with minority vote
            if vote.vote_type != consensus_result and vote.confidence > 0.9:
                behavior['suspicious_patterns'] += 1
                if behavior['suspicious_patterns'] >= 3:
                    suspicious.add(vote.voter_id)
            
            # Pattern 3: Always abstaining
            if vote.vote_type == VoteType.ABSTAIN:
                if behavior['total_votes'] > 3:
                    abstain_ratio = sum(1 for v in votes if v.voter_id == vote.voter_id 
                                       and v.vote_type == VoteType.ABSTAIN) / behavior['total_votes']
                    if abstain_ratio > 0.8:
                        suspicious.add(vote.voter_id)
        
        return suspicious
    
    def get_suspicion_score(self, node_id: int) -> float:
        """
        Get suspicion score for a node (0.0 = trustworthy, 1.0 = very suspicious).
        
        Args:
            node_id: Node to assess
            
        Returns:
            Suspicion score
        """
        behavior = self.node_behavior.get(node_id)
        if not behavior or behavior['total_votes'] < 3:
            return 0.0
        
        # Calculate score based on multiple factors
        minority_ratio = behavior['minority_votes'] / max(1, behavior['total_votes'])
        pattern_score = min(1.0, behavior['suspicious_patterns'] / 5.0)
        
        # Average confidence (low could indicate uncertainty or gaming)
        avg_confidence = behavior['confidence_sum'] / behavior['total_votes']
        confidence_factor = 1.0 - avg_confidence
        
        suspicion = (minority_ratio * 0.5 + pattern_score * 0.3 + confidence_factor * 0.2)
        
        return min(1.0, suspicion)
    
    def is_suspicious(self, node_id: int) -> bool:
        """Check if a node is considered suspicious."""
        return self.get_suspicion_score(node_id) >= self.suspicion_threshold


class ConsensusEngine:
    """
    Byzantine-resilient consensus engine.
    
    Implements consensus mechanisms that can tolerate up to f Byzantine faults
    in a network of 3f+1 nodes (i.e., can tolerate up to 33% malicious nodes).
    """
    
    def __init__(
        self,
        node_id: int,
        byzantine_tolerance: float = 0.33,
        default_quorum_ratio: float = 0.67
    ):
        """
        Initialize consensus engine.
        
        Args:
            node_id: ID of this node
            byzantine_tolerance: Maximum fraction of Byzantine nodes to tolerate
            default_quorum_ratio: Default ratio of nodes needed for quorum
        """
        self.node_id = node_id
        self.byzantine_tolerance = byzantine_tolerance
        self.default_quorum_ratio = default_quorum_ratio
        
        self.proposals: Dict[str, ConsensusProposal] = {}
        self.byzantine_detector = ByzantineDetector()
        
        # Trust-based vote weighting
        self.node_trust_scores: Dict[int, float] = {}
        
        logger.info(f"Consensus engine initialized for node {node_id} "
                   f"(tolerance={byzantine_tolerance}, quorum={default_quorum_ratio})")
    
    def set_node_trust(self, node_id: int, trust_score: float) -> None:
        """
        Set trust score for a node (used for vote weighting).
        
        Args:
            node_id: Node to set trust for
            trust_score: Trust score (0.0 to 1.0)
        """
        self.node_trust_scores[node_id] = max(0.0, min(1.0, trust_score))
    
    def initiate_consensus(
        self,
        consensus_type: ConsensusType,
        subject: Dict[str, Any],
        network_size: int,
        timeout: float = 300.0
    ) -> str:
        """
        Initiate a consensus process.
        
        Args:
            consensus_type: Type of consensus decision
            subject: What we're deciding about
            network_size: Total number of nodes in network
            timeout: Deadline in seconds
            
        Returns:
            Proposal ID
        """
        proposal_id = f"{consensus_type}_{self.node_id}_{int(time.time())}"
        
        # Calculate quorum size (need 2f+1 votes to tolerate f Byzantine nodes)
        quorum_size = max(1, int(network_size * self.default_quorum_ratio))
        
        proposal = ConsensusProposal(
            proposal_id=proposal_id,
            consensus_type=consensus_type,
            proposer_id=self.node_id,
            subject=subject,
            deadline=time.time() + timeout,
            quorum_size=quorum_size,
            required_approval_ratio=self.default_quorum_ratio
        )
        
        self.proposals[proposal_id] = proposal
        
        logger.info(f"Initiated consensus {proposal_id} for {consensus_type} "
                   f"(quorum={quorum_size}, timeout={timeout}s)")
        
        return proposal_id
    
    def cast_vote(
        self,
        proposal_id: str,
        voter_id: int,
        vote_type: VoteType,
        confidence: float,
        evidence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Cast a vote on a proposal.
        
        Args:
            proposal_id: ID of proposal to vote on
            voter_id: ID of voting node
            vote_type: Type of vote
            confidence: Confidence in vote (0.0 to 1.0)
            evidence: Optional evidence supporting vote
            
        Returns:
            Vote results
        """
        if proposal_id not in self.proposals:
            raise ValueError(f"Unknown proposal: {proposal_id}")
        
        proposal = self.proposals[proposal_id]
        
        # Check if already finalized
        if proposal.consensus_reached:
            return {
                'accepted': False,
                'reason': 'Proposal already finalized'
            }
        
        # Check deadline
        if proposal.deadline and time.time() > proposal.deadline:
            return {
                'accepted': False,
                'reason': 'Proposal deadline passed'
            }
        
        # Check for duplicate vote
        if any(v.voter_id == voter_id for v in proposal.votes):
            return {
                'accepted': False,
                'reason': 'Already voted'
            }
        
        # Create vote
        vote = Vote(
            voter_id=voter_id,
            vote_type=vote_type,
            confidence=confidence,
            evidence=evidence or {}
        )
        vote.signature = vote.compute_signature()
        
        # Add vote
        proposal.votes.append(vote)
        
        logger.debug(f"Vote cast on {proposal_id} by node {voter_id}: {vote_type}")
        
        # Check if consensus reached
        self._check_consensus(proposal)
        
        return {
            'accepted': True,
            'proposal_id': proposal_id,
            'votes_received': len(proposal.votes),
            'quorum_size': proposal.quorum_size,
            'consensus_reached': proposal.consensus_reached
        }
    
    def _check_consensus(self, proposal: ConsensusProposal) -> None:
        """
        Check if consensus has been reached on a proposal.
        
        Uses Byzantine-resilient voting that tolerates malicious nodes.
        """
        if proposal.consensus_reached:
            return
        
        # Need quorum
        if len(proposal.votes) < proposal.quorum_size:
            return
        
        # Weight votes by trust scores
        weighted_votes: Dict[VoteType, float] = defaultdict(float)
        total_weight = 0.0
        
        for vote in proposal.votes:
            # Get trust score (default to neutral if unknown)
            trust = self.node_trust_scores.get(vote.voter_id, 0.5)
            
            # Discount suspicious nodes
            if self.byzantine_detector.is_suspicious(vote.voter_id):
                trust *= 0.3  # Heavily discount suspicious nodes
            
            # Weight by trust and confidence
            weight = trust * vote.confidence
            weighted_votes[vote.vote_type] += weight
            total_weight += weight
        
        # Calculate weighted approval
        if total_weight == 0:
            return
        
        approval_weight = weighted_votes[VoteType.APPROVE] / total_weight
        rejection_weight = weighted_votes[VoteType.REJECT] / total_weight
        
        # Determine consensus (requires supermajority to handle Byzantine faults)
        if approval_weight >= proposal.required_approval_ratio:
            proposal.consensus_reached = True
            proposal.consensus_result = VoteType.APPROVE
            proposal.confidence = approval_weight
        elif rejection_weight >= proposal.required_approval_ratio:
            proposal.consensus_reached = True
            proposal.consensus_result = VoteType.REJECT
            proposal.confidence = rejection_weight
        
        if proposal.consensus_reached:
            proposal.finalized_at = time.time()
            
            # Update Byzantine detector
            for vote in proposal.votes:
                was_minority = vote.vote_type != proposal.consensus_result
                self.byzantine_detector.record_vote(vote.voter_id, vote, was_minority)
            
            # Detect suspicious patterns
            suspicious = self.byzantine_detector.detect_suspicious_patterns(
                proposal.votes,
                proposal.consensus_result
            )
            
            logger.info(f"Consensus reached on {proposal.proposal_id}: "
                       f"{proposal.consensus_result} (confidence={proposal.confidence:.2f}, "
                       f"suspicious_nodes={len(suspicious)})")
    
    def get_proposal_status(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a proposal."""
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            return None
        
        return {
            'proposal_id': proposal.proposal_id,
            'consensus_type': proposal.consensus_type,
            'proposer_id': proposal.proposer_id,
            'votes_received': len(proposal.votes),
            'quorum_size': proposal.quorum_size,
            'consensus_reached': proposal.consensus_reached,
            'consensus_result': proposal.consensus_result,
            'confidence': proposal.confidence,
            'time_remaining': max(0, proposal.deadline - time.time()) if proposal.deadline else None
        }
    
    def get_suspicious_nodes(self) -> List[Dict[str, Any]]:
        """Get list of nodes with suspicious voting patterns."""
        suspicious = []
        for node_id in self.byzantine_detector.node_behavior.keys():
            score = self.byzantine_detector.get_suspicion_score(node_id)
            if score >= self.byzantine_detector.suspicion_threshold:
                suspicious.append({
                    'node_id': node_id,
                    'suspicion_score': score,
                    'total_votes': self.byzantine_detector.node_behavior[node_id]['total_votes'],
                    'minority_votes': self.byzantine_detector.node_behavior[node_id]['minority_votes']
                })
        
        return sorted(suspicious, key=lambda x: x['suspicion_score'], reverse=True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get consensus engine statistics."""
        total_proposals = len(self.proposals)
        finalized = sum(1 for p in self.proposals.values() if p.consensus_reached)
        
        suspicious_nodes = self.get_suspicious_nodes()
        
        return {
            'node_id': self.node_id,
            'total_proposals': total_proposals,
            'finalized_proposals': finalized,
            'pending_proposals': total_proposals - finalized,
            'suspicious_nodes_count': len(suspicious_nodes),
            'byzantine_tolerance': self.byzantine_tolerance,
            'trusted_nodes_count': len([t for t in self.node_trust_scores.values() if t > 0.7])
        }
