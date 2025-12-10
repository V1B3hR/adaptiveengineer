"""
Tests for Byzantine-resilient consensus mechanism.
"""

import pytest
import time
from core.consensus import (
    ConsensusEngine,
    ConsensusProposal,
    ConsensusType,
    Vote,
    VoteType,
    ByzantineDetector
)


class TestVote:
    """Test Vote data structure."""
    
    def test_vote_creation(self):
        """Test creating a vote."""
        vote = Vote(
            voter_id=1,
            vote_type=VoteType.APPROVE,
            confidence=0.9,
            evidence={"reason": "test"}
        )
        
        assert vote.voter_id == 1
        assert vote.vote_type == VoteType.APPROVE
        assert vote.confidence == 0.9
    
    def test_vote_signature(self):
        """Test vote signature computation."""
        vote = Vote(
            voter_id=1,
            vote_type=VoteType.APPROVE,
            confidence=0.9
        )
        
        signature = vote.compute_signature("secret")
        assert signature is not None
        assert len(signature) == 64  # SHA256 produces 64 hex chars


class TestConsensusProposal:
    """Test ConsensusProposal data structure."""
    
    def test_proposal_creation(self):
        """Test creating a consensus proposal."""
        proposal = ConsensusProposal(
            proposal_id="test_prop_1",
            consensus_type=ConsensusType.ROOT_CAUSE,
            proposer_id=1,
            subject={"incident_id": "inc_123"}
        )
        
        assert proposal.proposal_id == "test_prop_1"
        assert proposal.consensus_type == ConsensusType.ROOT_CAUSE
        assert not proposal.consensus_reached


class TestByzantineDetector:
    """Test Byzantine node detection."""
    
    def test_detector_creation(self):
        """Test creating Byzantine detector."""
        detector = ByzantineDetector(suspicion_threshold=0.3)
        assert detector.suspicion_threshold == 0.3
    
    def test_record_vote(self):
        """Test recording votes for behavioral analysis."""
        detector = ByzantineDetector()
        vote = Vote(voter_id=1, vote_type=VoteType.APPROVE, confidence=0.9)
        
        detector.record_vote(1, vote, was_minority=False)
        
        behavior = detector.node_behavior[1]
        assert behavior['total_votes'] == 1
        assert behavior['minority_votes'] == 0
    
    def test_detect_suspicious_pattern(self):
        """Test detection of suspicious voting patterns."""
        detector = ByzantineDetector(suspicion_threshold=0.3)
        
        # Node consistently votes in minority
        for i in range(10):
            vote = Vote(voter_id=1, vote_type=VoteType.APPROVE, confidence=0.5)
            detector.record_vote(1, vote, was_minority=True)
        
        behavior = detector.node_behavior[1]
        assert behavior['total_votes'] == 10
        assert behavior['minority_votes'] == 10


class TestConsensusEngine:
    """Test consensus engine basic functionality."""
    
    def test_engine_creation(self):
        """Test creating consensus engine."""
        engine = ConsensusEngine(node_id=1)
        assert engine.node_id == 1
    
    def test_engine_configuration(self):
        """Test consensus engine configuration."""
        engine = ConsensusEngine(
            node_id=1,
            byzantine_tolerance=0.33,
            default_quorum_ratio=0.67
        )
        assert engine.node_id == 1
        assert engine.byzantine_tolerance == 0.33


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
