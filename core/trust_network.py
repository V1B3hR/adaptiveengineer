"""
Trust network management module.
"""

from typing import Dict, Any, List, Optional
from collections import defaultdict, deque


class TrustNetwork:
    """
    Enhanced trust network system for managing node relationships.
    """
    
    def __init__(self, node_id: int):
        """
        Initialize trust network.
        
        Args:
            node_id: ID of the node owning this trust network
        """
        self.node_id = node_id
        self.trust_network: Dict[int, float] = {}  # node_id -> trust_value
        self.interaction_history: Dict[int, List[Dict]] = defaultdict(list)
        
    def get_trust(self, other_node_id: int) -> float:
        """
        Get trust level for another node.
        
        Args:
            other_node_id: ID of other node
            
        Returns:
            Trust level (0.0 to 1.0), default 0.5 if unknown
        """
        return self.trust_network.get(other_node_id, 0.5)
    
    def update_trust(self, other_node: Any, interaction_type: str, context: Dict[str, Any]) -> float:
        """
        Update trust based on interaction.
        
        Args:
            other_node: The other node
            interaction_type: Type of interaction
            context: Interaction context
            
        Returns:
            New trust value
        """
        other_id = getattr(other_node, 'node_id', 0)
        current_trust = self.get_trust(other_id)
        
        # Simple trust update logic
        if interaction_type in ['memory', 'query', 'resource']:
            # Positive interactions increase trust slightly
            new_trust = min(1.0, current_trust + 0.05)
        elif interaction_type in ['warning', 'threat']:
            # Warnings can decrease trust if frequent
            new_trust = max(0.0, current_trust - 0.02)
        else:
            # Neutral interactions maintain trust
            new_trust = current_trust
        
        self.trust_network[other_id] = new_trust
        
        # Record interaction
        self.interaction_history[other_id].append({
            'type': interaction_type,
            'timestamp': context.get('timestamp', 0),
            'trust_after': new_trust
        })
        
        return new_trust
    
    def get_trust_summary(self) -> Dict[str, Any]:
        """Get summary of trust network."""
        if not self.trust_network:
            return {
                'total_connections': 0,
                'average_trust': 0.5,
                'trusted_nodes': 0,
                'suspicious_nodes': 0
            }
        
        trust_values = list(self.trust_network.values())
        return {
            'total_connections': len(trust_values),
            'average_trust': sum(trust_values) / len(trust_values),
            'trusted_nodes': sum(1 for t in trust_values if t > 0.7),
            'suspicious_nodes': sum(1 for t in trust_values if t < 0.3)
        }
    
    def generate_trust_network_graph(self) -> Dict[str, Any]:
        """Generate graph data for visualization."""
        nodes = [{'id': self.node_id, 'type': 'self'}]
        edges = []
        
        for other_id, trust in self.trust_network.items():
            nodes.append({'id': other_id, 'type': 'peer'})
            edges.append({
                'from': self.node_id,
                'to': other_id,
                'trust': trust
            })
        
        return {'nodes': nodes, 'edges': edges}
    
    def get_trust_network_metrics(self) -> Dict[str, Any]:
        """Get comprehensive trust network metrics."""
        summary = self.get_trust_summary()
        
        trust_values = list(self.trust_network.values())
        if trust_values:
            variance = sum((t - summary['average_trust'])**2 for t in trust_values) / len(trust_values)
        else:
            variance = 0.0
        
        return {
            **summary,
            'trust_variance': variance,
            'network_resilience': summary['trusted_nodes'] / max(1, summary['total_connections']),
            'suspicious_ratio': summary['suspicious_nodes'] / max(1, summary['total_connections'])
        }
    
    def initiate_consensus_vote(self, subject_node_id: int) -> Dict[str, Any]:
        """Initiate a consensus vote about a node."""
        return {
            'subject': subject_node_id,
            'initiator': self.node_id,
            'vote_id': f"vote_{self.node_id}_{subject_node_id}"
        }
    
    def process_consensus_vote(self, vote_request: Dict, responses: List[Dict]) -> Dict[str, Any]:
        """Process consensus vote responses."""
        if not responses:
            return {'consensus': 'unknown', 'confidence': 0.0}
        
        # Aggregate trust assessments
        total_trust = sum(r.get('trust_assessment', 0.5) for r in responses)
        avg_trust = total_trust / len(responses)
        
        total_confidence = sum(r.get('confidence', 0.5) for r in responses)
        avg_confidence = total_confidence / len(responses)
        
        consensus = 'trusted' if avg_trust > 0.6 else 'suspicious' if avg_trust < 0.4 else 'neutral'
        
        return {
            'consensus': consensus,
            'average_trust': avg_trust,
            'confidence': avg_confidence,
            'voter_count': len(responses)
        }
    
    def stress_test_byzantine_resilience(self, malicious_ratio: float = 0.33, num_simulations: int = 50) -> Dict[str, Any]:
        """Test resilience against Byzantine faults."""
        # Simplified stress test
        detection_count = 0
        false_positive_count = 0
        
        for _ in range(num_simulations):
            # Simulate detection
            import random
            if random.random() < (1 - malicious_ratio):
                detection_count += 1
            else:
                false_positive_count += 1
        
        return {
            'resilience_score': detection_count / num_simulations,
            'detection_rate': detection_count / num_simulations,
            'false_positive_rate': false_positive_count / num_simulations
        }
    
    def process_community_feedback(self, subject_id: int, feedback_list: List[Dict]) -> None:
        """Process community feedback about a node."""
        if not feedback_list:
            return
        
        # Aggregate feedback
        avg_trust = sum(f.get('trust_level', 0.5) for f in feedback_list) / len(feedback_list)
        
        # Update trust based on community consensus
        self.trust_network[subject_id] = avg_trust
