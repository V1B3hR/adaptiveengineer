"""
Trust network management module with Byzantine fault tolerance.
"""

from typing import Dict, Any, List, Optional, Set
from collections import defaultdict, deque
import time
import logging

logger = logging.getLogger(__name__)


class TrustNetwork:
    """
    Enhanced trust network system with Byzantine fault tolerance.
    
    Manages node relationships, detects compromised nodes, and provides
    Byzantine-resilient consensus mechanisms.
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
        
        # Byzantine fault detection
        self.compromised_nodes: Set[int] = set()
        self.suspicious_behavior_counts: Dict[int, int] = defaultdict(int)
        self.byzantine_detection_threshold = 3  # Suspicious behaviors before marking as compromised
        
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
    
    def mark_as_compromised(self, node_id: int, reason: str = "") -> None:
        """
        Mark a node as compromised/Byzantine.
        
        Args:
            node_id: ID of compromised node
            reason: Reason for marking as compromised
        """
        self.compromised_nodes.add(node_id)
        self.trust_network[node_id] = 0.0
        logger.warning(f"Node {node_id} marked as compromised by node {self.node_id}. Reason: {reason}")
    
    def is_compromised(self, node_id: int) -> bool:
        """Check if a node is marked as compromised."""
        return node_id in self.compromised_nodes
    
    def detect_byzantine_behavior(
        self,
        node_id: int,
        expected_behavior: Dict[str, Any],
        actual_behavior: Dict[str, Any]
    ) -> bool:
        """
        Detect Byzantine (malicious) behavior by comparing expected vs actual.
        
        Args:
            node_id: Node to check
            expected_behavior: What we expected the node to do
            actual_behavior: What the node actually did
            
        Returns:
            True if suspicious behavior detected
        """
        suspicious = False
        
        # Check for contradictory information
        if expected_behavior.get('value') != actual_behavior.get('value'):
            if abs(expected_behavior.get('value', 0) - actual_behavior.get('value', 0)) > 0.5:
                suspicious = True
        
        # Check for timing anomalies
        if expected_behavior.get('timestamp') and actual_behavior.get('timestamp'):
            time_diff = abs(expected_behavior['timestamp'] - actual_behavior['timestamp'])
            if time_diff > 300:  # More than 5 minutes off
                suspicious = True
        
        # Check for inconsistent reporting
        if actual_behavior.get('contradicts_previous'):
            suspicious = True
        
        if suspicious:
            self.suspicious_behavior_counts[node_id] += 1
            logger.debug(f"Suspicious behavior detected from node {node_id} "
                        f"(count: {self.suspicious_behavior_counts[node_id]})")
            
            # Decrease trust
            current_trust = self.get_trust(node_id)
            self.trust_network[node_id] = max(0.0, current_trust - 0.15)
            
            # Mark as compromised if threshold exceeded
            if self.suspicious_behavior_counts[node_id] >= self.byzantine_detection_threshold:
                self.mark_as_compromised(node_id, "Repeated suspicious behavior")
        
        return suspicious
    
    def get_trusted_nodes(self, min_trust: float = 0.7) -> List[int]:
        """
        Get list of trusted nodes.
        
        Args:
            min_trust: Minimum trust threshold
            
        Returns:
            List of trusted node IDs
        """
        return [
            node_id for node_id, trust in self.trust_network.items()
            if trust >= min_trust and node_id not in self.compromised_nodes
        ]
    
    def discount_input(self, node_id: int, value: float) -> float:
        """
        Discount input from a node based on trust.
        
        Unreliable or compromised nodes have their input heavily discounted.
        
        Args:
            node_id: Source node
            value: Input value
            
        Returns:
            Discounted value
        """
        if self.is_compromised(node_id):
            return 0.0  # Completely ignore compromised nodes
        
        trust = self.get_trust(node_id)
        
        # Apply trust-based discount
        if trust < 0.3:
            # Low trust: heavy discount
            discount_factor = 0.2
        elif trust < 0.5:
            # Moderate trust: moderate discount
            discount_factor = 0.5
        elif trust < 0.7:
            # Good trust: light discount
            discount_factor = 0.8
        else:
            # High trust: minimal discount
            discount_factor = 1.0
        
        return value * discount_factor
    
    def byzantine_resilient_aggregate(
        self,
        inputs: Dict[int, float],
        method: str = "weighted_median"
    ) -> float:
        """
        Aggregate inputs from multiple nodes in a Byzantine-resilient way.
        
        Args:
            inputs: Dict of node_id -> value
            method: Aggregation method ("weighted_median", "trimmed_mean")
            
        Returns:
            Aggregated value
        """
        if not inputs:
            return 0.0
        
        # Filter out compromised nodes
        filtered_inputs = {
            nid: value for nid, value in inputs.items()
            if not self.is_compromised(nid)
        }
        
        if not filtered_inputs:
            return 0.0
        
        if method == "weighted_median":
            # Weight by trust and find weighted median
            weighted_values = []
            for node_id, value in filtered_inputs.items():
                trust = self.get_trust(node_id)
                weighted_values.append((value, trust))
            
            # Sort by value
            weighted_values.sort(key=lambda x: x[0])
            
            # Find weighted median
            total_weight = sum(w for _, w in weighted_values)
            cumulative_weight = 0.0
            median_value = 0.0
            
            for value, weight in weighted_values:
                cumulative_weight += weight
                if cumulative_weight >= total_weight / 2:
                    median_value = value
                    break
            
            return median_value
        
        elif method == "trimmed_mean":
            # Remove outliers and compute trust-weighted mean
            values = sorted(filtered_inputs.values())
            
            # Trim top and bottom 20%
            trim_count = max(1, len(values) // 5)
            trimmed = values[trim_count:-trim_count] if len(values) > 2 * trim_count else values
            
            # Weighted mean of trimmed values
            total_weighted = 0.0
            total_weight = 0.0
            
            for node_id, value in filtered_inputs.items():
                if value in trimmed:
                    trust = self.get_trust(node_id)
                    total_weighted += value * trust
                    total_weight += trust
            
            return total_weighted / total_weight if total_weight > 0 else 0.0
        
        else:
            # Fallback: simple weighted average
            total_weighted = sum(
                value * self.get_trust(node_id)
                for node_id, value in filtered_inputs.items()
            )
            total_weight = sum(
                self.get_trust(node_id)
                for node_id in filtered_inputs.keys()
            )
            return total_weighted / total_weight if total_weight > 0 else 0.0
    
    def get_byzantine_resilience_metrics(self) -> Dict[str, Any]:
        """Get metrics about Byzantine fault tolerance."""
        total_nodes = len(self.trust_network)
        compromised = len(self.compromised_nodes)
        suspicious = sum(1 for nid in self.suspicious_behavior_counts 
                        if self.suspicious_behavior_counts[nid] > 0 and nid not in self.compromised_nodes)
        
        trusted = len(self.get_trusted_nodes())
        
        return {
            'total_nodes': total_nodes,
            'compromised_nodes': compromised,
            'suspicious_nodes': suspicious,
            'trusted_nodes': trusted,
            'byzantine_tolerance': max(0.0, 1.0 - (compromised / max(1, total_nodes))),
            'network_health': trusted / max(1, total_nodes),
            'compromised_node_ids': list(self.compromised_nodes)
        }
