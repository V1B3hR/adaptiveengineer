# Phase 3 Implementation: Learning, Evolution, Trust, and Consensus

This document describes the implementation of Phase 3 requirements from the roadmap.

## Overview

Phase 3 extends the system with advanced adaptive capabilities and Byzantine-resilient coordination:

1. **Adaptive Learning & Evolution** - Genetic algorithms for strategy optimization and threshold auto-tuning
2. **Trust Network & Byzantine-Resilient Consensus** - Fault-tolerant decision making in adversarial environments

## 1. Adaptive Learning & Evolution

### Evolutionary Learning Architecture

The evolutionary learning system uses genetic algorithms to optimize detection, mitigation, and recovery strategies through reproduction, variation, and selection.

#### Core Components

**`Strategy` (core/evolution_engine.py)**
- Represents an evolvable strategy (genome)
- Contains parameters that can be optimized
- Types: DETECTION, MITIGATION, RECOVERY, THRESHOLD
- Supports mutation and crossover operations
- Tracks fitness, generation, and parentage

**`EvolutionEngine` (core/evolution_engine.py)**
- Manages populations of strategies
- Implements genetic algorithm operations:
  - Selection (tournament selection)
  - Crossover (single-point)
  - Mutation (Gaussian noise)
  - Elitism (preserve best strategies)
- Evaluates fitness and drives evolution
- Supports multiple strategy types simultaneously

### Key Features

✅ **Strategy Evolution**: Genetic algorithms improve strategies over generations

```python
from core.evolution_engine import EvolutionEngine, StrategyType

# Initialize evolution engine
evolution = EvolutionEngine(
    population_size=20,
    mutation_rate=0.1,
    crossover_rate=0.7,
    elitism_count=2
)

# Define parameter ranges for detection strategies
detection_params = {
    'threshold': (0.5, 0.95),
    'sensitivity': (0.3, 1.0),
    'response_time': (0.1, 0.9)
}

# Initialize population
evolution.initialize_population(StrategyType.DETECTION, detection_params)

# Define fitness function
def evaluate_strategy(strategy):
    # Higher detection rate, lower false positives, faster response
    fitness = (strategy.parameters['sensitivity'] * 
               (1 - strategy.parameters['threshold']) +
               (1 - strategy.parameters['response_time']))
    return max(0.0, min(1.0, fitness))

# Evolve strategies
for generation in range(10):
    stats = evolution.evolve_generation(StrategyType.DETECTION, evaluate_strategy)
    print(f"Gen {stats['generation']}: best={stats['best_fitness']:.4f}")

# Get best evolved strategy
best = evolution.get_best_strategy(StrategyType.DETECTION)
print(f"Best strategy: threshold={best.parameters['threshold']:.3f}")
```

✅ **Reproduction and Variation**: Strategies reproduce through crossover and mutation

```python
# Crossover creates offspring from two parents
parent1 = Strategy(strategy_id="s1", strategy_type=StrategyType.DETECTION,
                  parameters={'threshold': 0.8, 'sensitivity': 0.9})
parent2 = Strategy(strategy_id="s2", strategy_type=StrategyType.DETECTION,
                  parameters={'threshold': 0.7, 'sensitivity': 0.8})

child1, child2 = Strategy.crossover(parent1, parent2)

# Mutation introduces variation
mutated = parent1.mutate(mutation_rate=0.1, mutation_strength=0.2)
```

✅ **Selection (Survival of the Fittest)**: Tournament selection favors better strategies

✅ **Multi-Population Support**: Different strategy types evolve independently

### Adaptive Learning System

The adaptive learning system learns "normal" behavior and automatically tunes thresholds.

#### Core Components

**`BehaviorProfile` (core/adaptive_learning.py)**
- Statistical profile of normal behavior
- Maintains running statistics (mean, variance, min, max)
- Adaptive threshold calculation
- Anomaly detection using z-scores
- Incremental learning (Welford's algorithm)

**`AdaptiveLearningSystem` (core/adaptive_learning.py)**
- Manages multiple behavior profiles
- Continual learning from observations
- Automatic threshold tuning
- Anomaly detection
- Learning data export/import for persistence

### Key Features

✅ **Learning Normal Behavior**: Observes and learns patterns

```python
from core.adaptive_learning import AdaptiveLearningSystem, BehaviorType

# Initialize learning system
learning = AdaptiveLearningSystem(
    learning_rate=0.01,
    auto_tune_interval=300.0  # Auto-tune every 5 minutes
)

# Observe normal service behavior
for i in range(100):
    cpu_usage = 0.4 + random.gauss(0, 0.05)  # Normal: ~40% CPU
    result = learning.observe(
        BehaviorType.SERVICE,
        cpu_usage,
        metadata={'metric': 'cpu_usage'}
    )
    
    if result['is_anomaly']:
        print(f"Anomaly detected: {cpu_usage:.2f}")

# Get learned profile
profile = learning.get_profile(BehaviorType.SERVICE)
print(f"Learned mean: {profile.mean:.3f}")
print(f"Learned std dev: {(profile.variance ** 0.5):.3f}")
```

✅ **Auto-Tuning Thresholds**: Automatically adjusts based on observed behavior

```python
# Force immediate threshold tuning
tune_results = learning.force_auto_tune(BehaviorType.SERVICE)

thresholds = tune_results[BehaviorType.SERVICE]
print(f"Lower threshold: {thresholds['lower_threshold']:.3f}")
print(f"Upper threshold: {thresholds['upper_threshold']:.3f}")
print(f"Mean: {thresholds['mean']:.3f}")
```

✅ **Anomaly Detection**: Statistical detection using learned profiles

✅ **Continual Adaptation**: Incrementally updates with each observation

✅ **Persistence**: Export and import learned parameters

```python
# Export learned parameters
params = learning.export_learned_parameters()

# Later, import to restore learned state
learning2 = AdaptiveLearningSystem()
learning2.import_learned_parameters(params)
```

## 2. Trust Network & Byzantine-Resilient Consensus

### Enhanced Trust Network

The trust network now includes Byzantine fault detection and resilient aggregation.

#### Enhanced Features in `TrustNetwork`

**Byzantine Fault Detection:**
- Detects malicious or compromised nodes
- Tracks suspicious behavior patterns
- Automatically marks nodes as compromised
- Maintains compromised node registry

**Input Discounting:**
- Discounts input from unreliable nodes based on trust
- Completely ignores compromised nodes
- Trust-weighted value adjustment

**Byzantine-Resilient Aggregation:**
- Weighted median aggregation
- Trimmed mean with outlier removal
- Filters out compromised node inputs

### Key Features

✅ **Byzantine Behavior Detection**: Identifies malicious nodes

```python
from core.trust_network import TrustNetwork

trust_net = TrustNetwork(node_id=1)

# Detect Byzantine behavior
is_suspicious = trust_net.detect_byzantine_behavior(
    node_id=5,
    expected_behavior={'value': 0.5, 'timestamp': time.time()},
    actual_behavior={'value': 0.9, 'contradicts_previous': True}
)

if trust_net.is_compromised(5):
    print("Node 5 is compromised!")
```

✅ **Input Discounting**: Trust-based value weighting

```python
# Discount input based on trust
original_value = 0.8
discounted = trust_net.discount_input(node_id=5, value=original_value)

# Compromised nodes return 0.0
# Low trust nodes heavily discounted
# High trust nodes minimally discounted
```

✅ **Byzantine-Resilient Aggregation**: Tolerates malicious inputs

```python
# Aggregate values from multiple nodes
inputs = {
    2: 0.7,  # Trusted node
    3: 0.65, # Trusted node
    5: 0.2,  # Compromised node (will be ignored)
    6: 0.4   # Neutral node
}

# Weighted median (Byzantine-resilient)
aggregated = trust_net.byzantine_resilient_aggregate(
    inputs,
    method="weighted_median"
)

print(f"Aggregated value: {aggregated:.3f}")
# Compromised node's extreme value is ignored
```

✅ **Network Health Metrics**: Monitor Byzantine resilience

```python
metrics = trust_net.get_byzantine_resilience_metrics()
print(f"Compromised nodes: {metrics['compromised_nodes']}")
print(f"Byzantine tolerance: {metrics['byzantine_tolerance']:.2f}")
print(f"Network health: {metrics['network_health']:.2f}")
```

### Consensus System

The consensus system implements Byzantine-resilient voting for distributed decision making.

#### Core Components

**`Vote` (core/consensus.py)**
- Individual vote with type (APPROVE, REJECT, ABSTAIN)
- Confidence level (0.0 to 1.0)
- Evidence and signature support
- Timestamp for vote ordering

**`ConsensusProposal` (core/consensus.py)**
- Proposal requiring consensus
- Types: ROOT_CAUSE, ATTACK_VALIDATION, COLLECTIVE_RESPONSE, TRUST_ASSESSMENT, THRESHOLD_UPDATE
- Vote collection and quorum management
- Byzantine-resilient result determination

**`ByzantineDetector` (core/consensus.py)**
- Detects suspicious voting patterns
- Tracks node voting behavior
- Identifies nodes consistently voting against consensus
- Calculates suspicion scores

**`ConsensusEngine` (core/consensus.py)**
- Manages consensus processes
- Trust-weighted vote aggregation
- Byzantine fault tolerance (tolerates up to 33% malicious nodes)
- Quorum-based decision making
- Supermajority requirements (67% approval)

### Key Features

✅ **Byzantine-Resilient Consensus**: Tolerates malicious nodes

```python
from core.consensus import ConsensusEngine, ConsensusType, VoteType

# Initialize consensus engine
engine = ConsensusEngine(
    node_id=1,
    byzantine_tolerance=0.33,  # Tolerate up to 33% malicious
    default_quorum_ratio=0.67   # Require 67% for quorum
)

# Set trust scores for nodes
engine.set_node_trust(node_id=2, trust_score=0.9)
engine.set_node_trust(node_id=3, trust_score=0.8)
engine.set_node_trust(node_id=5, trust_score=0.1)  # Low trust

# Initiate consensus on incident root cause
proposal_id = engine.initiate_consensus(
    ConsensusType.ROOT_CAUSE,
    subject={
        'incident_id': 'inc_001',
        'suspected_cause': 'memory_leak',
        'severity': 0.8
    },
    network_size=7,
    timeout=300.0
)
```

✅ **Trust-Weighted Voting**: Votes weighted by node trust

```python
# Cast votes from different nodes
engine.cast_vote(proposal_id, voter_id=2, vote_type=VoteType.APPROVE, 
                confidence=0.85)
engine.cast_vote(proposal_id, voter_id=3, vote_type=VoteType.APPROVE, 
                confidence=0.90)
engine.cast_vote(proposal_id, voter_id=5, vote_type=VoteType.REJECT, 
                confidence=0.95)  # Byzantine node

# Check consensus (Byzantine vote discounted)
status = engine.get_proposal_status(proposal_id)
if status['consensus_reached']:
    print(f"Consensus: {status['consensus_result']}")
    print(f"Confidence: {status['confidence']:.2f}")
```

✅ **Suspicious Node Detection**: Identifies malicious voting patterns

```python
# Get nodes with suspicious voting behavior
suspicious = engine.get_suspicious_nodes()
for node_info in suspicious:
    print(f"Node {node_info['node_id']}: "
          f"suspicion={node_info['suspicion_score']:.2f}")
```

✅ **Multiple Consensus Types**: Support for various decisions

- **ROOT_CAUSE**: Determine incident root cause
- **ATTACK_VALIDATION**: Validate detected attacks
- **COLLECTIVE_RESPONSE**: Coordinate response actions
- **TRUST_ASSESSMENT**: Assess node trustworthiness
- **THRESHOLD_UPDATE**: Update system thresholds collectively

✅ **Quorum Management**: Requires sufficient participation

✅ **Supermajority Requirements**: 67% approval for Byzantine tolerance

## Architecture Benefits

### Evolutionary Learning

- ✅ Strategies continuously improve through natural selection
- ✅ Automatic adaptation to changing environments
- ✅ No manual parameter tuning required
- ✅ Multi-objective optimization (detection vs. false positives)
- ✅ Preserves best strategies through elitism

### Adaptive Learning

- ✅ Learns normal behavior patterns automatically
- ✅ Auto-tunes thresholds based on observations
- ✅ Detects anomalies using statistical methods
- ✅ Adapts to system drift over time
- ✅ No predefined thresholds needed

### Byzantine-Resilient Trust & Consensus

- ✅ Tolerates up to 33% malicious nodes
- ✅ Detects and marks compromised nodes
- ✅ Trust-weighted decision making
- ✅ Resilient aggregation methods
- ✅ Collective intelligence for better decisions
- ✅ Distributed consensus without central authority

## Integration Example

Phase 3 capabilities integrate seamlessly with Phase 1 & 2:

```python
from adaptiveengineer import AliveLoopNode
from core.plugin_manager import PluginManager
from plugins.security import SecurityPlugin

from core.evolution_engine import EvolutionEngine, StrategyType
from core.adaptive_learning import AdaptiveLearningSystem, BehaviorType
from core.consensus import ConsensusEngine, ConsensusType, VoteType

# 1. Create node with plugins
node = AliveLoopNode(position=(0, 0), velocity=(0, 0), 
                     initial_energy=10.0, node_id=1)
pm = PluginManager()
pm.register_plugin(SecurityPlugin())
pm.initialize_all(node)

# 2. Evolve detection strategies
evolution = EvolutionEngine()
evolution.initialize_population(StrategyType.DETECTION, detection_params)

for gen in range(10):
    evolution.evolve_generation(StrategyType.DETECTION, fitness_function)

best_strategy = evolution.get_best_strategy(StrategyType.DETECTION)

# 3. Learn normal behavior and auto-tune
learning = AdaptiveLearningSystem()

for metric_value in system_metrics:
    result = learning.observe(BehaviorType.SERVICE, metric_value)
    if result['is_anomaly']:
        # Trigger investigation
        pass

# Auto-tuned thresholds ready to use
profile = learning.get_profile(BehaviorType.SERVICE)

# 4. Consensus on collective decisions
consensus = ConsensusEngine(node_id=1)

# Propose threshold update based on learned values
proposal_id = consensus.initiate_consensus(
    ConsensusType.THRESHOLD_UPDATE,
    subject={'new_threshold': profile.upper_threshold},
    network_size=7
)

# Nodes vote, Byzantine nodes discounted
# Collective decision reached
```

## Testing

Run the Phase 3 demonstration:

```bash
python3 example_phase3.py
```

The demonstration shows:

1. **Evolutionary Learning**: Genetic algorithms evolving detection strategies over 10 generations
2. **Adaptive Learning**: Learning normal CPU behavior and detecting anomalies
3. **Trust Network**: Detecting Byzantine nodes and discounting their input
4. **Byzantine-Resilient Consensus**: Reaching consensus on incident root cause despite malicious votes
5. **Integrated Operation**: Evolution, learning, and consensus working together

## Performance Characteristics

### Evolution Engine

- Population size: Configurable (default 20)
- Convergence: Typically 10-50 generations
- Mutation rate: 0.1 (10% of parameters mutated)
- Crossover rate: 0.7 (70% crossover vs. mutation)
- Time complexity: O(n * m) where n=population size, m=generations

### Adaptive Learning

- Memory: O(w) where w=window size (default 1000 observations)
- Update: O(1) incremental statistics
- Anomaly detection: O(1) z-score calculation
- Auto-tuning: Periodic (default every 5 minutes)

### Consensus

- Byzantine tolerance: Up to 33% malicious nodes (3f+1 model)
- Quorum requirement: 67% of nodes (default)
- Vote processing: O(n) where n=number of votes
- Consensus determination: O(n) aggregation

## Future Enhancements

The Phase 3 architecture supports future expansion:

- **Multi-Objective Evolution**: Pareto optimization for competing objectives
- **Deep Learning**: Neural networks for fitness evaluation
- **Advanced Anomaly Detection**: Machine learning models in learning system
- **PBFT Implementation**: Full Practical Byzantine Fault Tolerance
- **Distributed Evolution**: Population distributed across nodes
- **Transfer Learning**: Share learned parameters between nodes
- **Adaptive Byzantine Tolerance**: Dynamically adjust based on threat level

## Summary

Phase 3 establishes:

✅ **Adaptive Learning & Evolution (AL Principle #2 & #3)**
- Genetic algorithms optimize detection, mitigation, and recovery strategies
- Strategies evolve through reproduction, variation, and selection
- System learns normal behavior and auto-tunes thresholds
- Continual adaptation to changing environments
- Survival of the fittest drives improvement

✅ **Trust Network & Byzantine-Resilient Consensus**
- Detects and marks compromised/malicious nodes
- Discounts input from unreliable nodes
- Byzantine-resilient aggregation (weighted median, trimmed mean)
- Consensus for root cause, attack validation, collective response
- Decision voting tolerates up to 33% Byzantine faults
- Trust-weighted voting in adversarial environments
- Distributed decision making without central authority

The implementation provides adaptive, self-improving systems that can learn from experience, evolve better strategies, and make collective decisions even in the presence of malicious actors. This establishes the foundation for truly autonomous, resilient, and intelligent adaptive engineer systems.
