"""
Evolution engine for adaptive learning using genetic algorithms.

This module implements evolutionary mechanisms to improve detection,
mitigation, and recovery strategies through reproduction, variation,
and selection (survival of the fittest).
"""

import random
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class StrategyType(str, Enum):
    """Types of strategies that can evolve."""
    DETECTION = "detection"
    MITIGATION = "mitigation"
    RECOVERY = "recovery"
    THRESHOLD = "threshold"


@dataclass
class Strategy:
    """
    Represents an evolvable strategy (genome).
    
    Strategies encode parameters for detection, mitigation, or recovery
    that can be evolved through genetic algorithms.
    """
    strategy_id: str
    strategy_type: StrategyType
    parameters: Dict[str, float]  # Parameter name -> value
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    evaluations: int = 0
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.2) -> 'Strategy':
        """
        Create a mutated copy of this strategy.
        
        Args:
            mutation_rate: Probability of mutating each parameter
            mutation_strength: Magnitude of mutations (as fraction of value)
            
        Returns:
            Mutated strategy
        """
        new_params = {}
        for key, value in self.parameters.items():
            if random.random() < mutation_rate:
                # Apply Gaussian noise
                noise = random.gauss(0, mutation_strength * abs(value) if value != 0 else mutation_strength)
                new_params[key] = max(0.0, min(1.0, value + noise))
            else:
                new_params[key] = value
        
        return Strategy(
            strategy_id=f"{self.strategy_id}_mut_{random.randint(1000, 9999)}",
            strategy_type=self.strategy_type,
            parameters=new_params,
            generation=self.generation + 1,
            parent_ids=[self.strategy_id]
        )
    
    @staticmethod
    def crossover(parent1: 'Strategy', parent2: 'Strategy') -> Tuple['Strategy', 'Strategy']:
        """
        Create two offspring strategies through crossover.
        
        Args:
            parent1: First parent strategy
            parent2: Second parent strategy
            
        Returns:
            Two offspring strategies
        """
        if parent1.strategy_type != parent2.strategy_type:
            raise ValueError("Cannot crossover strategies of different types")
        
        # Single-point crossover
        keys = list(parent1.parameters.keys())
        crossover_point = random.randint(0, len(keys))
        
        child1_params = {}
        child2_params = {}
        
        for i, key in enumerate(keys):
            if i < crossover_point:
                child1_params[key] = parent1.parameters[key]
                child2_params[key] = parent2.parameters[key]
            else:
                child1_params[key] = parent2.parameters[key]
                child2_params[key] = parent1.parameters[key]
        
        child1 = Strategy(
            strategy_id=f"cross_{random.randint(1000, 9999)}",
            strategy_type=parent1.strategy_type,
            parameters=child1_params,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.strategy_id, parent2.strategy_id]
        )
        
        child2 = Strategy(
            strategy_id=f"cross_{random.randint(1000, 9999)}",
            strategy_type=parent1.strategy_type,
            parameters=child2_params,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.strategy_id, parent2.strategy_id]
        )
        
        return child1, child2


class EvolutionEngine:
    """
    Manages evolutionary optimization of strategies using genetic algorithms.
    
    Implements:
    - Population management
    - Fitness evaluation
    - Selection (tournament, roulette wheel)
    - Crossover and mutation
    - Elitism
    """
    
    def __init__(
        self,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.2,
        crossover_rate: float = 0.7,
        elitism_count: int = 2,
        tournament_size: int = 3
    ):
        """
        Initialize evolution engine.
        
        Args:
            population_size: Number of strategies in population
            mutation_rate: Probability of mutating each parameter
            mutation_strength: Magnitude of mutations
            crossover_rate: Probability of crossover vs. mutation
            elitism_count: Number of best strategies to preserve
            tournament_size: Number of candidates in tournament selection
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size
        
        self.populations: Dict[StrategyType, List[Strategy]] = {}
        self.generation = 0
        self.best_strategies: Dict[StrategyType, Strategy] = {}
        
        logger.info(f"Evolution engine initialized (pop={population_size}, "
                   f"mut_rate={mutation_rate}, cross_rate={crossover_rate})")
    
    def initialize_population(
        self,
        strategy_type: StrategyType,
        parameter_ranges: Dict[str, Tuple[float, float]]
    ) -> None:
        """
        Initialize a random population for a strategy type.
        
        Args:
            strategy_type: Type of strategy
            parameter_ranges: Dict of parameter_name -> (min, max)
        """
        population = []
        for i in range(self.population_size):
            params = {}
            for param_name, (min_val, max_val) in parameter_ranges.items():
                params[param_name] = random.uniform(min_val, max_val)
            
            strategy = Strategy(
                strategy_id=f"{strategy_type}_gen0_{i}",
                strategy_type=strategy_type,
                parameters=params,
                generation=0
            )
            population.append(strategy)
        
        self.populations[strategy_type] = population
        logger.info(f"Initialized population for {strategy_type} with {len(population)} strategies")
    
    def evaluate_fitness(
        self,
        strategy: Strategy,
        evaluation_function: Any
    ) -> float:
        """
        Evaluate fitness of a strategy.
        
        Args:
            strategy: Strategy to evaluate
            evaluation_function: Function that takes strategy and returns fitness score
            
        Returns:
            Fitness score
        """
        fitness = evaluation_function(strategy)
        strategy.fitness = fitness
        strategy.evaluations += 1
        return fitness
    
    def tournament_selection(self, population: List[Strategy]) -> Strategy:
        """
        Select a strategy using tournament selection.
        
        Args:
            population: Population to select from
            
        Returns:
            Selected strategy
        """
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return max(tournament, key=lambda s: s.fitness)
    
    def evolve_generation(
        self,
        strategy_type: StrategyType,
        evaluation_function: Any
    ) -> Dict[str, Any]:
        """
        Evolve one generation for a strategy type.
        
        Args:
            strategy_type: Type of strategy to evolve
            evaluation_function: Function to evaluate fitness
            
        Returns:
            Evolution statistics
        """
        if strategy_type not in self.populations:
            raise ValueError(f"Population not initialized for {strategy_type}")
        
        population = self.populations[strategy_type]
        
        # Evaluate fitness for all strategies
        for strategy in population:
            if strategy.evaluations == 0 or strategy.generation == self.generation:
                self.evaluate_fitness(strategy, evaluation_function)
        
        # Sort by fitness
        population.sort(key=lambda s: s.fitness, reverse=True)
        
        # Track best strategy
        best = population[0]
        self.best_strategies[strategy_type] = best
        
        # Create new generation
        new_population = []
        
        # Elitism: preserve best strategies
        for i in range(min(self.elitism_count, len(population))):
            new_population.append(population[i])
        
        # Generate offspring
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate and len(population) >= 2:
                # Crossover
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                child1, child2 = Strategy.crossover(parent1, parent2)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            else:
                # Mutation only
                parent = self.tournament_selection(population)
                child = parent.mutate(self.mutation_rate, self.mutation_strength)
                new_population.append(child)
        
        self.populations[strategy_type] = new_population[:self.population_size]
        self.generation += 1
        
        # Calculate statistics
        fitness_values = [s.fitness for s in population]
        stats = {
            'generation': self.generation,
            'best_fitness': best.fitness,
            'average_fitness': sum(fitness_values) / len(fitness_values),
            'worst_fitness': min(fitness_values),
            'best_strategy_id': best.strategy_id,
            'population_size': len(new_population)
        }
        
        logger.debug(f"Evolution gen {self.generation} for {strategy_type}: "
                    f"best={stats['best_fitness']:.4f}, avg={stats['average_fitness']:.4f}")
        
        return stats
    
    def get_best_strategy(self, strategy_type: StrategyType) -> Optional[Strategy]:
        """Get the best strategy for a type."""
        return self.best_strategies.get(strategy_type)
    
    def get_population_stats(self, strategy_type: StrategyType) -> Dict[str, Any]:
        """Get statistics for a population."""
        if strategy_type not in self.populations:
            return {}
        
        population = self.populations[strategy_type]
        fitness_values = [s.fitness for s in population]
        
        return {
            'population_size': len(population),
            'generation': self.generation,
            'best_fitness': max(fitness_values) if fitness_values else 0.0,
            'average_fitness': sum(fitness_values) / len(fitness_values) if fitness_values else 0.0,
            'worst_fitness': min(fitness_values) if fitness_values else 0.0,
            'fitness_variance': sum((f - sum(fitness_values) / len(fitness_values))**2 
                                   for f in fitness_values) / len(fitness_values) if fitness_values else 0.0
        }
