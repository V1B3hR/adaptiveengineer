"""
evolution_engine.py - Family Edition v3
Multi-parent + uncle/aunt crossover → twin kids (older brother & younger sister)
8 strategy types | persistence | parallel eval | diversity | UUIDs
Ready for production drop-in replacement.
"""
import random
import time
import uuid
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class StrategyType(str, Enum):
    DETECTION = "detection"
    MITIGATION = "mitigation"
    RECOVERY = "recovery"
    THRESHOLD = "threshold"
    EVASION = "evasion"
    HARDENING = "hardening"
    FORENSICS = "forensics"
    DECEPTION = "deception"

@dataclass
class Strategy:
    strategy_id: str
    strategy_type: StrategyType
    parameters: Dict[str, float]
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    uncle_aunt_ids: List[str] = field(default_factory=list)
    birth_order: str = "singleton"  # singleton | older_brother | younger_sister
    created_at: float = field(default_factory=time.time)
    evaluations: int = 0

    def mutate(self, mutation_rate: float = 0.05, mutation_strength: float = 0.1) -> 'Strategy':
        new_params = {}
        for k, v in self.parameters.items():
            if random.random() < mutation_rate:
                noise = random.gauss(0, mutation_strength * (abs(v) or 1))
                new_params[k] = max(0.0, min(1.0, v + noise))
            else:
                new_params[k] = v
        return Strategy(
            strategy_id=str(uuid.uuid4()),
            strategy_type=self.strategy_type,
            parameters=new_params,
            generation=self.generation + 1,
            parent_ids=[self.strategy_id],
            birth_order="singleton"
        )

    @staticmethod
    def family_crossover(parents: List['Strategy'], uncle_aunt: List['Strategy'] = None) -> Tuple['Strategy', 'Strategy']:
        uncle_aunt = uncle_aunt or []
        all_genes = parents + uncle_aunt
        if not all_genes:
            raise ValueError("Need parents")
        keys = list(parents[0].parameters.keys())
        older_params, younger_params = {}, {}

        for key in keys:
            sources = [s.parameters[key] for s in all_genes]
            weights = [2 if i < len(parents) else 1 for i in range(len(sources))]
            older_params[key] = random.choices(sources, weights=weights, k=1)[0]
            younger_params[key] = random.choice(sources)

        g = max(p.generation for p in parents) + 1
        older = Strategy(
            strategy_id=str(uuid.uuid4()),
            strategy_type=parents[0].strategy_type,
            parameters=older_params,
            generation=g,
            parent_ids=[p.strategy_id for p in parents],
            uncle_aunt_ids=[u.strategy_id for u in uncle_aunt],
            birth_order="older_brother"
        )
        younger = Strategy(
            strategy_id=str(uuid.uuid4()),
            strategy_type=parents[0].strategy_type,
            parameters=younger_params,
            generation=g,
            parent_ids=[p.strategy_id for p in parents],
            uncle_aunt_ids=[u.strategy_id for u in uncle_aunt],
            birth_order="younger_sister"
        )
        return older, younger

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['strategy_type'] = self.strategy_type.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Strategy':
        data = data.copy()
        data['strategy_type'] = StrategyType(data['strategy_type'])
        return cls(**data)


class EvolutionEngine:
    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.05,
        mutation_strength: float = 0.1,
        crossover_rate: float = 0.8,
        elitism_count: int = 5,
        tournament_size: int = 5,
        max_workers: int = 4,
        persistence_dir: str = "populations"
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size
        self.max_workers = max_workers
        self.persistence_dir = persistence_dir

        self.populations: Dict[StrategyType, List[Strategy]] = {}
        self.generation = 0
        self.best_strategies: Dict[StrategyType, Strategy] = {}

        os.makedirs(persistence_dir, exist_ok=True)
        logger.info("EvolutionEngine Family Edition ready")

    def initialize_population(self, strategy_type: StrategyType, parameter_ranges: Dict[str, Tuple[float, float]]) -> None:
        pop = [
            Strategy(
                strategy_id=str(uuid.uuid4()),
                strategy_type=strategy_type,
                parameters={k: random.uniform(mi, ma) for k, (mi, ma) in parameter_ranges.items()},
                generation=0
            )
            for _ in range(self.population_size)
        ]
        self.populations[strategy_type] = pop
        self.save_population(strategy_type)
        logger.info(f"Initialized {strategy_type} population ({self.population_size})")

    def evaluate_fitness(self, strategy: Strategy, eval_fn) -> float:
        strategy.fitness = eval_fn(strategy)
        strategy.evaluations += 1
        return strategy.fitness

    def parallel_evaluate(self, population: List[Strategy], eval_fn) -> None:
        to_eval = [s for s in population if s.evaluations == 0 or s.generation == self.generation]
        if not to_eval:
            return
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(self.evaluate_fitness, s, eval_fn): s for s in to_eval}
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    logger.error(f"Eval error {futures[f].strategy_id}: {e}")

    def tournament_selection(self, pop: List[Strategy]) -> Strategy:
        return max(random.sample(pop, min(self.tournament_size, len(pop))), key=lambda s: s.fitness)

    def evolve_generation(self, strategy_type: StrategyType, evaluation_function: Any) -> Dict[str, Any]:
        if strategy_type not in self.populations:
            raise ValueError(f"No population for {strategy_type}")

        pop = self.populations[strategy_type]
        self.parallel_evaluate(pop, evaluation_function)
        pop.sort(key=lambda s: s.fitness, reverse=True)

        best = pop[0]
        self.best_strategies[strategy_type] = best
        new_pop: List[Strategy] = pop[:self.elitism_count]

        while len(new_pop) < self.population_size:
            if random.random() < self.crossover_rate and len(pop) >= 4:
                p1 = self.tournament_selection(pop)
                p2 = self.tournament_selection(pop)
                while p2.strategy_id == p1.strategy_id:
                    p2 = self.tournament_selection(pop)
                uncles = random.sample([s for s in pop if s not in (p1, p2)], k=min(2, len(pop) - 2))
                older, younger = Strategy.family_crossover([p1, p2], uncles)
                new_pop.extend([older, younger])
            else:
                parent = self.tournament_selection(pop)
                new_pop.append(parent.mutate(self.mutation_rate, self.mutation_strength))

        self.maintain_diversity(new_pop)
        self.populations[strategy_type] = new_pop[:self.population_size]
        self.generation += 1
        self.save_population(strategy_type)

        fits = [s.fitness for s in new_pop]
        avg = sum(fits) / len(fits)
        stats = {
            "generation": self.generation,
            "best_fitness": max(fits),
            "avg_fitness": avg,
            "worst_fitness": min(fits),
            "variance": sum((f - avg) ** 2 for f in fits) / len(fits),
            "best_id": best.strategy_id,
            "population_size": len(new_pop)
        }
        logger.debug(f"Gen {self.generation} {strategy_type}: best={stats['best_fitness']:.4f}")
        return stats

    def maintain_diversity(self, pop: List[Strategy]) -> None:
        unique = []
        for s in sorted(pop, key=lambda x: x.fitness, reverse=True):
            if not any(self._similarity(s, u) > 0.95 for u in unique):
                unique.append(s)
        while len(unique) < self.population_size:
            unique.append(random.choice(unique).mutate(0.2, 0.3))
        pop[:] = unique[:self.population_size]

    def _similarity(self, s1: Strategy, s2: Strategy) -> float:
        v1 = list(s1.parameters.values())
        v2 = list(s2.parameters.values())
        dot = sum(a * b for a, b in zip(v1, v2))
        mag = (sum(a * a for a in v1) ** 0.5) * (sum(b * b for b in v2) ** 0.5)
        return dot / (mag + 1e-9) if mag > 0 else 0.0

    def get_best_strategy(self, strategy_type: StrategyType) -> Optional[Strategy]:
        return self.best_strategies.get(strategy_type)

    def save_population(self, strategy_type: StrategyType) -> None:
        path = os.path.join(self.persistence_dir, f"{strategy_type.value}_pop.json")
        json.dump([s.to_dict() for s in self.populations[strategy_type]], open(path, "w"), indent=2)

    def load_population(self, strategy_type: StrategyType) -> bool:
        path = os.path.join(self.persistence_dir, f"{strategy_type.value}_pop.json")
        if not os.path.exists(path):
            return False
        data = json.load(open(path))
        self.populations[strategy_type] = [Strategy.from_dict(d) for d in data]
        self.generation = max((s.generation for s in self.populations[strategy_type]), default=0)
        logger.info(f"Loaded {strategy_type} population from {path}")
        return True
