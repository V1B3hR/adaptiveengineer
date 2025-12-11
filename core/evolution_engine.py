"""
evolution_engine.py - Family Edition v4 (Advanced)
Multi-parent + uncle/aunt crossover → twin kids (older brother & younger sister)
8 strategy types | persistence | parallel eval | diversity | UUIDs
Advanced: adaptive mutation, early stopping, fitness cache, atomic persistence, timeouts/retries,
range-aware mutation and similarity, RNG seeding, stats history, seeding APIs.

Drop-in compatible with v3: same core classes and method names preserved, with safe extensions.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
    wait,
    FIRST_COMPLETED,
    Future,
    TimeoutError as FuturesTimeout,
)

logger = logging.getLogger(__name__)

# Types
ParameterRange = Tuple[float, float]
ParameterRanges = Dict[str, ParameterRange]
EvaluationFunction = Callable[["Strategy"], float]


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
    birth_order: str = (
        "singleton"  # singleton | older_brother | younger_sister
    )
    created_at: float = field(default_factory=time.time)
    evaluations: int = 0

    def mutate(
        self,
        mutation_rate: float = 0.05,
        mutation_strength: float = 0.1,
        rng=None,
        parameter_ranges: Optional[ParameterRanges] = None,
    ) -> "Strategy":
        """
        Mutate parameters with Gaussian noise.
        - Range-aware: clamps to provided parameter_ranges per key (fallback 0..1 if absent).
        - Uses provided RNG (thread-safe random.Random) if given.
        """
        import random as _random

        R = rng or _random
        new_params: Dict[str, float] = {}
        for k, v in self.parameters.items():
            if R.random() < mutation_rate:
                # Scale noise relative to absolute value or the range span
                span = None
                if parameter_ranges and k in parameter_ranges:
                    lo, hi = parameter_ranges[k]
                    span = max(1e-12, hi - lo)
                scale_base = abs(v) if not span else span
                noise = R.gauss(0, mutation_strength * (scale_base or 1.0))
                nv = v + noise
                if parameter_ranges and k in parameter_ranges:
                    lo, hi = parameter_ranges[k]
                    nv = max(lo, min(hi, nv))
                else:
                    nv = max(0.0, min(1.0, nv))
                new_params[k] = nv
            else:
                new_params[k] = v
        return Strategy(
            strategy_id=str(uuid.uuid4()),
            strategy_type=self.strategy_type,
            parameters=new_params,
            generation=self.generation + 1,
            parent_ids=[self.strategy_id],
            birth_order="singleton",
        )

    @staticmethod
    def family_crossover(
        parents: List["Strategy"],
        uncle_aunt: Optional[List["Strategy"]] = None,
        rng=None,
    ) -> Tuple["Strategy", "Strategy"]:
        """
        Multi-parent crossover with optional uncle/aunt contributions.
        Older brother: weighted draw favoring direct parents.
        Younger sister: uniform draw across all contributors.
        """
        import random as _random

        R = rng or _random

        uncle_aunt = uncle_aunt or []
        all_genes = parents + uncle_aunt
        if not parents:
            raise ValueError("family_crossover requires at least 1 parent")
        if not all_genes:
            raise ValueError("family_crossover requires genetic sources")

        keys = list(parents[0].parameters.keys())
        older_params: Dict[str, float] = {}
        younger_params: Dict[str, float] = {}

        for key in keys:
            sources = [s.parameters[key] for s in all_genes]
            weights = [
                2 if i < len(parents) else 1 for i in range(len(sources))
            ]
            try:
                older_params[key] = R.choices(sources, weights=weights, k=1)[0]  # type: ignore[attr-defined]
            except AttributeError:
                # Fallback for environments without Random.choices (should be available in py3.6+)
                # Weighted roulette wheel
                total = float(sum(weights))
                pick = R.random() * total
                cur = 0.0
                for val, w in zip(sources, weights):
                    cur += w
                    if cur >= pick:
                        older_params[key] = val
                        break
                else:
                    older_params[key] = sources[-1]
            younger_params[key] = R.choice(sources)

        g = max(p.generation for p in parents) + 1
        older = Strategy(
            strategy_id=str(uuid.uuid4()),
            strategy_type=parents[0].strategy_type,
            parameters=older_params,
            generation=g,
            parent_ids=[p.strategy_id for p in parents],
            uncle_aunt_ids=[u.strategy_id for u in uncle_aunt],
            birth_order="older_brother",
        )
        younger = Strategy(
            strategy_id=str(uuid.uuid4()),
            strategy_type=parents[0].strategy_type,
            parameters=younger_params,
            generation=g,
            parent_ids=[p.strategy_id for p in parents],
            uncle_aunt_ids=[u.strategy_id for u in uncle_aunt],
            birth_order="younger_sister",
        )
        return older, younger

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["strategy_type"] = self.strategy_type.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Strategy":
        data = data.copy()
        data["strategy_type"] = StrategyType(data["strategy_type"])
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
        persistence_dir: str = "populations",
        # Advanced controls
        random_seed: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        retry_failed_evals: int = 0,
        checkpoint_interval: int = 1,
        early_stopping_rounds: Optional[int] = None,
        target_fitness: Optional[float] = None,
        adaptive_mutation: bool = True,
    ):
        """
        Advanced Evolution Engine.
        - timeout_seconds: cancel outstanding parallel evaluations after this many seconds (batch-level).
        - retry_failed_evals: resubmit evaluations that error/timeout up to this many times.
        - early_stopping_rounds: stop evolving if best_fitness doesn't improve for this many generations.
        - target_fitness: stop evolving once best_fitness >= target.
        - adaptive_mutation: automatically adjust mutation_rate based on diversity & improvement.
        """
        import random as _random

        self.population_size = population_size
        self.base_mutation_rate = mutation_rate
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size
        self.max_workers = max_workers
        self.persistence_dir = persistence_dir

        self.timeout_seconds = timeout_seconds
        self.retry_failed_evals = max(0, retry_failed_evals)
        self.checkpoint_interval = max(1, checkpoint_interval)
        self.early_stopping_rounds = early_stopping_rounds
        self.target_fitness = target_fitness
        self.adaptive_mutation = adaptive_mutation

        # RNG encapsulation for reproducibility
        self.rng = _random.Random()
        if random_seed is not None:
            self.rng.seed(int(random_seed))
        else:
            self.rng.seed(int(time.time() * 1e6) ^ os.getpid())

        self.populations: Dict[StrategyType, List[Strategy]] = {}
        self.parameter_spaces: Dict[StrategyType, ParameterRanges] = {}
        self.generation = 0
        self.best_strategies: Dict[StrategyType, Strategy] = {}
        self.best_fitness_history: Dict[StrategyType, List[float]] = {}
        self._no_improve_rounds: Dict[StrategyType, int] = {}
        self._early_stopped: Dict[StrategyType, bool] = {}

        # Fitness cache keyed by parameter fingerprint
        self._fitness_cache: Dict[str, float] = {}

        os.makedirs(persistence_dir, exist_ok=True)
        logger.info("EvolutionEngine Family Edition (Advanced) ready")

    # ---------- Population management ----------

    def initialize_population(
        self, strategy_type: StrategyType, parameter_ranges: ParameterRanges
    ) -> None:
        """
        Initialize a fresh population sampled uniformly from given parameter_ranges per key.
        """
        self.parameter_spaces[strategy_type] = parameter_ranges
        pop = [
            Strategy(
                strategy_id=str(uuid.uuid4()),
                strategy_type=strategy_type,
                parameters={
                    k: self.rng.uniform(mi, ma)
                    for k, (mi, ma) in parameter_ranges.items()
                },
                generation=0,
            )
            for _ in range(self.population_size)
        ]
        self.populations[strategy_type] = pop
        self.best_fitness_history.setdefault(strategy_type, [])
        self._no_improve_rounds[strategy_type] = 0
        self._early_stopped[strategy_type] = False
        self.generation = 0
        self.save_population(strategy_type)
        logger.info(
            f"Initialized {strategy_type.value} population ({self.population_size})"
        )

    def seed_population(
        self, strategy_type: StrategyType, seeds: Iterable[Dict[str, float]]
    ) -> None:
        """
        Seed the population with provided parameter dicts (filled up to population_size with randoms).
        """
        if strategy_type not in self.parameter_spaces:
            raise ValueError(
                "Initialize parameter ranges before seeding population."
            )
        param_ranges = self.parameter_spaces[strategy_type]
        seeded: List[Strategy] = []
        for p in seeds:
            params = {}
            for k, (lo, hi) in param_ranges.items():
                v = p.get(k, self.rng.uniform(lo, hi))
                v = max(lo, min(hi, float(v)))
                params[k] = v
            seeded.append(
                Strategy(
                    strategy_id=str(uuid.uuid4()),
                    strategy_type=strategy_type,
                    parameters=params,
                )
            )
        # Fill with randoms to population size
        while len(seeded) < self.population_size:
            seeded.append(
                Strategy(
                    strategy_id=str(uuid.uuid4()),
                    strategy_type=strategy_type,
                    parameters={
                        k: self.rng.uniform(lo, hi)
                        for k, (lo, hi) in param_ranges.items()
                    },
                )
            )
        self.populations[strategy_type] = seeded[: self.population_size]
        self.save_population(strategy_type)

    def get_population(self, strategy_type: StrategyType) -> List[Strategy]:
        return list(self.populations.get(strategy_type, []))

    # ---------- Evaluation ----------

    def _fingerprint(self, strategy: Strategy) -> str:
        # Stable fingerprint by sorted keys and rounded floats
        items = tuple(
            (k, round(float(v), 8))
            for k, v in sorted(strategy.parameters.items())
        )
        return json.dumps(items)

    def evaluate_fitness(
        self, strategy: Strategy, eval_fn: EvaluationFunction
    ) -> float:
        """
        Evaluate a single strategy with caching.
        """
        fp = self._fingerprint(strategy)
        cached = self._fitness_cache.get(fp)
        if cached is not None:
            strategy.fitness = cached
            # Do not increment evaluations for cache hits; keep real evaluation count meaningful
            return strategy.fitness

        fitness = eval_fn(strategy)
        strategy.fitness = float(fitness)
        strategy.evaluations += 1
        self._fitness_cache[fp] = strategy.fitness
        return strategy.fitness

    def parallel_evaluate(
        self, population: List[Strategy], eval_fn: EvaluationFunction
    ) -> None:
        """
        Evaluate strategies in parallel.
        - Skips cache hits.
        - Supports global batch timeout; cancels outstanding futures.
        - Retries failed/timeout items up to retry_failed_evals times.
        """
        # Determine which to evaluate (new or current-gen)
        to_eval = [
            s
            for s in population
            if s.evaluations == 0 or s.generation == self.generation
        ]

        # Exclude cache hits upfront
        pending: List[Strategy] = []
        for s in to_eval:
            if self._fingerprint(s) in self._fitness_cache:
                # Populate from cache
                s.fitness = self._fitness_cache[self._fingerprint(s)]
            else:
                pending.append(s)

        if not pending:
            return

        attempts = 0
        remaining: List[Strategy] = pending
        while remaining and attempts <= self.retry_failed_evals:
            attempts += 1
            failed: List[Strategy] = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                future_map: Dict[Future, Strategy] = {
                    ex.submit(self.evaluate_fitness, s, eval_fn): s
                    for s in remaining
                }

                if self.timeout_seconds is None or self.timeout_seconds <= 0:
                    for fut in as_completed(future_map):
                        s = future_map[fut]
                        try:
                            fut.result()
                        except Exception as e:  # noqa: BLE001
                            logger.error(f"Eval error {s.strategy_id}: {e}")
                            failed.append(s)
                else:
                    # Global batch timeout
                    done, not_done = wait(
                        set(future_map.keys()),
                        timeout=self.timeout_seconds,
                        return_when=FIRST_COMPLETED,
                    )
                    # Keep waiting until either all done or timeout window has elapsed
                    # We implement soft timeout: after first timeout window, we cancel any that still run.
                    # Gather results for done futures
                    for fut in list(done):
                        s = future_map[fut]
                        try:
                            fut.result()
                        except Exception as e:  # noqa: BLE001
                            logger.error(f"Eval error {s.strategy_id}: {e}")
                            failed.append(s)

                    # Cancel outstanding futures as timed out
                    for fut in not_done:
                        s = future_map[fut]
                        try:
                            fut.cancel()
                        except Exception:
                            pass
                        logger.warning(
                            f"Eval timeout for {s.strategy_id} after {self.timeout_seconds}s"
                        )
                        failed.append(s)

            # Prepare for retry loop
            remaining = failed

            # Mark definitive failures at last attempt
            if remaining and attempts > self.retry_failed_evals:
                for s in remaining:
                    s.fitness = float("-inf")
                    # Count as an attempted evaluation to avoid infinite re-queues externally
                    s.evaluations += 1

    # ---------- Selection ----------

    def tournament_selection(self, pop: List[Strategy]) -> Strategy:
        k = min(self.tournament_size, max(1, len(pop)))
        # random.sample will raise if k > len; ensure proper k
        # Use RNG encapsulation for reproducibility
        candidates = self.rng.sample(pop, k) if len(pop) >= k else list(pop)
        return max(candidates, key=lambda s: s.fitness)

    # ---------- Evolution ----------

    def evolve_generation(
        self,
        strategy_type: StrategyType,
        evaluation_function: EvaluationFunction,
    ) -> Dict[str, Any]:
        """
        Evolve population one generation:
        - Parallel evaluation with caching, timeout, retries.
        - Elitism, family crossover with uncles/aunts, mutation.
        - Diversity maintenance (range-aware cosine).
        - Adaptive mutation (optional).
        - Early stopping and target fitness checks.
        - Atomic persistence + stats history.
        Returns runtime stats dict.
        """
        if strategy_type not in self.populations:
            raise ValueError(f"No population for {strategy_type}")
        if self._early_stopped.get(strategy_type):
            return {
                "generation": self.generation,
                "best_fitness": (
                    self.best_strategies.get(
                        strategy_type, Strategy("", strategy_type, {})
                    ).fitness
                    if strategy_type in self.best_strategies
                    else None
                ),
                "avg_fitness": None,
                "early_stopped": True,
                "reason": "early_stopping_triggered",
            }

        pop = self.populations[strategy_type]
        param_ranges = self.parameter_spaces.get(strategy_type, {})

        # Evaluate and rank
        self.parallel_evaluate(pop, evaluation_function)
        pop.sort(key=lambda s: s.fitness, reverse=True)

        best = pop[0]
        self.best_strategies[strategy_type] = best
        best_history = self.best_fitness_history.setdefault(strategy_type, [])
        prev_best = best_history[-1] if best_history else None
        best_history.append(best.fitness)

        new_pop: List[Strategy] = pop[: self.elitism_count]

        # Breeding
        while len(new_pop) < self.population_size:
            # Crossover path
            if self.rng.random() < self.crossover_rate and len(pop) >= 4:
                p1 = self.tournament_selection(pop)
                p2 = self.tournament_selection(pop)
                # Ensure different parents
                tries = 0
                while p2.strategy_id == p1.strategy_id and tries < 5:
                    p2 = self.tournament_selection(pop)
                    tries += 1
                # Pick up to 2 uncles/aunts distinct from parents
                candidates = [
                    s
                    for s in pop
                    if s.strategy_id not in (p1.strategy_id, p2.strategy_id)
                ]
                uncles = (
                    self.rng.sample(candidates, k=min(2, len(candidates)))
                    if candidates
                    else []
                )
                older, younger = Strategy.family_crossover(
                    [p1, p2], uncles, rng=self.rng
                )
                new_pop.extend([older, younger])
            else:
                parent = self.tournament_selection(pop)
                child = parent.mutate(
                    self.mutation_rate,
                    self.mutation_strength,
                    rng=self.rng,
                    parameter_ranges=param_ranges,
                )
                new_pop.append(child)

        # Diversity preservation
        self.maintain_diversity(new_pop, param_ranges)

        # Trim to correct size
        self.populations[strategy_type] = new_pop[: self.population_size]
        self.generation += 1

        # Adaptive mutation rate based on similarity and improvement
        if self.adaptive_mutation:
            avg_sim = self._average_similarity(
                self.populations[strategy_type], param_ranges
            )
            improved = prev_best is None or (best.fitness > prev_best + 1e-12)
            # If no improvement and population is too similar -> increase mutation
            if not improved and avg_sim > 0.95:
                self.mutation_rate = min(0.5, self.mutation_rate * 1.25 + 0.01)
            # If improving and diverse -> gently reduce mutation
            elif improved and avg_sim < 0.85:
                self.mutation_rate = max(
                    self.base_mutation_rate * 0.5, self.mutation_rate * 0.9
                )
            # Bound mutation rate
            self.mutation_rate = float(max(1e-4, min(0.9, self.mutation_rate)))

        # Persistence
        if self.generation % self.checkpoint_interval == 0:
            self.save_population(strategy_type)
        self._append_stats(strategy_type, self.populations[strategy_type])

        # Stats
        fits = [s.fitness for s in self.populations[strategy_type]]
        avg = sum(fits) / len(fits) if fits else float("nan")
        variance = (
            sum((f - avg) ** 2 for f in fits) / len(fits)
            if fits
            else float("nan")
        )

        stats = {
            "generation": self.generation,
            "best_fitness": best.fitness,
            "avg_fitness": avg,
            "worst_fitness": min(fits) if fits else None,
            "variance": variance,
            "best_id": best.strategy_id,
            "population_size": len(self.populations[strategy_type]),
            "mutation_rate": self.mutation_rate,
            "avg_similarity": self._average_similarity(
                self.populations[strategy_type], param_ranges
            ),
        }

        # Early stopping checks
        stop_reason: Optional[str] = None
        if (
            self.target_fitness is not None
            and best.fitness >= self.target_fitness
        ):
            stop_reason = "target_fitness_reached"
        else:
            # Improvement tracking
            if prev_best is None or best.fitness > (prev_best + 1e-12):
                self._no_improve_rounds[strategy_type] = 0
            else:
                self._no_improve_rounds[strategy_type] = (
                    self._no_improve_rounds.get(strategy_type, 0) + 1
                )
                if (
                    self.early_stopping_rounds is not None
                    and self._no_improve_rounds[strategy_type]
                    >= self.early_stopping_rounds
                ):
                    stop_reason = "early_stopping_no_improvement"

        if stop_reason:
            self._early_stopped[strategy_type] = True
            stats["early_stopped"] = True
            stats["reason"] = stop_reason

        logger.debug(
            f"Gen {self.generation} {strategy_type.value}: best={stats['best_fitness']:.6f} avg={stats['avg_fitness']:.6f} "
            f"mut={self.mutation_rate:.4f} sim={stats['avg_similarity']:.4f}"
        )
        return stats

    # ---------- Diversity ----------

    def maintain_diversity(
        self, pop: List[Strategy], param_ranges: ParameterRanges
    ) -> None:
        """
        Keep only sufficiently dissimilar high-fitness individuals.
        Range-aware cosine similarity thresholding with refill by mutation.
        """
        unique: List[Strategy] = []
        for s in sorted(pop, key=lambda x: x.fitness, reverse=True):
            if not any(
                self._similarity(s, u, param_ranges) > 0.95 for u in unique
            ):
                unique.append(s)
        # If we lost too many, refill by mutating elite with stronger noise
        while len(unique) < self.population_size:
            base = self.rng.choice(unique) if unique else self.rng.choice(pop)
            unique.append(
                base.mutate(
                    mutation_rate=min(1.0, max(0.2, self.mutation_rate * 1.5)),
                    mutation_strength=min(1.0, self.mutation_strength * 1.5),
                    rng=self.rng,
                    parameter_ranges=param_ranges,
                )
            )
        pop[:] = unique[: self.population_size]

    def _normalize_vector(
        self, params: Dict[str, float], param_ranges: ParameterRanges
    ) -> List[float]:
        vec: List[float] = []
        for k in sorted(params.keys()):
            v = params[k]
            if k in param_ranges:
                lo, hi = param_ranges[k]
                span = max(1e-12, hi - lo)
                nv = (v - lo) / span
            else:
                # Assume already ~[0,1]
                nv = v
            vec.append(float(nv))
        return vec

    def _similarity(
        self, s1: Strategy, s2: Strategy, param_ranges: ParameterRanges
    ) -> float:
        v1 = self._normalize_vector(s1.parameters, param_ranges)
        v2 = self._normalize_vector(s2.parameters, param_ranges)
        dot = sum(a * b for a, b in zip(v1, v2))
        mag = (sum(a * a for a in v1) ** 0.5) * (sum(b * b for b in v2) ** 0.5)
        return dot / (mag + 1e-12) if mag > 0 else 0.0

    def _average_similarity(
        self, pop: List[Strategy], param_ranges: ParameterRanges
    ) -> float:
        n = len(pop)
        if n < 2:
            return 0.0
        total = 0.0
        count = 0
        # Sampled pairwise similarity to avoid O(n^2) for large pops
        limit = min(200, n * (n - 1) // 2)
        # Simple sampling of indices
        i = 0
        while count < limit:
            a = self.rng.randrange(0, n)
            b = self.rng.randrange(0, n - 1)
            if b >= a:
                b += 1
            total += self._similarity(pop[a], pop[b], param_ranges)
            count += 1
            i += 1
        if count == 0:
            return 0.0
        return total / count

    # ---------- Best ----------

    def get_best_strategy(
        self, strategy_type: StrategyType
    ) -> Optional[Strategy]:
        return self.best_strategies.get(strategy_type)

    # ---------- Persistence ----------

    def _atomic_write(self, path: str, data: str) -> None:
        tmp_path = f"{path}.tmp-{uuid.uuid4()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp_path, path)

    def save_population(self, strategy_type: StrategyType) -> None:
        """
        Persist current population and metadata atomically.
        - populations/{type}_pop.json: array of strategies
        - populations/{type}_meta.json: engine + generation metadata
        - populations/{type}_stats.jsonl: per-generation stats append-only
        """
        pop_path = os.path.join(
            self.persistence_dir, f"{strategy_type.value}_pop.json"
        )
        meta_path = os.path.join(
            self.persistence_dir, f"{strategy_type.value}_meta.json"
        )

        pop_payload = json.dumps(
            [s.to_dict() for s in self.populations[strategy_type]], indent=2
        )
        self._atomic_write(pop_path, pop_payload)

        meta = {
            "generation": self.generation,
            "population_size": len(self.populations[strategy_type]),
            "mutation_rate": self.mutation_rate,
            "mutation_strength": self.mutation_strength,
            "crossover_rate": self.crossover_rate,
            "elitism_count": self.elitism_count,
            "tournament_size": self.tournament_size,
            "max_workers": self.max_workers,
            "random_state": None,  # intentionally not dumping for portability
            "timestamp": time.time(),
            "parameter_ranges": self.parameter_spaces.get(strategy_type, {}),
        }
        self._atomic_write(meta_path, json.dumps(meta, indent=2))
        logger.debug(
            f"Saved population & meta for {strategy_type.value} at gen {self.generation}"
        )

    def _append_stats(
        self, strategy_type: StrategyType, pop: List[Strategy]
    ) -> None:
        stats_path = os.path.join(
            self.persistence_dir, f"{strategy_type.value}_stats.jsonl"
        )
        fits = [s.fitness for s in pop]
        if not fits:
            return
        avg = sum(fits) / len(fits)
        payload = {
            "timestamp": time.time(),
            "generation": self.generation,
            "best_fitness": max(fits),
            "avg_fitness": avg,
            "worst_fitness": min(fits),
            "variance": sum((f - avg) ** 2 for f in fits) / len(fits),
            "population_size": len(pop),
            "mutation_rate": self.mutation_rate,
        }
        line = json.dumps(payload)
        # Append atomically: write to temp then append via rename+append isn't atomic; we can simple append safely.
        with open(stats_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def load_population(self, strategy_type: StrategyType) -> bool:
        """
        Load population if present; restores generation from max(strategy.generation)
        and parameter ranges from meta if available.
        """
        pop_path = os.path.join(
            self.persistence_dir, f"{strategy_type.value}_pop.json"
        )
        meta_path = os.path.join(
            self.persistence_dir, f"{strategy_type.value}_meta.json"
        )
        if not os.path.exists(pop_path):
            return False
        with open(pop_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.populations[strategy_type] = [Strategy.from_dict(d) for d in data]
        try:
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                pr = meta.get("parameter_ranges")
                if isinstance(pr, dict):
                    # ensure tuple ranges
                    self.parameter_spaces[strategy_type] = {k: tuple(v) for k, v in pr.items()}  # type: ignore[assignment]
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Failed to load meta for {strategy_type.value}: {e}"
            )

        self.generation = max(
            (s.generation for s in self.populations[strategy_type]), default=0
        )
        # Rebuild best tracker
        if self.populations[strategy_type]:
            best = max(
                self.populations[strategy_type], key=lambda s: s.fitness
            )
            self.best_strategies[strategy_type] = best
        logger.info(
            f"Loaded {strategy_type.value} population from {pop_path} at gen {self.generation}"
        )
        return True
