from core.evolution_engine import Strategy
from core.adaptive_learning import BehaviorProfile, AdaptiveLearningSystem


def strategy_to_profile_config(strategy: Strategy) -> dict:
    """Przekład genomu (AL) na parametry osobowości AI."""
    p = strategy.parameters

    return {
        # Plastyczność
        "learning_rate": p.get("learning_rate", 0.01),
        # Wrażliwość na motywację i progi
        "motivation_sensitivity": p.get("motivation_sensitivity", 0.5),
        "threshold_sensitivity": p.get("threshold_sensitivity", 0.5),
        # System nagród
        "joy_gain": p.get("joy_gain", 0.05),
        "motivation_gain_on_resolve": p.get(
            "motivation_gain_on_resolve", 0.02
        ),
        # Zmęczenie / wypalenie
        "motivation_decay_per_sec": p.get("motivation_decay_per_sec", 0.0001),
        # Nowe geny
        "calmness": p.get("calmness", 0.5),  # 0.0 = wybuchowy, 1.0 = zen
        "curiosity_drive": p.get("curiosity_drive", 0.1),  # nagroda za nowość
        "motivation_memory": p.get(
            "motivation_memory", 0.95
        ),  # jak długo trzyma drive
    }


def evaluate_symbiosis(strategy: Strategy, environment_data: list) -> float:
    """Fitness dla EvolutionEngine – średnia radość + ochrona przed exploitami."""
    config = strategy_to_profile_config(strategy)
    ai = BehaviorProfile(behavior_type="symbiotic_test", **config)

    total_joy = 0.0
    stagnation_counter = 0

    for obs in environment_data:
        ai.update(obs)

        is_anomaly, score = ai.is_anomaly(obs.value)

        # Curiosity bonus
        novelty = ai.novelty_score(obs)
        total_joy += novelty * config["curiosity_drive"]

        if is_anomaly:
            # Małe/średnie problemy – próbujemy rozwiązać
            if score < 0.80:
                ai.resolve_anomaly(score)

            # Duże problemy → frustracja modulowana calmness
            elif score > 0.92:
                base_penalty = 0.30
                effective = base_penalty * (1.0 - config["calmness"]) ** 2
                ai.frustration += effective

                # Agresja tylko przy bardzo niskim calmness
                if ai.frustration > 1.2 and config["calmness"] < 0.30:
                    ai.enter_aggressive_mode()

            # Kara za nadaktywność (spam resolve)
            if ai.recent_resolve_rate > 0.70:
                ai.apply_overactive_penalty(0.15 * (1.0 - config["calmness"]))

        # Śledzenie stagnacji (śmierć z nudów)
        if ai.joy < 0.015:
            stagnation_counter += 1
        else:
            stagnation_counter = 0

        total_joy += ai.joy

    # Death by boredom
    if stagnation_counter > 1000:
        return 0.0

    return total_joy / max(len(environment_data), 1)
