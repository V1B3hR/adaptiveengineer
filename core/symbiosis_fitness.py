"""
Symbiosis fitness evaluation for evolution engine. 

Evaluates how well a strategy performs in a symbiotic relationship
with the environment based on emotional responses and adaptation. 
"""

import numpy as np
from typing import List, Any


def evaluate_symbiosis(strategy, scenario_data: List[Any]) -> float:
    """
    Evaluate fitness of a strategy based on symbiotic behavior.
    
    This function measures how well an agent adapts to environmental
    changes, balancing exploration (curiosity) with stability (calmness).
    
    Fitness is calculated based on:
    - Joy accumulation (positive emotional state)
    - Stability (avoiding excessive stress/frustration)
    - Adaptation speed (learning from observations)
    
    Args:
        strategy: Strategy object with parameters dict containing:
            - learning_rate: How fast the agent learns
            - calmness: Baseline emotional stability
            - curiosity_drive: Exploration tendency
            - motivation_memory: How long motivation persists
            - joy_gain: Rate of joy accumulation
            
        scenario_data: List of Observation objects with:
            - timestamp: When observation occurred
            - value: Observed environmental value
            - behavior_type: Type of behavior being tested
    
    Returns:
        float: Fitness score (0.0 to 1.0, higher is better)
    """
    if not scenario_data:
        return 0.0
    
    # Extract strategy parameters
    params = strategy.parameters if hasattr(strategy, 'parameters') else {}
    learning_rate = params.get('learning_rate', 0.01)
    calmness = params.get('calmness', 0.5)
    curiosity_drive = params.get('curiosity_drive', 0.1)
    joy_gain = params.get('joy_gain', 0.05)
    
    # Initialize emotional state
    joy = 0.0
    stress = 0.0
    adaptation_score = 0.0
    
    # Baseline expectation (learns from data)
    expected_value = scenario_data[0].value if hasattr(scenario_data[0], 'value') else 10.0
    
    # Process each observation
    for i, obs in enumerate(scenario_data):
        value = obs.value if hasattr(obs, 'value') else 10.0
        
        # Calculate prediction error
        error = abs(value - expected_value)
        
        # Update expectation (learning)
        expected_value += learning_rate * (value - expected_value)
        
        # Emotional responses
        if error < 2.0:  # Low error = stable environment
            joy += joy_gain * calmness  # Calm agents gain more joy from stability
            stress = max(0.0, stress - 0.02)
        else:  # High error = surprising/chaotic environment
            curiosity_response = curiosity_drive * min(error, 5.0)
            joy += curiosity_response * 0.1  # Curious agents enjoy novelty
            stress += (1.0 - calmness) * error * 0.05  # Non-calm agents get stressed
        
        # Adaptation score: how well does agent handle change?
        if error < 3.0:
            adaptation_score += 0. 01
    
    # Normalize scores
    avg_joy = joy / len(scenario_data)
    avg_stress = stress / len(scenario_data)
    avg_adaptation = adaptation_score / len(scenario_data)
    
    # Fitness = joy - stress penalty + adaptation bonus
    fitness = max(0.0, min(1.0, avg_joy - (avg_stress * 0.5) + (avg_adaptation * 2.0)))
    
    return fitness
