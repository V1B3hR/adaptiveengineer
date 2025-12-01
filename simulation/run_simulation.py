import random
import math
import time
from core.evolution_engine import EvolutionEngine, StrategyType
from core.adaptive_learning import Observation, BehaviorType
from core.symbiosis_fitness import evaluate_symbiosis

# --- KREATOR ŚWIATA (SCENARIUSZE) ---

def generate_bio_rhythm_scenario(length=200, chaos_level=0.0):
    """
    Generuje dane przypominające biologiczny rytm (np. bicie serca serwera),
    przerywany anomaliami (chorobami/atakami).
    """
    data = []
    base_val = 10.0
    
    for t in range(length):
        # 1. Normalny rytm (sinusoida + lekki szum)
        val = base_val + math.sin(t * 0.2) * 2.0 + random.gauss(0, 0.2)
        
        # 2. Zdarzenia losowe (Chaos/Fizyka)
        if random.random() < chaos_level:
            val += random.uniform(-5.0, 5.0) # Szum tła (ciekawość powinna to lubić)

        # 3. Anomalie (Choroba/Atak) - to testuje frustrację i agresję
        # Występują rzadko, ale są silne
        if 50 < t < 70:  # "Infekcja" w środku symulacji
            val += 8.0   # Skok powyżej normy
        
        # Tworzymy obserwację
        obs = Observation(
            timestamp=time.time() + t,
            value=val,
            behavior_type=BehaviorType.SYMBIOTIC_TEST
        )
        data.append(obs)
            
    return data

# --- KONFIGURACJA EWOLUCJI ---

def run_cyber_evolution():
    print("=== INICJALIZACJA SYMULACJI: CYFROWA BIOLOGIA v1 ===")
    
    # 1. Definiujemy DNA (zakresy parametrów)
    # Ewolucja będzie szukać idealnej kombinacji tych cech
    dna_ranges = {
        # Cechy podstawowe
        "learning_rate": (0.001, 0.1),
        
        # Osobowość (To nas najbardziej interesuje!)
        "calmness": (0.0, 1.0),          # 0.0=Wybuchowy, 1.0=Stoik
        "curiosity_drive": (0.0, 0.5),   # Jak bardzo lubi nowości
        "motivation_memory": (0.8, 0.99),# Jak długo pamięta sukcesy
        
        # Reaktywność emocjonalna
        "threshold_sensitivity": (0.1, 0.9),
        "motivation_sensitivity": (0.1, 0.9),
        
        # Fizjologia
        "joy_gain": (0.01, 0.1),
        "motivation_decay_per_sec": (0.0001, 0.001)
    }

    # 2. Tworzymy silnik ewolucyjny
    engine = EvolutionEngine(
        population_size=20,      # Mała populacja dla testu
        mutation_rate=0.1,
        elitism_count=3,
        persistence_dir="sim_data"
    )

    # Inicjalizacja populacji "Symbiotic Test"
    engine.initialize_population(StrategyType.THRESHOLD, dna_ranges) 
    # (Używamy typu THRESHOLD jako kontenera, ale traktujemy to jako Symbiotic)

    # 3. Pętla Ewolucyjna
    generations = 5
    scenario_data = generate_bio_rhythm_scenario(length=150, chaos_level=0.05)

    print(f"\nScenariusz wygenerowany: {len(scenario_data)} próbek.")
    print("Rozpoczynam proces selekcji naturalnej...\n")

    for gen in range(generations):
        print(f"--- POKOLENIE {gen + 1} ---")
        
        # Funkcja oceniająca (Fitness Function)
        # To tutaj most (symbiosis_fitness) łączy Ewolucję z AI
        def evaluation_wrapper(strategy):
            # Każdy osobnik przeżywa ten sam scenariusz
            return evaluate_symbiosis(strategy, scenario_data)

        # Uruchom ewolucję
        stats = engine.evolve_generation(StrategyType.THRESHOLD, evaluation_wrapper)
        
        best_dna = engine.get_best_strategy(StrategyType.THRESHOLD)
        
        print(f"  Najlepszy Fitness (Radość): {stats['best_fitness']:.4f}")
        print(f"  Średnia populacji: {stats['avg_fitness']:.4f}")
        
        # Podgląd zwycięzcy tej rundy
        if best_dna:
            p = best_dna.parameters
            personality = "Nieznana"
            if p['calmness'] > 0.7: personality = "Mędrzec (Wysoki Spokój)"
            elif p['calmness'] < 0.3: personality = "Wojownik (Niski Spokój)"
            elif p['curiosity_drive'] > 0.3: personality = "Odkrywca (Ciekawski)"
            else: personality = "Zbalansowany"
            
            print(f"  Dominujący Fenotyp: {personality}")
            print(f"  Cechy: Calm={p['calmness']:.2f}, Curiosity={p['curiosity_drive']:.2f}, LearnRate={p['learning_rate']:.3f}")

    print("\n=== SYMULACJA ZAKOŃCZONA ===")
    winner = engine.get_best_strategy(StrategyType.THRESHOLD)
    print("Zwycięski Genotyp (Najlepsze AI do tego środowiska):")
    print(winner.parameters)

if __name__ == "__main__":
    run_cyber_evolution()
