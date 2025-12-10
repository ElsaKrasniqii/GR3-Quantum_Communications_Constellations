import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt
import json
from datetime import datetime
import warnings
import time

warnings.filterwarnings('ignore')

try:
    import pygmo as pg
    PYGMO_AVAILABLE = True
except ImportError:
    PYGMO_AVAILABLE = False
    print("Pygmo not available - optimization disabled")
    print("Install with: pip install pygmo")

try:
    from constellation_udp import constellation_udp
except Exception as e:
    raise RuntimeError(f"Failed loading constellation_udp: {e}")


class QuantumCommunicationsOptimizer:

    def __init__(self, population_size=48, generations=100, random_seed=42, use_multithreading=False):
        original_size = population_size

        if population_size < 5:
            population_size = 8
            print(f"⚠ Warning: Population too small ({original_size}). Increased to {population_size}.")

        if population_size % 4 != 0:
            population_size = ((population_size + 3) // 4) * 4
            print(f"⚠ Warning: Population {original_size} adjusted to {population_size} (must be multiple of 4).")

        self.population = population_size
        self.generations = generations
        self.seed = random_seed
        self.use_multithreading = use_multithreading

        self.udp = constellation_udp()
        self.best_solutions = []
        np.random.seed(self.seed)

    def _add_heuristic_solutions(self, pop):
        try:
            example_sol = self.udp.example()
            bounds = self.udp.get_bounds()
            lower = np.array(bounds[0])
            upper = np.array(bounds[1])

            if len(example_sol) == len(lower):
                pop.set_x(0, example_sol)

            for i in range(1, min(5, len(pop))):
                random_sol = lower + np.random.random(len(lower)) * (upper - lower)
                pop.set_x(i, random_sol)

        except Exception as e:
            if len(pop) > 0:
                print(f"Warning: Could not add heuristic solutions: {e}")

        return pop

    def optimize(self, verbose=True):

        prob = pg.problem(self.udp)

        if verbose:
            print("\n" + "=" * 60)
            print("          RUNNING OPTIMIZATION WITH NSGA-II")
            print("=" * 60)
            print(f"Population: {self.population}")
            print(f"Generations: {self.generations}")
            print(f"Seed: {self.seed}")
            print(f"Problem: {self.udp}")
            print(f"Number of objectives: {prob.get_nobj()}")
            print(f"Dimension: {prob.get_nx()}")
            print("-" * 60)

        algo = pg.algorithm(
            pg.nsga2(gen=1, cr=0.9, eta_c=10, m=0.01, eta_m=20)
        )

        pop = pg.population(prob, size=self.population, seed=self.seed)
        pop = self._add_heuristic_solutions(pop)

        hypervolume_scores = []

        print("\nStarting evolution...\n")

        for gen_idx in range(self.generations):
            pop = algo.evolve(pop)
            fits = pop.get_f()

            ideal1 = np.min(fits[:, 0])
            ideal2 = np.min(fits[:, 1])

            hv = self.compute_hypervolume(fits)
            hypervolume_scores.append(hv)

            if verbose:
                print(
                    f"Generation {gen_idx + 1} | "
                    f"ideal1: {ideal1:.6f} | "
                    f"ideal2: {ideal2:.6f} | "
                    f"Hypervolume: {hv:.6f}"
                )

        fits = pop.get_f()
        xs = pop.get_x()

        try:
            if fits.shape[1] == 2:
                front = pg.non_dominated_front_2d(fits)
            else:
                front = pg.non_dominated_front(fits)
        except Exception:
            front = list(range(len(fits)))

        sols = []

        for idx in front:
            f = fits[idx]
            x = xs[idx]
            sols.append({
                "x": x,
                "fitness": [float(f[0]), float(f[1])],
                "pareto_optimal": True
            })

        if len(sols) < 10:
            for idx in range(len(fits)):
                if idx not in front:
                    f = fits[idx]
                    x = xs[idx]
                    sols.append({
                        "x": x,
                        "fitness": [float(f[0]), float(f[1])],
                        "pareto_optimal": False
                    })
                    if len(sols) >= 20:
                        break

        sols.sort(key=lambda s: s["fitness"][0])
        self.best_solutions = sols

        if verbose:
            pareto_count = len([s for s in sols if s["pareto_optimal"]])
            print(f"✅ Found {pareto_count} Pareto-optimal solutions.")
            print(f"✅ Total solutions saved in memory: {len(sols)}")

        return sols

    def compute_hypervolume(self, fits):
        try:
            if fits.shape[1] == 2:
                reference_point = [1.2, 1.4]
                return pg.hypervolume(fits).compute(reference_point)
            return 0.0
        except Exception as e:
            print(f"Gabim gjatë llogaritjes së hypervolume: {e}")
            return 0.0

    def analyze_solutions(self, show_top=10):
        if not self.best_solutions:
            print("No solutions available. Run optimize() first.")
            return

        print(f"\n=== SOLUTION ANALYSIS (showing top {show_top}) ===")
        print(f"Total solutions: {len(self.best_solutions)}")

        print("\n--- TOP SOLUTIONS ---")
        for i, sol in enumerate(self.best_solutions[:show_top]):
            pareto_label = "(Pareto-optimal)" if sol.get("pareto_optimal", False) else ""
            print(f"\nSolution #{i+1} {pareto_label}")
            print(f"  J1 (Communication cost): {sol['fitness'][0]:.6f}")
            print(f"  J2 (Infrastructure cost): {sol['fitness'][1]:.6f}")
