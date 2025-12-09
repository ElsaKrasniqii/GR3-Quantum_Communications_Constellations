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
        # Sigurohu që popullsia plotëson kërkesat e NSGA-II
        original_size = population_size
        
        # Kërkesat e NSGA-II:
        # 1. Të paktën 5 individë
        # 2. Shumëfish i 4
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
        """Add heuristic solutions to the initial population."""
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
        # Krijo problemin
        prob = pg.problem(self.udp)

        if verbose:
            print("\n" + "="*60)
            print("          RUNNING OPTIMIZATION WITH NSGA-II")
            print("="*60)
            print(f"Population: {self.population}")
            print(f"Generations: {self.generations}")
            print(f"Seed: {self.seed}")
            print(f"Problem: {self.udp}")
            print(f"Number of objectives: {prob.get_nobj()}")
            print(f"Number of inequality constraints: {prob.get_nic()}")
            print(f"Number of equality constraints: {prob.get_nec()}")
            print(f"Dimension: {prob.get_nx()}")
            print("-"*60)
        
        # Krijo algoritmin
        algo = pg.algorithm(
            pg.nsga2(
                gen=self.generations,
                cr=0.9,
                eta_c=10,
                m=0.01,
                eta_m=20
            )
        )

        if verbose:
            algo.set_verbosity(1)
        

        pop = pg.population(prob, size=self.population, seed=self.seed)

        pop = self._add_heuristic_solutions(pop)

        fitness_history = {"generation": [], "best_J1": [], "best_J2": [], "avg_J1": [], "avg_J2": []}

        if verbose:
            print("\nStarting evolution...")
            print("-" * 50)

        # Ekzekuto optimizimin
        pop = algo.evolve(pop)
        
        # Nxjerr rezultatet
        fits = pop.get_f()
        xs = pop.get_x()

        # Merr frontin Pareto
        try:
            if fits.shape[1] == 2:
                front = pg.non_dominated_front_2d(fits)
            else:
                front = pg.non_dominated_front(fits)
        except:
            front = list(range(len(fits)))
        
        # Nxjerr zgjidhjet
        sols = []
        for idx in front:
            f = fits[idx]
            x = xs[idx]
            

            try:
                analysis = self.udp.analyze_solution(x)
                rover_violated = analysis['constraints']['rover_violated']
                sat_violated = analysis['constraints']['sat_violated']
                feasible = not (rover_violated or sat_violated)
                
                constraint_info = {
                    'rover_distance': analysis['distances']['rover_distance'],
                    'satellite_distance': analysis['distances']['satellite_distance'],
                    'rover_violated': rover_violated,
                    'sat_violated': sat_violated
                }
            except Exception as e:
                feasible = True
                constraint_info = {}

            try:
                crowding_distance = float(pop.get_cd()[idx])
            except:
                crowding_distance = 0.0

            sols.append({
                "x": x,
                "fitness": [float(f[0]), float(f[1])],
                "feasible": feasible,
                "constraint_info": constraint_info,
                "crowding_distance": crowding_distance,
                "pareto_optimal": True
            })

        # Shto diversitet
        if len(sols) < 10:
            for idx in range(len(fits)):
                if idx not in front:
                    f = fits[idx]
                    x = xs[idx]
                    
                    try:
                        analysis = self.udp.analyze_solution(x)
                        rover_violated = analysis['constraints']['rover_violated']
                        sat_violated = analysis['constraints']['sat_violated']
                        feasible = not (rover_violated or sat_violated)
                        
                        constraint_info = {
                            'rover_distance': analysis['distances']['rover_distance'],
                            'satellite_distance': analysis['distances']['satellite_distance'],
                            'rover_violated': rover_violated,
                            'sat_violated': sat_violated
                        }
                    except:
                        feasible = True
                        constraint_info = {}

                    sols.append({
                        "x": x,
                        "fitness": [float(f[0]), float(f[1])],
                        "feasible": feasible,
                        "constraint_info": constraint_info,
                        "crowding_distance": 0.0,
                        "pareto_optimal": False
                    })

                    if len(sols) >= 20:
                        break

        # Rendit dhe ruaj
        sols.sort(key=lambda s: s["fitness"][0])
        self.best_solutions = sols

        if verbose and sols:
            pareto_count = len([s for s in sols if s['pareto_optimal']])
            print(f"\nFound {pareto_count} Pareto-optimal solutions.")
            print(f"Total solutions: {len(sols)}")
            
            if pareto_count > 0:
                pareto_solutions = [s for s in sols if s['pareto_optimal']]
                print(f"Best J1: {min(s['fitness'][0] for s in pareto_solutions):.6f}")
                print(f"Best J2: {min(s['fitness'][1] for s in pareto_solutions):.6f}")

        # Ruaj në file
        self.save_solutions_to_json(sols)

        # Vizualizo
        self.plot_fitness_progress(fitness_history)
        self.plot_pareto_front()

        print("\n✅ Optimization complete! Solutions saved to 'solutions.json'.")
        
        return sols

    def plot_fitness_progress(self, fitness_history):
        """Plot fitness progress over generations."""
        if not fitness_history["generation"]:
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot J1 progress
        axes[0].plot(fitness_history["generation"], fitness_history["best_J1"], 'b-', label='Best J1', linewidth=2)
        axes[0].plot(fitness_history["generation"], fitness_history["avg_J1"], 'b--', label='Avg J1', alpha=0.7)
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('J1 - Communication Cost')
        axes[0].set_title('J1 Progress Over Generations')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot J2 progress
        axes[1].plot(fitness_history["generation"], fitness_history["best_J2"], 'r-', label='Best J2', linewidth=2)
        axes[1].plot(fitness_history["generation"], fitness_history["avg_J2"], 'r--', label='Avg J2', alpha=0.7)
        axes[1].set_xlabel('Generation')
        axes[1].set_ylabel('J2 - Infrastructure Cost')
        axes[1].set_title('J2 Progress Over Generations')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_pareto_front(self):
        """Plot Pareto front."""
        if not self.best_solutions:
            print("No solutions to plot.")
            return

        # Kategorizo
        pareto_feasible = [s for s in self.best_solutions if s["feasible"] and s.get("pareto_optimal", False)]
        non_pareto_feasible = [s for s in self.best_solutions if s["feasible"] and not s.get("pareto_optimal", False)]
        infeasible = [s for s in self.best_solutions if not s["feasible"]]
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Plot
        if infeasible:
            infeas_f1 = [s["fitness"][0] for s in infeasible]
            infeas_f2 = [s["fitness"][1] for s in infeasible]
            ax.scatter(infeas_f1, infeas_f2, c="red", alpha=0.2, label=f'Infeasible ({len(infeasible)})', marker='x')
        
        if non_pareto_feasible:
            non_pareto_f1 = [s["fitness"][0] for s in non_pareto_feasible]
            non_pareto_f2 = [s["fitness"][1] for s in non_pareto_feasible]
            ax.scatter(non_pareto_f1, non_pareto_f2, c="blue", alpha=0.4, label=f'Feasible non-Pareto ({len(non_pareto_feasible)})')
        
        if pareto_feasible:
            pareto_f1 = [s["fitness"][0] for s in pareto_feasible]
            pareto_f2 = [s["fitness"][1] for s in pareto_feasible]
            
            ax.scatter(pareto_f1, pareto_f2, c="green", edgecolor="black", s=80, label=f'Pareto-optimal ({len(pareto_feasible)})', zorder=5)
            
            if len(pareto_f1) > 1:
                sorted_indices = np.argsort(pareto_f1)
                ax.plot([pareto_f1[i] for i in sorted_indices], 
                       [pareto_f2[i] for i in sorted_indices], 'g--', alpha=0.7, linewidth=2, zorder=4)
        
        ax.set_xlabel('J1 – Communication Cost')
        ax.set_ylabel('J2 – Infrastructure Cost')
        ax.set_title(f'Pareto Front\nPopulation: {self.population}, Generations: {self.generations}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def save_solutions_to_json(self, solutions, filename="solutions.json"):
        """Save solutions to JSON file."""
        try:
            solutions_json = []
            for sol in solutions:
                solutions_json.append({
                    "fitness": sol["fitness"],
                    "feasible": sol["feasible"],
                    "crowding_distance": sol["crowding_distance"],
                    "constraint_info": sol["constraint_info"],
                    "x": sol["x"].tolist()
                })
            
            with open(filename, "w") as f:
                json.dump({
                    "timestamp": datetime.utcnow().isoformat(),
                    "optimizer": "QuantumCommunicationsOptimizer",
                    "parameters": {
                        "population": self.population,
                        "generations": self.generations,
                        "seed": self.seed
                    },
                    "statistics": {
                        "total_solutions": len(solutions),
                        "pareto_optimal": len([s for s in solutions if s["pareto_optimal"]]),
                        "feasible": len([s for s in solutions if s["feasible"]]),
                        "submitted": len(solutions_json)
                    },
                    "solutions": solutions_json
                }, f, indent=2)

            print(f"\n✓ Solutions saved to {filename}")
        except Exception as e:
            print(f"✗ Error saving solutions: {e}")

    def analyze_solutions(self, show_top=10):
        """Analyze and display solutions."""
        if not self.best_solutions:
            print("No solutions available. Run optimize() first.")
            return

        print(f"\n=== SOLUTION ANALYSIS (showing top {show_top}) ===")
        

        feasible = [s for s in self.best_solutions if s["feasible"]]
        infeasible = [s for s in self.best_solutions if not s["feasible"]]
        
        print(f"Total solutions: {len(self.best_solutions)}")
        print(f"Feasible solutions: {len(feasible)}")
        print(f"Infeasible solutions: {len(infeasible)}")
        
        if feasible:
            print("\n--- TOP FEASIBLE SOLUTIONS ---")
            for i, sol in enumerate(feasible[:show_top]):
                pareto_label = "(Pareto-optimal)" if sol.get("pareto_optimal", False) else ""
                print(f"\nSolution #{i+1} {pareto_label}")
                print(f"  J1 (Communication cost): {sol['fitness'][0]:.6f}")
                print(f"  J2 (Infrastructure cost): {sol['fitness'][1]:.6f}")
                print(f"  Crowding distance: {sol['crowding_distance']:.6f}")
                
                if 'constraint_info' in sol and sol['constraint_info']:
                    info = sol['constraint_info']
                    print(f"  Rover distance: {info.get('rover_distance', 'N/A'):.2f} km")
                    print(f"  Satellite distance: {info.get('satellite_distance', 'N/A'):.2f} km")

    @staticmethod
    def run_test():
        """Run a quick test."""
        print("Running quick test...")

        try:

            optimizer = QuantumCommunicationsOptimizer(
                population_size=8,
                generations=5,
                random_seed=42
            )

            solutions = optimizer.optimize(verbose=True)

            if solutions:
                print(f"\n✓ Test successful! Found {len(solutions)} solutions.")

                return True
            else:
                print("✗ Test failed: No solutions found.")
                return False

        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            return False


# Funksion për demo
def run_optimization_demo(population_size=20, generations=30):
    """Run a demo optimization."""
    print("=" * 60)
    print("QUANTUM COMMUNICATIONS OPTIMIZATION DEMO")
    print("=" * 60)
    
    optimizer = QuantumCommunicationsOptimizer(
        population_size=population_size,
        generations=generations,
        random_seed=42
    )
    
    solutions = optimizer.optimize(verbose=True)
    
    if solutions:
        optimizer.analyze_solutions(show_top=10)
    
    return optimizer
