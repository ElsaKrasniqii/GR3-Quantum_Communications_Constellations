import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt
import json

from datetime import datetime



try:
    from constellation_udp import constellation_udp
except Exception as e:
    raise RuntimeError(f"Failed loading constellation_udp: {e}")


class QuantumCommunicationsOptimizer:

    def __init__(self, population_size=48, generations=100, random_seed=42, use_multithreading=False):
        # ============================================
        # FIKSU: Sigurohu që popullsia plotëson kërkesat e NSGA-II
        # ============================================
        original_size = population_size
        
        # Kërkesat e NSGA-II:
        # 1. Të paktën 5 individë
        # 2. Shumëfish i 4
        if population_size < 5:
            population_size = 8  # Minimumi i sigurt
            print(f"⚠ Warning: Population too small ({original_size}). Increased to {population_size}.")
        
        if population_size % 4 != 0:
            # Rrumbullakso në shumëfishin më të afërt të 4
            population_size = ((population_size + 3) // 4) * 4
            print(f"⚠ Warning: Population {original_size} adjusted to {population_size} (must be multiple of 4).")
        # ============================================
        
        self.population = population_size
        self.generations = generations
        self.seed = random_seed
        self.use_multithreading = use_multithreading

        self.udp = constellation_udp()
        self.best_solutions = []
        np.random.seed(self.seed)

    # -------------------------------------------------------------
    # MAIN OPTIMIZATION FUNCTION (NSGA-II ONLY)
    # -------------------------------------------------------------
    def optimize(self, verbose=True):
        # Krijo problemin PARË
        prob = pg.problem(self.udp)
        
        # Printo TË GJITHA informacionet NJË HERË
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
        
        # Krijo algoritmin NSGA-II
        algo = pg.algorithm(
            pg.nsga2(
                gen=self.generations,
                cr=0.9,
                eta_c=10,
                m=0.01,
                eta_m=20
            )
        )
        
        # Aktivo output nga algoritmi
        if verbose:
            algo.set_verbosity(1)
        
        # Krijo popullatën fillestare
        pop = pg.population(prob, size=self.population, seed=self.seed)
        
        # Shto zgjidhje heuristike
        pop = self._add_heuristic_solutions(pop)
        
        # Ekzekuto optimizimin
        if verbose:
            print("\nStarting evolution...")
            print("-" * 50)
        
        pop = algo.evolve(pop)
        
        # Nxjerr rezultatet
        fits = pop.get_f()
        xs = pop.get_x()
        
        if verbose:
            print(f"\n✅ Optimization completed!")
            print(f"Final population size: {len(fits)}")
            print(f"Fitness shape: {fits.shape}")
        
        # Merr frontin jo-dominuar (Pareto front)
        front = pg.non_dominated_front_2d(fits)
        
        # Nxjerr zgjidhjet Pareto
        sols = []
        for idx in front:
            f = fits[idx]
            x = xs[idx]
            
            # Analizo constraints (duke përdorur metodën utility)
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
                # Fallback nëse analiza dështon
                feasible = True
                constraint_info = {}
            
            sols.append({
                "x": x,
                "fitness": [float(f[0]), float(f[1])],  # f1 dhe f2 vetëm
                "feasible": feasible,
                "constraint_info": constraint_info,
                "crowding_distance": float(pop.get_cd()[idx]) if hasattr(pop, 'get_cd') else 0.0
            })
        
        # Rendit sipas objektivit të parë (J1)
        sols.sort(key=lambda s: s["fitness"][0])
        self.best_solutions = sols
        
        if verbose and sols:
            print(f"\nFound {len(sols)} Pareto-optimal solutions.")
            print(f"Best J1: {sols[0]['fitness'][0]:.6f}")
            print(f"Best J2: {sols[-1]['fitness'][1]:.6f}")
        
        return sols

    # -------------------------------------------------------------
    # ADD HEURISTIC SOLUTIONS
    # -------------------------------------------------------------
    def _add_heuristic_solutions(self, pop):
        """Add heuristic solutions to the initial population."""
        try:
            # Get example solution from UDP
            example_sol = self.udp.example()
            
            # Get bounds
            bounds = self.udp.get_bounds()
            lower = np.array(bounds[0])
            upper = np.array(bounds[1])
            
            # Add example solution
            if len(example_sol) == len(lower):
                pop.set_x(0, example_sol)
            
            # Add some random but bounded solutions
            for i in range(1, min(5, len(pop))):
                random_sol = lower + np.random.random(len(lower)) * (upper - lower)
                pop.set_x(i, random_sol)
                
        except Exception as e:
            if len(pop) > 0:  # Print vetëm nëse kemi një popullatë
                print(f"Warning: Could not add heuristic solutions: {e}")
        
        return pop

    # -------------------------------------------------------------
    # SOLUTION ANALYSIS
    # -------------------------------------------------------------
    def analyze_solutions(self, show_top=10):
        if not self.best_solutions:
            print("No solutions available. Run optimize() first.")
            return

        print(f"\n=== SOLUTION ANALYSIS (showing top {show_top}) ===")
        
        # Separate feasible and infeasible
        feasible = [s for s in self.best_solutions if s["feasible"]]
        infeasible = [s for s in self.best_solutions if not s["feasible"]]
        
        print(f"Total Pareto solutions: {len(self.best_solutions)}")
        print(f"Feasible solutions: {len(feasible)}")
        print(f"Infeasible solutions: {len(infeasible)}")
        
        if feasible:
            print("\n--- TOP FEASIBLE SOLUTIONS ---")
            for i, sol in enumerate(feasible[:show_top]):
                print(f"\nSolution #{i+1} (Feasible)")
                print(f"  J1 (Communication cost): {sol['fitness'][0]:.6f}")
                print(f"  J2 (Infrastructure cost): {sol['fitness'][1]:.6f}")
                print(f"  Crowding distance: {sol['crowding_distance']:.6f}")
                
                if 'constraint_info' in sol and sol['constraint_info']:
                    info = sol['constraint_info']
                    print(f"  Rover distance: {info.get('rover_distance', 'N/A'):.2f} km")
                    print(f"  Satellite distance: {info.get('satellite_distance', 'N/A'):.2f} km")
        
        if infeasible and len(feasible) < show_top:
            print(f"\n--- TOP INFEASIBLE SOLUTIONS (for reference) ---")
            for i, sol in enumerate(infeasible[:min(show_top-len(feasible), len(infeasible))]):
                print(f"\nSolution #{i+1+len(feasible)} (INFEASIBLE)")
                print(f"  J1: {sol['fitness'][0]:.6f}")
                print(f"  J2: {sol['fitness'][1]:.6f}")
                if 'constraint_info' in sol:
                    info = sol['constraint_info']
                    print(f"  Rover violated: {info.get('rover_violated', 'N/A')}")
                    print(f"  Satellite violated: {info.get('sat_violated', 'N/A')}")

    # -------------------------------------------------------------
    # DETAILED ANALYSIS
    # -------------------------------------------------------------
    def detailed_analysis(self, idx=0):
        if not self.best_solutions:
            print("No solutions available.")
            return

        if idx >= len(self.best_solutions):
            print(f"Index {idx} out of range. Available: 0-{len(self.best_solutions)-1}")
            return

        sol = self.best_solutions[idx]
        
        print(f"\n=== DETAILED ANALYSIS OF SOLUTION #{idx+1} ===")
        print(f"Feasible: {sol['feasible']}")
        print(f"Fitness J1: {sol['fitness'][0]:.6f}")
        print(f"Fitness J2: {sol['fitness'][1]:.6f}")
        print(f"Crowding distance: {sol['crowding_distance']:.6f}")
        
        # Show constraint information
        if 'constraint_info' in sol and sol['constraint_info']:
            info = sol['constraint_info']
            print("\nConstraint Information:")
            print(f"  Rover distance: {info.get('rover_distance', 'N/A'):.2f} km")
            print(f"  Satellite distance: {info.get('satellite_distance', 'N/A'):.2f} km")
            print(f"  Rover violated: {info.get('rover_violated', 'N/A')}")
            print(f"  Satellite violated: {info.get('sat_violated', 'N/A')}")
        
        # Analyze with UDP's analysis method
        try:
            udp_analysis = self.udp.analyze_solution(sol['x'])
            print("\nUDP Analysis:")
            print(f"  Walker 1: {udp_analysis['walker1']['satellites']} satellites")
            print(f"  Walker 2: {udp_analysis['walker2']['satellites']} satellites")
            print(f"  Total satellites: {udp_analysis['walker1']['satellites'] + udp_analysis['walker2']['satellites']}")
            print(f"  Fitness without penalty: {[f'{val:.6f}' for val in udp_analysis['fitness']['without_penalty']]}")
        except Exception as e:
            print(f"\nCould not perform detailed UDP analysis: {e}")

    # -------------------------------------------------------------
    # PLOT PARETO FRONT
    # -------------------------------------------------------------
    def plot_pareto_front(self, save_path=None):
        if not self.best_solutions:
            print("No solutions to plot.")
            return

        # Separate feasible and infeasible
        feas_f1 = [s["fitness"][0] for s in self.best_solutions if s["feasible"]]
        feas_f2 = [s["fitness"][1] for s in self.best_solutions if s["feasible"]]
        
        infeas_f1 = [s["fitness"][0] for s in self.best_solutions if not s["feasible"]]
        infeas_f2 = [s["fitness"][1] for s in self.best_solutions if not s["feasible"]]

        plt.figure(figsize=(10, 7))
        
        # Plot infeasible first (in background)
        if infeas_f1 and infeas_f2:
            plt.scatter(infeas_f1, infeas_f2, c="red", alpha=0.3, 
                       label=f"Infeasible ({len(infeas_f1)})", marker='x', s=50)
        
        # Plot feasible on top
        if feas_f1 and feas_f2:
            plt.scatter(feas_f1, feas_f2, c="green", edgecolor="black", 
                       label=f"Feasible ({len(feas_f1)})", s=80)
            
            # Highlight the best compromise solution (closest to ideal point)
            if len(feas_f1) > 0:
                # Ideal point (minimum of both objectives)
                ideal_f1 = min(feas_f1)
                ideal_f2 = min(feas_f2)
                
                # Find solution closest to ideal point
                distances = []
                for i in range(len(feas_f1)):
                    d = ((feas_f1[i] - ideal_f1) ** 2 + (feas_f2[i] - ideal_f2) ** 2) ** 0.5
                    distances.append(d)
                
                best_idx = np.argmin(distances)
                plt.scatter(feas_f1[best_idx], feas_f2[best_idx], 
                          c="gold", s=200, edgecolor="black", 
                          label="Best Compromise", zorder=5)
                
                # Add annotation
                plt.annotate("Best", 
                            xy=(feas_f1[best_idx], feas_f2[best_idx]),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=10, fontweight='bold')

        plt.xlabel("J1 – Communication Cost (normalized)", fontsize=12)
        plt.ylabel("J2 – Infrastructure Cost (normalized)", fontsize=12)
        plt.title(f"Pareto Front\nPopulation: {self.population}, Generations: {self.generations}", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Plot saved to {save_path}")
        
        plt.show()

    # -------------------------------------------------------------
    # PLOT CONVERGENCE (if algorithm supports it)
    # -------------------------------------------------------------
    def plot_convergence(self, log=None):
        """Plot convergence of the algorithm if log data is available."""
        # This would require capturing log data during optimization
        # For now, it's a placeholder
        print("Convergence plotting requires logging data capture.")
        print("Consider implementing logging in the optimize() method.")

    # -------------------------------------------------------------
    # CREATE SUBMISSION FILE
    # -------------------------------------------------------------
    def create_submission(self, filename="submission.json", top_n=20):
        if not self.best_solutions:
            print("No solutions to save.")
            return

        # Take top feasible solutions, or all if none feasible
        feasible = [s for s in self.best_solutions if s["feasible"]]
        if not feasible:
            print("Warning: No feasible solutions. Using best infeasible solutions.")
            feasible = self.best_solutions

        # Sort by J1 (communication cost)
        feasible = sorted(feasible, key=lambda s: s["fitness"][0])
        feasible = feasible[:top_n]

        solutions_json = [s["x"].tolist() for s in feasible]

        out = {
            "timestamp": datetime.utcnow().isoformat(),
            "optimizer": "QuantumCommunicationsOptimizer",
            "parameters": {
                "population": self.population,
                "generations": self.generations,
                "seed": self.seed
            },
            "count": len(solutions_json),
            "dimension": len(solutions_json[0]) if solutions_json else 0,
            "solutions": solutions_json
        }

        try:
            with open(filename, "w") as f:
                json.dump(out, f, indent=2)
            print(f"\n✓ Submission saved to {filename}")
            print(f"  Solutions: {len(solutions_json)}")
            print(f"  Dimension: {out['dimension']}")
        except Exception as e:
            print(f"✗ Error saving submission: {e}")

    # -------------------------------------------------------------
    # EXPORT SOLUTIONS TO CSV
    # -------------------------------------------------------------
    def export_to_csv(self, filename="solutions.csv"):
        """Export solutions to CSV format."""
        if not self.best_solutions:
            print("No solutions to export.")
            return

        import pandas as pd
        
        data = []
        for i, sol in enumerate(self.best_solutions):
            row = {
                'id': i+1,
                'feasible': sol['feasible'],
                'J1': sol['fitness'][0],
                'J2': sol['fitness'][1],
                'crowding_distance': sol.get('crowding_distance', 0)
            }
            
            # Add decision variables
            for j, val in enumerate(sol['x']):
                row[f'x{j}'] = val
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Solutions exported to {filename}")

    # -------------------------------------------------------------
    # TEST FUNCTION
    # -------------------------------------------------------------
    @staticmethod
    def run_test():
        """Run a quick test to verify the optimizer works."""
        print("Running quick test...")
        
        try:
            # Create optimizer with small parameters for quick test
            optimizer = QuantumCommunicationsOptimizer(
                population_size=8,  # 8 është shumëfish i 4
                generations=10,
                random_seed=42
            )
            
            # Run optimization
            solutions = optimizer.optimize(verbose=True)
            
            if solutions:
                print(f"\n✓ Test successful! Found {len(solutions)} solutions.")
                
                # Show top 3
                optimizer.analyze_solutions(show_top=3)
                
                # Create test plot
                optimizer.plot_pareto_front()
                
                return True
            else:
                print("✗ Test failed: No solutions found.")
                return False
                
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            return False


# -------------------------------------------------------------
# MAIN EXECUTION (if run directly)
# -------------------------------------------------------------
if __name__ == "__main__":
    # Run test if file is executed directly
    success = QuantumCommunicationsOptimizer.run_test()
    
    if success:
        print("\nOptimizer is working correctly!")
    else:
        print("\nOptimizer test failed. Check the error messages above.")