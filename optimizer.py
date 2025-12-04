import matplotlib.pyplot as plt
import json
from datetime import datetime
import warnings
import numpy as np
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
   CONSTELLATION_UDP_AVAILABLE = True
except ImportError as e:
   CONSTELLATION_UDP_AVAILABLE = False
   print(f"Constellation UDP not available: {e}")


class QuantumCommunicationsOptimizer:
   """Main optimizer class for the quantum communications constellation problem"""
  
   def __init__(self, population_size=50, generations=100,
                use_multithreading=False, random_seed=42):
       """
       Initialize the optimizer
      
       Parameters:
       -----------
       population_size : int
           Size of the population for NSGA-II
       generations : int
           Number of generations to evolve
       use_multithreading : bool
           Enable parallel evaluation if available
       random_seed : int
           Seed for reproducibility
       """
      
       if not CONSTELLATION_UDP_AVAILABLE:
           raise ImportError("constellation_udp module not found")
          
       self.udp = constellation_udp()
       self.population_size = population_size
       self.generations = generations
       self.use_multithreading = use_multithreading
       self.random_seed = random_seed
       self.best_solutions = []
       self.history = {
           'generations': [],
           'best_fitness': [],
           'feasible_count': []
       }
      
       # Set random seed for reproducibility
       np.random.seed(random_seed)
      
   def optimize(self, verbose=True):
       """Execute multi-objective optimization"""
       if not PYGMO_AVAILABLE:
           print("Pygmo not available. Cannot run optimization.")
           print("Install with: pip install pygmo")
           return []
          
       print("="*60)
       print("Quantum Communications Constellation Optimization")
       print("="*60)
       print(f"Population size: {self.population_size}")
       print(f"Generations: {self.generations}")
       print(f"Random seed: {self.random_seed}")
       print("-"*60)
      
       try:
           # Create problem
           prob = pg.problem(self.udp)
          
           # Configure algorithm
           if self.use_multithreading:
               # Enable parallel evaluation if available
               try:
                   algo = pg.algorithm(pg.nsga2(
                       gen=self.generations,
                       cr=0.9,  # Crossover probability
                       eta_c=10.0,  # Crossover distribution index
                       m=0.01,  # Mutation probability
                       eta_m=50.0  # Mutation distribution index
                   ))
                   # Enable parallelization
                   algo.set_verbosity(1)
               except:
                   # Fallback to non-parallel version
                   algo = pg.algorithm(pg.nsga2(gen=self.generations))
           else:
               algo = pg.algorithm(pg.nsga2(
                   gen=self.generations,
                   cr=0.9,
                   eta_c=10.0,
                   m=0.01,
                   eta_m=50.0
               ))
          
           # Create population with diverse initial solutions
           print("Generating initial population...")
           pop = pg.population(prob, size=self.population_size, seed=self.random_seed)
          
           # Add some heuristic initial solutions
           self._add_heuristic_solutions(pop)
          
           # Track initial statistics
           self._update_history(pop, generation=0)
          
           # Execute optimization
           print(f"\nRunning {self.generations} generations...")
          
           # Callback for progress tracking
           def log_callback(algo, pop):
               gen = algo.get_generation()
               if verbose and gen % 10 == 0:
                   feasible = sum(1 for i in range(pop.size())
                                if all(c <= 0 for c in pop.get_f()[i, 2:]))
                   print(f"Generation {gen:3d}: {feasible}/{pop.size()} feasible solutions")
          
           # Set callback if available
           try:
               algo.set_callback(log_callback)
           except:
               pass
          
           # Evolve population
           pop = algo.evolve(pop)
          
           # Extract Pareto-optimal solutions
           self.best_solutions = self._extract_pareto_front(pop)
          
           print(f"\nOptimization completed!")
           print(f"Found {len(self.best_solutions)} Pareto-optimal solutions")
          
           return self.best_solutions
          
       except Exception as e:
           print(f"Optimization failed with error: {e}")
           import traceback
           traceback.print_exc()
           return []
  
   def _add_heuristic_solutions(self, pop):
       """Add heuristic initial solutions to improve convergence"""
      
       # Get population size
       pop_size = pop.size()
      
       # Example solution from UDP
       try:
           example_x = self.udp.example()
           if len(example_x) == 20:  # Ensure correct length
               pop.set_x(0, example_x)
       except:
           pass
      
       # Add some LEO solutions (common for communications)
       leo_solutions = [
           # Low Earth Orbit with different inclinations
           [6871, 0.001, np.radians(53), 0, 60,  # Walker 1: ~500km, 53° inclination
            7071, 0.001, np.radians(97), 0, 40,  # Walker 2: ~700km, sun-synchronous
            6, 4, 1, 8, 3, 2,  # Constellation parameters
            10, 25, 40, 55],   # Rover indices
           
           # Medium Earth Orbit
           [12000, 0.01, np.radians(55), np.radians(90), 75,
            14000, 0.01, np.radians(65), np.radians(90), 50,
            4, 3, 1, 6, 2, 1,
            5, 20, 35, 50],
           
           # Mixed altitudes
           [7500, 0.001, np.radians(86), 0, 80,
            10500, 0.005, np.radians(70), np.radians(180), 30,
            5, 5, 2, 7, 4, 1,
            15, 30, 45, 60]
       ]



       # Add heuristic solutions to population
       for i, sol in enumerate(leo_solutions, start=1):
           if i < pop_size and len(sol) == 20:
               try:
                   pop.set_x(i, sol)
               except:
                   pass
  
   def _update_history(self, pop, generation):
       """Update optimization history"""
       fits = pop.get_f()
      
       # Count feasible solutions
       feasible = sum(1 for i in range(pop.size())
                     if all(c <= 0 for c in fits[i, 2:]))
      
       # Get best fitness values (lower is better)
       if len(fits) > 0:
           best_idx = np.argmin(fits[:, 0] + fits[:, 1])  # Simple aggregation
           best_fit = fits[best_idx, :2]
       else:
           best_fit = [np.inf, np.inf]
      
       self.history['generations'].append(generation)
       self.history['best_fitness'].append(best_fit)
       self.history['feasible_count'].append(feasible)
  
   def _extract_pareto_front(self, pop):
       """Extract Pareto front from population"""
       try:
           fits = pop.get_f()
          
           # Use 2D Pareto front (first two objectives)
           best_idx = pg.non_dominated_front_2d(fits[:, :2])
          
           solutions = []
           for idx in best_idx:
               x = pop.get_x()[idx]
               f = fits[idx]
              
               # Calculate constraint violations
               constraint_violations = [f[2], f[3]]
               feasible = all(c <= 0 for c in constraint_violations)
              
               # Decode solution for readability
               decoded = self._decode_solution(x)
              
               solutions.append({
                   'x': x,
                   'fitness': f[:2],  # First two objectives only
                   'constraints': constraint_violations,
                   'feasible': feasible,
                   'decoded': decoded
               })
          
           # Sort by first objective (communication cost)
           solutions.sort(key=lambda s: s['fitness'][0])
          
           return solutions
          
       except Exception as e:
           print(f"Error extracting Pareto front: {e}")
           return []
  
   def _decode_solution(self, x):
       """Decode chromosome into human-readable format"""
       try:
           a1, e1, i1, w1, eta1, a2, e2, i2, w2, eta2, S1, P1, F1, S2, P2, F2, r1, r2, r3, r4 = x
          
           return {
               'walker1': {
                   'semi_major_axis_km': float(a1),
                   'eccentricity': float(e1),
                   'inclination_deg': float(np.degrees(i1)),
                   'arg_perigee_deg': float(np.degrees(w1)),
                   'quality_eta': float(eta1),
                   'satellites_per_plane': int(S1),
                   'planes': int(P1),
                   'phasing': int(F1),
                   'total_satellites': int(S1 * P1)
               },
               'walker2': {
                   'semi_major_axis_km': float(a2),
                   'eccentricity': float(e2),
                   'inclination_deg': float(np.degrees(i2)),
                   'arg_perigee_deg': float(np.degrees(w2)),
                   'quality_eta': float(eta2),
                   'satellites_per_plane': int(S2),
                   'planes': int(P2),
                   'phasing': int(F2),
                   'total_satellites': int(S2 * P2)
               },
               'rovers': {
                   'indices': [int(r1), int(r2), int(r3), int(r4)],
                   'total_satellites': int(S1 * P1 + S2 * P2),
                   'total_eta': float(eta1 * S1 * P1 + eta2 * S2 * P2)
               }
           }
       except:
           return {}
  
   def analyze_solutions(self, show_top=10):
       """Analyze and display found solutions"""
       if not self.best_solutions:
           print("No solutions to analyze. Run optimize() first.")
           return
      
       feasible_solutions = [s for s in self.best_solutions if s['feasible']]
       infeasible_solutions = [s for s in self.best_solutions if not s['feasible']]
      
       print("\n" + "="*60)
       print("SOLUTION ANALYSIS")
       print("="*60)
       print(f"Total Pareto-optimal solutions: {len(self.best_solutions)}")
       print(f"Feasible solutions: {len(feasible_solutions)} ({len(feasible_solutions)/len(self.best_solutions)*100:.1f}%)")
       print(f"Infeasible solutions: {len(infeasible_solutions)}")
      
       if feasible_solutions:
           print(f"\nTop {min(show_top, len(feasible_solutions))} feasible solutions:")
           for i, sol in enumerate(feasible_solutions[:show_top]):
               print(f"\nSolution {i+1} (Rank {i+1}):")
               print(f"  Communication cost (J1): {sol['fitness'][0]:.6f}")
               print(f"  Infrastructure cost (J2): {sol['fitness'][1]:.6f}")
               print(f"  Total cost (J1+J2): {sum(sol['fitness']):.6f}")
               print(f"  Rover constraint violation: {sol['constraints'][0]:.6f}")
               print(f"  Satellite constraint violation: {sol['constraints'][1]:.6f}")
              
               if 'decoded' in sol and sol['decoded']:
                   w1 = sol['decoded']['walker1']
                   w2 = sol['decoded']['walker2']
                   print(f"  Total satellites: {w1['total_satellites'] + w2['total_satellites']}")
                   print(f"  Total eta: {sol['decoded']['rovers']['total_eta']:.1f}")
      
       # Show statistics
       if feasible_solutions:
           f1_vals = [s['fitness'][0] for s in feasible_solutions]
           f2_vals = [s['fitness'][1] for s in feasible_solutions]
          
           print(f"\nStatistics for feasible solutions:")
           print(f"  J1 range: [{min(f1_vals):.6f}, {max(f1_vals):.6f}]")
           print(f"  J2 range: [{min(f2_vals):.6f}, {max(f2_vals):.6f}]")
           print(f"  Average J1: {np.mean(f1_vals):.6f}")
           print(f"  Average J2: {np.mean(f2_vals):.6f}")
  
   def plot_pareto_front(self, show_dominated=True, save_path=None):
       """Visualize the Pareto front"""
       if not self.best_solutions:
           print("No solutions to plot.")
           return
      
       plt.figure(figsize=(12, 10))
      
       # Separate feasible and infeasible
       feasible = [s for s in self.best_solutions if s['feasible']]
       infeasible = [s for s in self.best_solutions if not s['feasible']]
      
       # Plot dominated solutions if requested
       if show_dominated and len(feasible) > 1:
           # Create convex hull for visualization
           f1_feas = [s['fitness'][0] for s in feasible]
           f2_feas = [s['fitness'][1] for s in feasible]
          
           # Sort by first objective
           sorted_points = sorted(zip(f1_feas, f2_feas), key=lambda x: x[0])
           f1_sorted, f2_sorted = zip(*sorted_points)
          
           # Plot dominated region (lower is better)
           plt.fill_between(f1_sorted, 0, f2_sorted,
                          alpha=0.1, color='green', label='Dominated region')
      
       # Plot points
       if feasible:
           f1_feas = [s['fitness'][0] for s in feasible]
           f2_feas = [s['fitness'][1] for s in feasible]
          
           # Size based on total cost
           sizes = [100 * (2 - (f1 + f2) / max(f1_feas[i] + f2_feas[i] for i in range(len(f1_feas))))
                   for i, (f1, f2) in enumerate(zip(f1_feas, f2_feas))]
          
           scatter = plt.scatter(f1_feas, f2_feas, c='green', s=sizes,
                               label='Feasible', alpha=0.8, edgecolors='black', linewidth=0.5)
          
           # Annotate best solutions
           for i, (f1, f2) in enumerate(zip(f1_feas[:3], f2_feas[:3])):
               plt.annotate(f'{i+1}', (f1, f2),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=9, fontweight='bold')
      
       if infeasible:
           f1_infeas = [s['fitness'][0] for s in infeasible]
           f2_infeas = [s['fitness'][1] for s in infeasible]
           plt.scatter(f1_infeas, f2_infeas, c='red', s=20,
                      label='Infeasible', alpha=0.5, marker='x')


 # Reference point
       ref_point = [1.2, 1.4]
       plt.scatter(ref_point[0], ref_point[1], c='blue', s=100,
                  marker='*', label='Reference point', edgecolors='black')
      
       plt.xlabel('J1 - Average Communication Cost (normalized)', fontsize=12)
       plt.ylabel('J2 - Infrastructure Cost (normalized)', fontsize=12)
       plt.title('Pareto Front - Quantum Communications Constellations', fontsize=14, fontweight='bold')
       plt.legend(fontsize=11)
       plt.grid(True, alpha=0.3)
       plt.tight_layout()
      
       if save_path:
           plt.savefig(save_path, dpi=300, bbox_inches='tight')
           print(f"Plot saved to: {save_path}")
      
       plt.show()
  
   def plot_optimization_history(self):
       """Plot optimization progress over generations"""
       if not self.history['generations']:
           print("No optimization history available.")
           return
      
       fig, axes = plt.subplots(2, 2, figsize=(14, 10))
      
       # Plot 1: Best fitness over generations
       gens = self.history['generations']
       best_f1 = [f[0] for f in self.history['best_fitness']]
       best_f2 = [f[1] for f in self.history['best_fitness']]
      
       axes[0, 0].plot(gens, best_f1, 'b-', linewidth=2, label='J1 (Communication)')
       axes[0, 0].plot(gens, best_f2, 'r-', linewidth=2, label='J2 (Infrastructure)')
       axes[0, 0].set_xlabel('Generation')
       axes[0, 0].set_ylabel('Best Fitness Value')
       axes[0, 0].set_title('Best Fitness Over Generations')
       axes[0, 0].legend()
       axes[0, 0].grid(True, alpha=0.3)
      
       # Plot 2: Feasible solutions count
       axes[0, 1].plot(gens, self.history['feasible_count'], 'g-', linewidth=2)
       axes[0, 1].fill_between(gens, 0, self.history['feasible_count'], alpha=0.3, color='green')
       axes[0, 1].set_xlabel('Generation')
       axes[0, 1].set_ylabel('Feasible Solutions')
       axes[0, 1].set_title('Feasible Solutions Over Generations')
       axes[0, 1].grid(True, alpha=0.3)
      
       # Plot 3: Combined cost
       total_cost = [f1 + f2 for f1, f2 in self.history['best_fitness']]
       axes[1, 0].plot(gens, total_cost, 'purple', linewidth=2)
       axes[1, 0].set_xlabel('Generation')
       axes[1, 0].set_ylabel('Total Cost (J1 + J2)')
       axes[1, 0].set_title('Total Cost Over Generations')
       axes[1, 0].grid(True, alpha=0.3)
      
       # Plot 4: Pareto front animation (last generation)
       if self.best_solutions:
           feasible = [s for s in self.best_solutions if s['feasible']]
           if feasible:
               f1 = [s['fitness'][0] for s in feasible]
               f2 = [s['fitness'][1] for s in feasible]
               axes[1, 1].scatter(f1, f2, c='green', s=50, alpha=0.7)
               axes[1, 1].set_xlabel('J1')
               axes[1, 1].set_ylabel('J2')
               axes[1, 1].set_title('Final Pareto Front')
               axes[1, 1].grid(True, alpha=0.3)
      
       plt.tight_layout()
       plt.show()
  
   def create_submission(self, filename="submission.json", top_n=20):
       """Create submission file"""
       if not self.best_solutions:
           print("No solutions to submit. Run optimize() first.")
           return
      
       # Get feasible solutions sorted by total cost
       feasible_solutions = [s for s in self.best_solutions if s['feasible']]
       feasible_solutions.sort(key=lambda s: s['fitness'][0] + s['fitness'][1])
      
       # Take top N solutions
       submission_solutions = []
       for sol in feasible_solutions[:top_n]:
           # Ensure solution is in correct format
           x = sol['x']
           if len(x) == 20:  # Expected length
               submission_solutions.append(x.tolist() if hasattr(x, 'tolist') else list(x))
      
       # If no feasible solutions, use best infeasible
       if not submission_solutions and self.best_solutions:
           for sol in self.best_solutions[:top_n]:
               x = sol['x']
               if len(x) == 20:
                   submission_solutions.append(x.tolist() if hasattr(x, 'tolist') else list(x))
      
       if not submission_solutions:
           print("ERROR: No valid solutions found for submission.")
           return
      
       # Create submission data
       submission_data = {
           "problem": "quantum-communications-constellations",
           "challenge": "spoc-2-quantum-communications-constellations",
           "timestamp": datetime.now().isoformat(),
           "optimizer": {
               "name": "QuantumCommunicationsOptimizer",
               "population_size": self.population_size,
               "generations": self.generations,
               "random_seed": self.random_seed
           },
           "solutions": submission_solutions,
           "metadata": {
               "number_of_solutions": len(submission_solutions),
               "solution_dimension": len(submission_solutions[0]) if submission_solutions else 0
           }
       }

# Save to file
       with open(filename, 'w') as f:
           json.dump([submission_data], f, indent=2)
      
       print(f"\nSubmission created: {filename}")
       print(f"Number of solutions: {len(submission_solutions)}")
       print(f"Solution dimension: {len(submission_solutions[0])}")
      
       return submission_data
  
   def detailed_analysis(self, solution_index=0):
       """Show detailed analysis for a specific solution"""
       if not self.best_solutions or solution_index >= len(self.best_solutions):
           print("Invalid solution index.")
           return
      
       sol = self.best_solutions[solution_index]
       x = sol['x']
      
       print("\n" + "="*60)
       print(f"DETAILED ANALYSIS - Solution {solution_index + 1}")
       print("="*60)
      
       print(f"\nFeasibility: {'✓ FEASIBLE' if sol['feasible'] else '✗ INFEASIBLE'}")
       print(f"Communication cost (J1): {sol['fitness'][0]:.6f}")
       print(f"Infrastructure cost (J2): {sol['fitness'][1]:.6f}")
       print(f"Total cost (J1+J2): {sum(sol['fitness']):.6f}")
       print(f"Rover constraint violation: {sol['constraints'][0]:.6f}")
       print(f"Satellite constraint violation: {sol['constraints'][1]:.6f}")
      
       # Decode chromosome
       try:
           a1, e1, i1, w1, eta1, a2, e2, i2, w2, eta2, S1, P1, F1, S2, P2, F2, r1, r2, r3, r4 = x
          
           print(f"\n{'='*40}")
           print("WALKER CONSTELLATION 1")
           print(f"{'='*40}")
           print(f"  Semi-major axis: {a1:.1f} km (Altitude: {a1 - 6371:.1f} km)")
           print(f"  Eccentricity: {e1:.6f}")
           print(f"  Inclination: {np.degrees(i1):.2f}°")
           print(f"  Argument of perigee: {np.degrees(w1):.2f}°")
           print(f"  Quality factor (η): {eta1:.1f}")
           print(f"  Satellites per plane (S1): {int(S1)}")
           print(f"  Planes (P1): {int(P1)}")
           print(f"  Phasing factor (F1): {int(F1)}")
           print(f"  Total satellites: {int(S1 * P1)}")
           print(f"  Total η contribution: {eta1 * S1 * P1:.1f}")
          
           print(f"\n{'='*40}")
           print("WALKER CONSTELLATION 2")
           print(f"{'='*40}")
           print(f"  Semi-major axis: {a2:.1f} km (Altitude: {a2 - 6371:.1f} km)")
           print(f"  Eccentricity: {e2:.6f}")
           print(f"  Inclination: {np.degrees(i2):.2f}°")
           print(f"  Argument of perigee: {np.degrees(w2):.2f}°")
           print(f"  Quality factor (η): {eta2:.1f}")
           print(f"  Satellites per plane (S2): {int(S2)}")
           print(f"  Planes (P2): {int(P2)}")
           print(f"  Phasing factor (F2): {int(F2)}")
           print(f"  Total satellites: {int(S2 * P2)}")
           print(f"  Total η contribution: {eta2 * S2 * P2:.1f}")
          
           print(f"\n{'='*40}")
           print("SUMMARY")
           print(f"{'='*40}")
           total_sats = int(S1 * P1 + S2 * P2)
           total_eta = eta1 * S1 * P1 + eta2 * S2 * P2
           print(f"  Total satellites: {total_sats}")
           print(f"  Total quality (η): {total_eta:.1f}")
           print(f"  Average η per satellite: {total_eta/total_sats if total_sats > 0 else 0:.2f}")
           print(f"  Rover indices: {int(r1)}, {int(r2)}, {int(r3)}, {int(r4)}")
          
       except Exception as e:
           print(f"Error decoding solution: {e}")
      
       return sol
  
   def visualize_best_solution(self, solution_index=0):
       """Visualize the best solution"""
       if not self.best_solutions or solution_index >= len(self.best_solutions):
           print("Invalid solution index.")
           return
      
       best_x = self.best_solutions[solution_index]['x']
      
       fig = plt.figure(figsize=(18, 6))
      
       # Plot for different epochs
       epochs = [0, self.n_epochs//2, self.n_epochs-1]
       for i, epoch in enumerate(epochs):
           ax = fig.add_subplot(1, 3, i+1, projection='3d')
          
           # Use the plot method from constellation_udp
           try:
               self.udp.plot(best_x, src=0, dst=0, ep=epoch)
               ax.set_title(f'Epoch {epoch} (Time: {epoch*self.udp._duration/self.udp.n_epochs:.1f} days)')
           except:
               # Fallback simple plot
               ax.set_title(f'Epoch {epoch} (Plot failed)')
      
       plt.suptitle(f'Solution {solution_index + 1} Visualization', fontsize=16, fontweight='bold')
       plt.tight_layout()
       plt.show()
  
   def export_results(self, filename="optimization_results.json"):
       """Export all optimization results"""
       if not self.best_solutions:
           print("No results to export.")
           return
      
       export_data = {
           "optimization_parameters": {
               "population_size": self.population_size,
               "generations": self.generations,
               "random_seed": self.random_seed,
               "timestamp": datetime.now().isoformat()
           },
           "history": self.history,
           "best_solutions": [
               {
                   "x": sol['x'].tolist() if hasattr(sol['x'], 'tolist') else list(sol['x']),
                   "fitness": sol['fitness'].tolist() if hasattr(sol['fitness'], 'tolist') else list(sol['fitness']),
                   "constraints": sol['constraints'].tolist() if hasattr(sol['constraints'], 'tolist') else list(sol['constraints']),
                   "feasible": sol['feasible'],
                   "decoded": sol.get('decoded', {})
               }
               for sol in self.best_solutions
           ]
       }
      
       with open(filename, 'w') as f:
           json.dump(export_data, f, indent=2)
      
       print(f"Results exported to: {filename}")




# Quick test function
def run_optimization_demo(population_size=30, generations=50):
   """Run a quick demo of the optimization"""
  
   if not PYGMO_AVAILABLE or not CONSTELLATION_UDP_AVAILABLE:
       print("Required dependencies not available.")
       print("Install with: pip install pygmo")
       return
  
   print("Running optimization demo...")
  
   # Create optimizer
   optimizer = QuantumCommunicationsOptimizer(
       population_size=population_size,
       generations=generations,
       random_seed=42
   )
  
   # Run optimization
   solutions = optimizer.optimize(verbose=True)
  
   if solutions:
       # Analyze results
       optimizer.analyze_solutions(show_top=5)
      
       # Plot Pareto front
       optimizer.plot_pareto_front()
      
       # Plot optimization history
       optimizer.plot_optimization_history()
      
       # Create submission
       optimizer.create_submission()
      
       # Detailed analysis of best solution
       optimizer.detailed_analysis(0)
      
       # Export results
       optimizer.export_results()
  
   return optimizer




if __name__ == "__main__":
   # Test with smaller parameters for quick demo
   optimizer = run_optimization_demo(population_size=20, generations=30)


  


