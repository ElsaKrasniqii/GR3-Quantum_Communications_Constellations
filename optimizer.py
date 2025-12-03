import matplotlib.pyplot as plt
import json
from datetime import datetime
import warnings
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
