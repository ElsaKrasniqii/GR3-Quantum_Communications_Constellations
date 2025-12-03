      
#!/usr/bin/env python3
"""
Quantum Communications Constellations Optimizer
Main entry point for the optimization problem
SPOC-2 Challenge: Quantum Communications Constellations
"""

import sys
import os
import argparse
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
   from Optimizer import QuantumCommunicationsOptimizer
   from utils import (
       check_dependencies,
       print_solution_summary,
       validate_solution,
       save_solutions,
       load_solutions,
       combine_scores
   )
   from constellation_udp import constellation_udp
   IMPORTS_SUCCESSFUL = True
except ImportError as e:
   print(f"Import error: {e}")
   print("Please ensure all required files are in the current directory:")
   print("  - constellation_udp.py")
   print("  - Optimizer.py")
   print("  - utils.py")
   IMPORTS_SUCCESSFUL = False

def print_banner():
   """Print application banner"""
   banner = """
╔══════════════════════════════════════════════════════════════╗
║     QUANTUM COMMUNICATIONS CONSTELLATIONS OPTIMIZER          ║
║                     SPOC-2 Challenge                         ║
╚══════════════════════════════════════════════════════════════╝
   """
   print(banner)

def check_environment():
   """Check if the environment is properly set up"""
   if not IMPORTS_SUCCESSFUL:
       return False
  
   print("Checking environment and dependencies...")
   all_ok, critical_ok, deps = check_dependencies(verbose=False)
  
   if not critical_ok:
       print("\n CRITICAL ERROR: Missing critical dependencies!")
       print("Please install missing dependencies before running the optimizer.")
       check_dependencies(verbose=True)
       return False
  
   rover_file = "./data/spoc2/constellations/rovers.txt"
   if not os.path.exists(rover_file):
       print(f"\n WARNING: Rover data file not found: {rover_file}")
       print("The optimizer may not work correctly without this file.")
       response = input("Continue anyway? (y/n): ").strip().lower()
       if response != 'y':
           return False
  
   print("\n Environment check passed!")
   return True

def run_full_optimization(args):
   """Run the complete optimization process"""
   print("\n" + "="*70)
   print("STARTING FULL OPTIMIZATION")
   print("="*70)
  
   start_time = time.time()
  
   try:
       optimizer = QuantumCommunicationsOptimizer(
           population_size=args.population,
           generations=args.generations,
           use_multithreading=args.parallel,
           random_seed=args.seed
       )
      
       print(f"Optimization parameters:")
       print(f"  Population size: {args.population}")
       print(f"  Generations: {args.generations}")
       print(f"  Random seed: {args.seed}")
       print(f"  Parallel: {'Yes' if args.parallel else 'No'}")
       print(f"  Output file: {args.output}")
       print("-"*70)
      
       print("\n Starting optimization...")
       solutions = optimizer.optimize(verbose=True)
      
       if solutions:
           elapsed_time = time.time() - start_time
           print(f"\n Optimization completed in {elapsed_time:.1f} seconds")
          
           if args.save_backup:
               backup_file = f"solutions_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
               save_solutions(solutions, backup_file)
          
           print("\n" + "="*70)
           print("RESULTS ANALYSIS")
           print("="*70)
           optimizer.analyze_solutions(show_top=10)
          
           if not args.batch and args.plot:
               print("\n Generating Pareto front plot...")
               optimizer.plot_pareto_front()
          
           if not args.batch and args.plot:
               print("\n Generating optimization history plot...")
               optimizer.plot_optimization_history()
          
           if args.detailed:
               print("\n" + "="*70)
               print("DETAILED ANALYSIS OF BEST SOLUTION")
               print("="*70)
               optimizer.detailed_analysis(0)
          
           if not args.batch and args.visualize:
               print("\n Visualizing best solution...")
               optimizer.visualize_best_solution(0)
          
           print("\n Creating submission file...")
           optimizer.create_submission(args.output, top_n=args.top_solutions)
          
           if args.export:
               results_file = f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
               optimizer.export_results(results_file)
          
           print(f"\n Optimization process completed successfully!")
           print(f"   Submission saved to: {args.output}")
          
           return optimizer
       else:
           print("\n No solutions found during optimization.")
           return None
          
   except Exception as e:
       print(f"\n Optimization failed with error: {e}")
       import traceback
       traceback.print_exc()
       return None




def run_example_analysis(args):
   """Run example analysis without optimization"""
   print("\n" + "="*70)
   print("RUNNING EXAMPLE ANALYSIS")
   print("="*70)
  
   try:
       udp = constellation_udp()
       x_example = udp.example(verbose=True)
      
       print("\n Example solution parameters:")
       print_solution_summary(x_example, udp)
      
       print("\n Validating example solution...")
       validation = validate_solution(x_example, udp)
       from utils import print_validation_results
       print_validation_results(validation)
      
       try:
           test_points = [
               [x_example[0]/10000, x_example[1]/100],
               [x_example[2]/10, x_example[3]/10]
           ]
           hv_score = combine_scores(test_points)
           print(f"\n Hypervolume score for test points: {hv_score:.4f}")
       except:
           pass
      
       if not args.batch and args.visualize:
           print("\n Visualizing example solution...")
           import matplotlib.pyplot as plt
          
           fig = plt.figure(figsize=(15, 5))
          
           epochs = [0, udp.n_epochs//2, udp.n_epochs-1]
           for i, epoch in enumerate(epochs):
               ax = fig.add_subplot(1, 3, i+1, projection='3d')
               udp.plot(x_example, src=0, dst=0, ep=epoch)
               ax.set_title(f'Epoch {epoch}')
          
           plt.suptitle('Example Solution Visualization', fontsize=14, fontweight='bold')
           plt.tight_layout()
           plt.show()
      
       return x_example
      
   except Exception as e:
       print(f"\n Example analysis failed: {e}")
       import traceback
       traceback.print_exc()
       return None

def analyze_solution_file(args):
   """Analyze a solution from a file"""
   print("\n" + "="*70)
   print(f"ANALYZING SOLUTION FILE: {args.analyze}")
   print("="*70)
  
   try:
       if args.analyze.endswith('.json'):
           solutions = load_solutions(args.analyze)
           if solutions:
               print(f"Found {len(solutions)} solutions in file.")
              
               for i, sol in enumerate(solutions[:min(5, len(solutions))]):
                   print(f"\n Solution {i+1}:")
                   print_solution_summary(sol['x'])
                  
                   if not args.batch:
                       import matplotlib.pyplot as plt
                       from Optimizer import QuantumCommunicationsOptimizer
                      
                       optimizer = QuantumCommunicationsOptimizer()
                       optimizer.best_solutions = [sol]
                       optimizer.visualize_best_solution(0)
                      
                       if i < len(solutions) - 1:
                           response = input(f"\nShow next solution? (y/n): ").strip().lower()
                           if response != 'y':
                               break
           else:
               print("No solutions found in the file.")
       else:
           with open(args.analyze, 'r') as f:
               content = f.read().strip()
          
           try:
               import json
               x = json.loads(content)
           except:
               import re
               numbers = re.findall(r"[-+]?\d*\.\d+|\d+", content)
               x = [float(num) for num in numbers]
          
           if len(x) != 20:
               print(f"Warning: Expected 20 parameters, got {len(x)}")
          
           udp = constellation_udp()
           print_solution_summary(x, udp)
          
           if not args.batch and args.visualize:
               import matplotlib.pyplot as plt
              
               fig = plt.figure(figsize=(15, 5))
               epochs = [0, 5, 9]
               for i, epoch in enumerate(epochs):
                   ax = fig.add_subplot(1, 3, i+1, projection='3d')
                   udp.plot(x, src=0, dst=0, ep=epoch)
                   ax.set_title(f'Epoch {epoch}')
              
               plt.suptitle('Loaded Solution Visualization', fontsize=14, fontweight='bold')
               plt.tight_layout()
               plt.show()
              
   except Exception as e:
       print(f"\n Error analyzing file: {e}")
       import traceback
       traceback.print_exc()

def run_quick_test(args):
   """Run a quick test to verify everything works"""
   print("\n" + "="*70)
   print("RUNNING QUICK TEST")
   print("="*70)
  
   print("\n1. Testing imports...")
   if IMPORTS_SUCCESSFUL:
       print("    All imports successful")
   else:
       print("    Import failed")
       return False
  
   print("\n2. Testing UDP class...")
   try:
       udp = constellation_udp()
       print("   UDP class instantiated successfully")
      
       x_example = udp.example()
       print(f"    Example solution generated (length: {len(x_example)})")
      
       fitness = udp.fitness(x_example)
       print(f"    Fitness evaluation successful: {fitness}")
      
   except Exception as e:
       print(f"    UDP test failed: {e}")
       return False
  
   print("\n3. Testing optimizer...")
   try:
       from Optimizer import QuantumCommunicationsOptimizer
       optimizer = QuantumCommunicationsOptimizer(population_size=5, generations=2)
       print("    Optimizer instantiated successfully")
   except Exception as e:
       print(f"    Optimizer test failed: {e}")
       return False
  
   print("\n4. Testing utility functions...")
   try:
       from utils import validate_solution, combine_scores
      
       validation = validate_solution(x_example)
       print(f"   Validation function works: {validation['valid']}")
      
       test_points = [[0.5, 0.3], [0.6, 0.2]]
       hv = combine_scores(test_points)
       print(f"   Hypervolume calculation works: {hv:.4f}")
      
   except Exception as e:
       print(f"   Utility functions test failed: {e}")
       return False
  
   print("\n" + "="*70)
   print("All tests passed! The system is ready for optimization.")
   print("="*70)
  
   return True




def setup_argument_parser():
   """Set up command line argument parser"""
   parser = argparse.ArgumentParser(
       description="Quantum Communications Constellations Optimizer for SPOC-2 Challenge",
       formatter_class=argparse.RawDescriptionHelpFormatter,
       epilog="""
Examples:
 %(prog)s optimize                 
 %(prog)s example                  
 %(prog)s test                     
 %(prog)s --pop 100 --gen 200      
 %(prog)s --batch --output my_submission.json 
       """
   )
  
   parser.add_argument(
       'command',
       nargs='?',
       default='interactive',
       choices=['optimize', 'example', 'test', 'analyze', 'interactive'],
       help='Command to execute (default: interactive)'
   )
  
   parser.add_argument(
       '--pop', '--population',
       dest='population',
       type=int,
       default=50,
       help='Population size for NSGA-II (default: 50)'
   )
   parser.add_argument(
       '--gen', '--generations',
       dest='generations',
       type=int,
       default=100,
       help='Number of generations (default: 100)'
   )
   parser.add_argument(
       '--seed',
       type=int,
       default=42,
       help='Random seed for reproducibility (default: 42)'
   )
   parser.add_argument(
       '--parallel',
       action='store_true',
       help='Enable parallel evaluation (if available)'
   )
   parser.add_argument(
       '--top',
       dest='top_solutions',
       type=int,
       default=20,
       help='Number of top solutions to include in submission (default: 20)'
   )
  
   parser.add_argument(
       '-o', '--output',
       default='quantum_constellations_submission.json',
       help='Output submission filename (default: quantum_constellations_submission.json)'
   )
   parser.add_argument(
       '--export',
       action='store_true',
       help='Export full optimization results to JSON'
   )
   parser.add_argument(
       '--save-backup',
       action='store_true',
       help='Save intermediate solutions backup'
   )
  
   parser.add_argument(
       '--analyze',
       metavar='FILE',
       help='Analyze a solution file (JSON or text)'
   )
   parser.add_argument(
       '--detailed',
       action='store_true',
       help='Show detailed analysis of solutions'
   )
  
   parser.add_argument(
       '--plot',
       action='store_true',
       default=True,
       help='Generate plots (default: True)'
   )
   parser.add_argument(
       '--no-plot',
       dest='plot',
       action='store_false',
       help='Disable plotting'
   )
   parser.add_argument(
       '--visualize',
       action='store_true',
       default=True,
       help='Visualize solutions (default: True)'
   )
   parser.add_argument(
       '--no-visualize',
       dest='visualize',
       action='store_false',
       help='Disable visualization'
   )
   parser.add_argument(
       '--batch',
       action='store_true',
       help='Run in batch mode (no interactive prompts, minimal output)'
   )
  
   return parser

def interactive_menu():
   """Display interactive menu"""
   print_banner()
  
   while True:
       print("\n" + "="*70)
       print("MAIN MENU")
       print("="*70)
       print("1. Run full optimization")
       print("2. Run example analysis")
       print("3. Run quick test")
       print("4. Analyze solution file")
       print("5. Check dependencies")
       print("6. Exit")
       print("="*70)
      
       try:
           choice = input("\nEnter your choice (1-6): ").strip()
          
           if choice == "1":
               print("\nOptimization Parameters:")
               pop = input(f"  Population size [50]: ").strip()
               pop = int(pop) if pop else 50
              
               gen = input(f"  Generations [100]: ").strip()
               gen = int(gen) if gen else 100
              
               seed = input(f"  Random seed [42]: ").strip()
               seed = int(seed) if seed else 42
              
               class Args:
                   pass
               args = Args()
               args.population = pop
               args.generations = gen
               args.seed = seed
               args.parallel = False
               args.top_solutions = 20
               args.output = "quantum_constellations_submission.json"
               args.export = True
               args.save_backup = True
               args.detailed = True
               args.plot = True
               args.visualize = True
               args.batch = False
              
               run_full_optimization(args)
              
           elif choice == "2":
               class Args:
                   pass
               args = Args()
               args.batch = False
               args.visualize = True
               run_example_analysis(args)
              
           elif choice == "3":
               class Args:
                   pass
               args = Args()
               run_quick_test(args)
              
           elif choice == "4":
               filename = input("\nEnter solution filename: ").strip()
               if os.path.exists(filename):
                   class Args:
                       pass
                   args = Args()
                   args.analyze = filename
                   args.batch = False
                   args.visualize = True
                   analyze_solution_file(args)
               else:
                   print(f"File not found: {filename}")
                  
           elif choice == "5":
               check_dependencies(verbose=True)
              
           elif choice == "6":
               print("\nGoodbye!")
               break
              
           else:
               print("Invalid choice. Please enter a number between 1 and 6.")
              
       except KeyboardInterrupt:
           print("\n\nOperation cancelled by user.")
           break
       except Exception as e:
           print(f"\nError: {e}")


def main():
   """Main function with command line interface"""
   parser = setup_argument_parser()
   args = parser.parse_args()
  
   print_banner()
  
   if args.command not in ['test', 'interactive']:
       if not check_environment():
           print("\n Environment check failed. Exiting.")
           sys.exit(1)
  
   if args.command == "optimize":
       run_full_optimization(args)
      
   elif args.command == "example":
       run_example_analysis(args)
      
   elif args.command == "test":
       success = run_quick_test(args)
       sys.exit(0 if success else 1)
      
   elif args.command == "analyze":
       if args.analyze:
           analyze_solution_file(args)
       else:
           print("\nPlease specify a file to analyze with --analyze FILENAME")
           parser.print_help()
          
   elif args.command == "interactive":
       if args.batch:
           print("Cannot use batch mode with interactive command")
           sys.exit(1)
       interactive_menu()
  
   else:
       print(f"\nUnknown command: {args.command}")
       parser.print_help()
       sys.exit(1)


if __name__ == "__main__":
   try:
       main()
   except KeyboardInterrupt:
       print("\n\nOperation cancelled by user.")
       sys.exit(0)
   except Exception as e:
       print(f"\n❌ Unexpected error: {e}")
       import traceback
       traceback.print_exc()
       sys.exit(1)


