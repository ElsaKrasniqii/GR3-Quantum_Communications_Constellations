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

# Add current directory to path
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
    banner = """
╔══════════════════════════════════════════════════════════════╗
║     QUANTUM COMMUNICATIONS CONSTELLATIONS OPTIMIZER          ║
║                     SPOC-2 Challenge                         ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_environment():
    if not IMPORTS_SUCCESSFUL:
        return False

    print("Checking environment and dependencies...")
    all_ok, critical_ok, deps = check_dependencies(verbose=False)

    if not critical_ok:
        print("\n CRITICAL ERROR: Missing critical dependencies!")
        check_dependencies(verbose=True)
        return False

    rover_file = "./data/spoc2/constellations/rovers.txt"
    if not os.path.exists(rover_file):
        print(f"\nWARNING: Rover data file not found: {rover_file}")
        if input("Continue anyway? (y/n): ").strip().lower() != 'y':
            return False

    print("\n Environment check passed!")
    return True

def setup_argument_parser():
    parser = argparse.ArgumentParser(description="Quantum Communications Constellations Optimizer")
    parser.add_argument('command', nargs='?', default='interactive',
                        choices=['optimize', 'example', 'test', 'analyze', 'interactive'])
    return parser
def run_full_optimization(args):
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

        print("\nStarting optimization...")
        solutions = optimizer.optimize(verbose=True)

        if solutions:
            elapsed = time.time() - start_time
            print(f"\nOptimization completed in {elapsed:.1f} seconds")

            optimizer.analyze_solutions(show_top=10)

            if not args.batch and args.plot:
                optimizer.plot_pareto_front()
                optimizer.plot_optimization_history()

            optimizer.create_submission(args.output, top_n=args.top_solutions)

            return optimizer
        else:
            print("\nNo solutions found during optimization.")
            return None

    except Exception as e:
        print(f"\nOptimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def run_example_analysis(args):
    print("\n" + "="*70)
    print("RUNNING EXAMPLE ANALYSIS")
    print("="*70)

    try:
        udp = constellation_udp()
        x_example = udp.example(verbose=True)

        print("\nExample solution parameters:")
        print_solution_summary(x_example, udp)

        print("\nValidating example solution...")
        validation = validate_solution(x_example, udp)
        from utils import print_validation_results
        print_validation_results(validation)

        try:
            test_points = [
                [x_example[0]/10000, x_example[1]/100],
                [x_example[2]/10, x_example[3]/10]
            ]
            hv_score = combine_scores(test_points)
            print(f"\nHypervolume score: {hv_score:.4f}")
        except:
            pass

        if not args.batch and args.visualize:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(15, 5))
            epochs = [0, udp.n_epochs//2, udp.n_epochs-1]

            for i, epoch in enumerate(epochs):
                ax = fig.add_subplot(1, 3, i+1, projection='3d')
                udp.plot(x_example, src=0, dst=0, ep=epoch)
                ax.set_title(f'Epoch {epoch}')

            plt.tight_layout()
            plt.show()

        return x_example

    except Exception as e:
        print(f"\nExample analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = setup_argument_parser()
    args = parser.parse_args()
    print_banner()

    if args.command not in ['test', 'interactive']:
        if not check_environment():
            sys.exit(1)

if __name__ == "__main__":
    main()
