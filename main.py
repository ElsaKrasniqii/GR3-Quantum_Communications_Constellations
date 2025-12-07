#!/usr/bin/env python3
"""
Quantum Communications Constellations Optimizer
ESA SpOC-2 Challenge 
"""

import sys
import os

import time

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try imports
try:
    from optimizer import QuantumCommunicationsOptimizer
    from utils import print_solution_summary
    from constellation_udp import constellation_udp
    IMPORTS_OK = True
except Exception as e:
    print(f" Import error: {e}")
    IMPORTS_OK = False


# ==========================================================
# Banner
# ==========================================================
def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║   QUANTUM COMMUNICATIONS CONSTELLATIONS OPTIMIZER    ║
║                     ESA SPOC-2                       ║
╚══════════════════════════════════════════════════════╝
""")


# ==========================================================
# FULL OPTIMIZATION
# ==========================================================
def run_full_optimization():
    print("\n" + "="*60)
    print("                 FULL OPTIMIZATION")
    print("="*60)
    
    # Get parameters with defaults
    print("\nEnter optimization parameters:")
    try:
        pop = int(input("Population size [50]: ") or 50)
        gen = int(input("Generations [100]: ") or 100)
        seed = int(input("Random seed [42]: ") or 42)
    except ValueError:
        print("⚠ Invalid input! Using defaults...")
        pop, gen, seed = 50, 100, 42
    
    print(f"\nStarting optimization with:")
    print(f"  Population: {pop}")
    print(f"  Generations: {gen}")
    print(f"  Seed: {seed}")
    print("-"*60)

    start = time.time()

    try:
        optimizer = QuantumCommunicationsOptimizer(
            population_size=pop,
            generations=gen,
            random_seed=seed
        )

        # Run optimization
        sols = optimizer.optimize(verbose=True)

        if not sols:
            print(" No solutions found.")
            return

        elapsed = time.time() - start
        print(f"\nOptimization completed in {elapsed:.2f} seconds")
        print(f"   Found {len(sols)} Pareto-optimal solutions\n")

        # Show top solutions
        optimizer.analyze_solutions(show_top=10)

        # Plot Pareto front
        try:
            optimizer.plot_pareto_front()
        except:
            print("⚠ Could not generate plot")

        # Create submission file
        print("\n Creating submission file...")
        optimizer.create_submission("submission.json")

        print("\n OPTIMIZATION FINISHED!")
        print("   Submission file: submission.json\n")

    except Exception as e:
        print(f"\n OPTIMIZATION FAILED: {e}")


# ==========================================================
# EXAMPLE ANALYSIS
# ==========================================================
def run_example_analysis():
    print("\n" + "="*60)
    print("                EXAMPLE ANALYSIS")
    print("="*60 + "\n")

    try:
        udp = constellation_udp()
        x = udp.example()

        print(" Example solution loaded successfully!")
        print_solution_summary(x, udp)

    except Exception as e:
        print(f" Example analysis failed: {e}")


# ==========================================================
# QUICK TEST
# ==========================================================
def run_quick_test():
    print("\n" + "="*60)
    print("                   QUICK TEST")
    print("="*60 + "\n")

    print("Testing basic functionality...\n")
    
    try:
        # Test 1: UDP
        print("1. Testing constellation_udp...")
        udp = constellation_udp()
        print("    constellation_udp loaded")
        
        # Test 2: Example
        print("\n2. Testing example solution...")
        x = udp.example()
        fitness = udp.fitness(x)
        print(f"    Example fitness: {fitness}")
        
        # Test 3: Optimizer
        print("\n3. Testing optimizer...")
        optimizer = QuantumCommunicationsOptimizer(10, 5)
        print("    Optimizer created")
        
        print("\n" + "="*60)
        print(" ALL TESTS PASSED!")
        print("="*60)
        
    except Exception as e:
        print(f"\n TEST FAILED: {e}")


# ==========================================================
# MAIN MENU
# ==========================================================
def main_menu():
    while True:
        print("\n" + "="*60)
        print("                    MAIN MENU")
        print("="*60)
        print("1. Run full optimization")
        print("2. Run example analysis")
        print("3. Run quick test")
        print("4. Exit")
        print("="*60)

        choice = input("\nSelect option (1-4): ").strip()

        if choice == "1":
            run_full_optimization()
        elif choice == "2":
            run_example_analysis()
        elif choice == "3":
            run_quick_test()
        elif choice == "4":
            print("\n" + "="*60)
            print("Goodbye!")
            print("="*60 + "\n")
            break
        else:
            print(" Invalid option. Please choose 1-4.")


# ==========================================================
# MAIN
# ==========================================================
def main():
    print_banner()

    if not IMPORTS_OK:
        print(" Fix import errors first.")
        print("\nRequired packages:")
        print("  numpy, scipy, pykep, sgp4, networkx, pygmo, matplotlib")
        return

    main_menu()


# ==========================================================
# ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    main()
