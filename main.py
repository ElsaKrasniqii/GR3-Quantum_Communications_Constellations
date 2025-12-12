


import sys



import os

import time
import json
import numpy as np

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try imports
try:
    from optimizer import QuantumCommunicationsOptimizer
    from utils import print_solution_summary
    from constellation_udp import constellation_udp
    IMPORTS_OK = True

except Exception as e:

    print(f"❌ Import error: {e}")

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
# PRINT TOP SOLUTIONS WITH DETAILS
# ==========================================================
def print_top_solutions_detailed(solutions, show_top=10):
    """Print detailed information for top solutions."""
    if not solutions:
        print("No solutions to display.")
        return
    
    print(f"\n📊 TOP {min(show_top, len(solutions))} SOLUTIONS ANALYSIS")
    print("="*60)
    
    # Kategorizo zgjidhjet
    #feasible_solutions = [s for s in solutions if s.get("feasible", False)]
    pareto_solutions = [s for s in solutions if s.get("pareto_optimal", False)]
    
    print(f"Total Solutions: {len(solutions)}")
    print(f"Pareto-Optimal: {len(pareto_solutions)}")
    #print(f"Feasible: {len(feasible_solutions)}")
    #print(f"Infeasible: {len(solutions) - len(feasible_solutions)}")
    
    # Rendit sipas J1 (Communication Cost)
    sorted_solutions = sorted(solutions, key=lambda s: s["fitness"][0])
    
    # Shfaq top solutions
    for i, sol in enumerate(sorted_solutions[:show_top]):
        print(f"\n{'='*40}")
        print(f"SOLUTION #{i+1}")
        print(f"{'='*40}")
        
        # Status indicators
        status = []
        #if sol.get("feasible", False):
         #   status.append("✅ FEASIBLE")
        #else:
         #   status.append("❌ INFEASIBLE")
        
        if sol.get("pareto_optimal", False):
            status.append("⭐ PARETO-OPTIMAL")
        
        print(f"Status: {' | '.join(status)}")
        print(f"Crowding Distance: {sol.get('crowding_distance', 0):.6f}")
        
        # Fitness values
        fitness = sol["fitness"]
        print(f"\n📈 FITNESS VALUES:")
        print(f"  J1 (Communication Cost): {fitness[0]:.6f}")
        print(f"  J2 (Infrastructure Cost): {fitness[1]:.6f}")
        
        # Constraint information
        if "constraint_info" in sol and sol["constraint_info"]:
            info = sol["constraint_info"]
            print(f"\n📏 CONSTRAINT ANALYSIS:")
            
            if "rover_distance" in info:
                rover_dist = info["rover_distance"]
                rover_violated = info.get("rover_violated", False)
                rover_status = "✅" if not rover_violated else "❌"
                print(f"  {rover_status} Rover Distance: {rover_dist:.2f} km")
            
            if "satellite_distance" in info:
                sat_dist = info["satellite_distance"]
                sat_violated = info.get("sat_violated", False)
                sat_status = "✅" if not sat_violated else "❌"
                print(f"  {sat_status} Satellite Distance: {sat_dist:.2f} km")
        
        # Decision variables (first 5 values for brevity)
        if "x" in sol and sol["x"] is not None:
            x_values = sol["x"]
            if hasattr(x_values, 'tolist'):
                x_values = x_values.tolist()
            
            print(f"\n🎯 DECISION VARIABLES (first 5 of {len(x_values)}):")
            for j in range(min(5, len(x_values))):
                print(f"  x[{j}] = {x_values[j]:.4f}")
            
            if len(x_values) > 5:
                print(f"  ... and {len(x_values) - 5} more variables")
        
        # Additional metrics
        if "additional_metrics" in sol:
            metrics = sol["additional_metrics"]
            print(f"\n📊 ADDITIONAL METRICS:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")


# ==========================================================
# SAVE SUBMISSION FILE (ROBUST VERSION)
# ==========================================================
def save_submission_from_optimizer(optimizer, solutions, filename="submission_file.json"):
    """
    Ruan submission JSON duke nxjerrë decision vectors
    nga struktura reale e zgjidhjeve.
    Funksionon edhe nëse s'ka get_x().
    """
    decision_vectors = []

    # RASTI 1: solutions është listë dictionaries me "x"
    if isinstance(solutions, list):
        for sol in solutions:
            if isinstance(sol, dict) and "x" in sol and sol["x"] is not None:
                x = sol["x"]
                if hasattr(x, "tolist"):
                    x = x.tolist()
                decision_vectors.append(x)

    # RASTI 2: optimizer ka atribut population si listë
    if not decision_vectors and hasattr(optimizer, "population"):
        pop = optimizer.population
        if isinstance(pop, list):
            for item in pop:
                if isinstance(item, dict) and "x" in item:
                    x = item["x"]
                    if hasattr(x, "tolist"):
                        x = x.tolist()
                    decision_vectors.append(x)

    if not decision_vectors:
        print("❌ Nuk u gjet asnjë decision vector (x). Submission NUK u krijua.")
        return

    submission = {
        "challenge": "spoc-2-quantum-communications-constellations",
        "problem": "quantum-communications-constellations",
        "decisionVector": decision_vectors
    }

    with open(filename, "w") as f:
        json.dump(submission, f, indent=2)

    print(f"\n📁 Submission file u ruajt me sukses: {filename}")
    print(f"📊 Numri i decision vectors: {len(decision_vectors)}")


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
            print("❌ No solutions found.")
            return

        elapsed = time.time() - start
        print(f"\n✅ Optimization completed in {elapsed:.2f} seconds")
        print(f"   Found {len(sols)} Pareto-optimal solutions\n")

        # ✅ SHFAQ VETËM TOP 10 SOLUTIONS
        optimizer.analyze_solutions(show_top=10)

        save_submission_from_optimizer(
        optimizer,
        sols,
        filename="submission_file.json"
        )

        print("\n🎉 OPTIMIZATION FINISHED SUCCESSFULLY!")

    except Exception as e:
        print(f"\n❌ OPTIMIZATION FAILED: {e}")
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
      
        print("🔧 Example solution loaded successfully!")



        print_solution_summary(x, udp)

    except Exception as e:

        print(f"❌ Example analysis failed: {e}")




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

        print("   ✅ constellation_udp loaded")

        # Test 2: Example
        print("\n2. Testing example solution...")
        x = udp.example()
        fitness = udp.fitness(x)

        print(f"   ✅ Example fitness: {fitness}")

        # Test 3: Optimizer
        print("\n3. Testing optimizer...")

        optimizer = QuantumCommunicationsOptimizer(
            population_size=10, 
            generations=5,
            random_seed=42
        )

        print("   ✅ Optimizer created")
        
        # Run quick optimization
        print("\n4. Running quick optimization...")
        solutions = optimizer.optimize(verbose=False)
        
        if solutions:
            print(f"   ✅ Found {len(solutions)} solutions")
            
            # Show top 3 solutions
            print("\n5. Showing top 3 solutions:")
            print_top_solutions_detailed(solutions, show_top=3)
        else:
            print("   ❌ No solutions found")

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)

    except Exception as e:

        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


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

            print("Goodbye! 👋")
            print("="*60 + "\n")
            break
        else:
            print("❌ Invalid option. Please choose 1-5.")
        
        # Pause between operations
        if choice in ["1", "3", "4"]:
            input("\nPress Enter to continue...")

# ==========================================================
# MAIN
# ==========================================================
def main():
    print_banner()

    if not IMPORTS_OK:

        print("❌ Fix import errors first.")
        
        print("\nRequired packages:")
        print("  numpy, scipy, pykep, sgp4, networkx, pygmo, matplotlib")
        return

    main_menu()


# ==========================================================
# ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    main()
 