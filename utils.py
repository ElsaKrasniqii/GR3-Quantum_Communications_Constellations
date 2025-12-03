# utils.py - Version 1
import numpy as np
import json
from datetime import datetime
import warnings
from datetime import datetime
import json


warnings.filterwarnings('ignore')

__version__ = "1.0.0"
__author__ = "Quantum Communications Team"
__date__ = "2025"

print("Initializing utils module...")


def combine_scores(points, ref_point=None):
    """
    Function for aggregating single solutions into one score using hypervolume indicator
    """
    try:
        import pygmo as pg
        
        if ref_point is None:
            ref_point = np.array([1.2, 1.4])
        
        points_array = np.array(points)
        
        if points_array.shape[1] < 2:
            raise ValueError("Points must have at least 2 objectives")
        
        filtered_points = []
        for point in points_array:
            if np.any(np.isnan(point)) or np.any(np.isinf(point)):
                continue
            
            if len(point) >= 2:
                dominates = all(p <= r for p, r in zip(point[:2], ref_point))
                if dominates:
                    filtered_points.append(point[:2])
        
        if len(filtered_points) == 0:
            print("Warning: No solutions dominate the reference point")
            return 0.0
        else:
            filtered_array = np.array(filtered_points)
            max_vals = filtered_array.max(axis=0) + 1e-10
            min_vals = filtered_array.min(axis=0)
            
            normalized_points = (filtered_array - min_vals) / (max_vals - min_vals)
            normalized_ref = (ref_point - min_vals[:2]) / (max_vals[:2] - min_vals[:2])
            
            hv = pg.hypervolume(normalized_points)
            hypervolume_value = hv.compute(normalized_ref)
            
            scaled_hv = -hypervolume_value * 10000
            
            print(f"Hypervolume calculation: {len(filtered_points)} points, HV = {hypervolume_value:.6f}")
            return scaled_hv
            
    except ImportError:
        print("Pygmo not available. Using simple aggregation.")
        if len(points) > 0:
            points_array = np.array(points)
            if points_array.shape[1] >= 2:
                normalized = points_array[:, :2] / np.max(points_array[:, :2], axis=0)
                return -np.sum(np.min(normalized, axis=0))
        return 0.0
    except Exception as e:
        print(f"Error in hypervolume calculation: {e}")
        return 0.0
    

  def save_solutions(solutions, filename="solutions_backup.json", metadata=None):
    """Save solutions to a JSON file for later use"""
    try:
        solutions_data = []
        for sol in solutions:

            # Verify required keys exist
            for key in ["x", "fitness", "constraints", "feasible"]:
                if key not in sol:
                    raise KeyError(f"Missing key '{key}' in solution: {sol}")

            solution_data = {
                "x": sol['x'].tolist() if hasattr(sol['x'], 'tolist') else list(sol['x']),
                "fitness": sol['fitness'].tolist() if hasattr(sol['fitness'], 'tolist') else list(sol['fitness']),
                "constraints": sol['constraints'].tolist() if hasattr(sol['constraints'], 'tolist') else list(sol['constraints']),
                "feasible": sol['feasible']
            }

            if 'decoded' in sol:
                solution_data['decoded'] = sol['decoded']

            solutions_data.append(solution_data)

        if metadata is None:
            metadata = {}

        data = {
            "timestamp": datetime.now().isoformat(),
            "number_of_solutions": len(solutions),
            "solution_dimension": len(solutions[0]['x']) if solutions else 0,
            "metadata": metadata,
            "solutions": solutions_data
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"✓ Saved {len(solutions)} solutions to {filename}")
        return True

    except Exception as e:
        print(f"✗ Error saving solutions: {e}")
        return False


def load_solutions(filename="solutions_backup.json", verbose=True):
    """Load solutions from a JSON file"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        solutions = []
        for sol_data in data['solutions']:
            solution = {
                "x": np.array(sol_data['x']),
                "fitness": np.array(sol_data['fitness']),
                "constraints": np.array(sol_data['constraints']),
                "feasible": sol_data['feasible']
            }
            if 'decoded' in sol_data:
                solution['decoded'] = sol_data['decoded']
            solutions.append(solution)
        
        if verbose:
            print(f"✓ Loaded {len(solutions)} solutions from {filename}")
            print(f"  Timestamp: {data['timestamp']}")
            print(f"  Solution dimension: {data['solution_dimension']}")
        
        return solutions
        
    except FileNotFoundError:
        print(f"✗ File {filename} not found")
        return []
    except json.JSONDecodeError:
        print(f"✗ Error parsing JSON file {filename}")
        return []
    except Exception as e:
        print(f"✗ Error loading solutions: {e}")
        return []
    


def print_solution_summary(x, udp_instance=None):
    """Print a clean and safe summary of a solution"""

    try:
        # Ensure x can be treated as a list
        x = list(x)

        if len(x) != 20:
            print(f"Warning: Expected 20 parameters, got {len(x)}")
            return

        # Unpack parameters
        (a1, e1, i1, w1, eta1,
         a2, e2, i2, w2, eta2,
         S1, P1, F1, S2, P2, F2,
         r1, r2, r3, r4) = x

        print("\n" + "="*60)
        print("SOLUTION SUMMARY")
        print("="*60)

        # ---------------- WALKER 1 ----------------
        print("\n--- WALKER CONSTELLATION 1 ---")
        print(f"  Semi-major axis: {a1:.1f} km (Altitude: {a1 - 6371:.1f} km)")
        print(f"  Eccentricity: {e1:.6f}")
        print(f"  Inclination: {np.degrees(i1):.2f}°")
        print(f"  Argument of perigee: {np.degrees(w1):.2f}°")
        print(f"  Quality factor (η): {eta1:.1f}")
        print(f"  Satellites per plane (S): {int(S1)}")
        print(f"  Planes (P): {int(P1)}")
        print(f"  Phasing factor (F): {int(F1)}")
        print(f"  Total satellites: {int(S1 * P1)}")

        # ---------------- WALKER 2 ----------------
        print("\n--- WALKER CONSTELLATION 2 ---")
        print(f"  Semi-major axis: {a2:.1f} km (Altitude: {a2 - 6371:.1f} km)")
        print(f"  Eccentricity: {e2:.6f}")
        print(f"  Inclination: {np.degrees(i2):.2f}°")
        print(f"  Argument of perigee: {np.degrees(w2):.2f}°")
        print(f"  Quality factor (η): {eta2:.1f}")
        print(f"  Satellites per plane (S): {int(S2)}")
        print(f"  Planes (P): {int(P2)}")
        print(f"  Phasing factor (F): {int(F2)}")
        print(f"  Total satellites: {int(S2 * P2)}")

        # ---------------- SUMMARY ----------------
        print("\n--- SUMMARY ---")
        total_sats = int(S1 * P1 + S2 * P2)
        total_eta = eta1 * S1 * P1 + eta2 * S2 * P2

        print(f"  Total satellites: {total_sats}")
        print(f"  Total quality (η): {total_eta:.1f}")
        print(f"  Average η per satellite: {total_eta/total_sats if total_sats>0 else 0:.2f}")
        print(f"  Rover indices: {int(r1)}, {int(r2)}, {int(r3)}, {int(r4)}")

        # ---------------- FITNESS EVALUATION ----------------
        if udp_instance is not None:
            try:
                f = udp_instance.fitness(x)

                print("\n--- FITNESS EVALUATION ---")

                if isinstance(f, (list, tuple)) and len(f) == 4:
                    f1, f2, c1, c2 = f
                    print(f"  J1 (Communication): {f1:.6f}")
                    print(f"  J2 (Infrastructure): {f2:.6f}")
                    print(f"  Total cost (J1+J2): {f1 + f2:.6f}")
                    print(f"  Rover constraint: {c1:.6f} ({'✓ OK' if c1 <= 0 else '✗ VIOLATED'})")
                    print(f"  Satellite constraint: {c2:.6f} ({'✓ OK' if c2 <= 0 else '✗ VIOLATED'})")
                    print(f"  Feasible: {'✓ YES' if c1 <= 0 and c2 <= 0 else '✗ NO'}")
                else:
                    print("  Fitness result:", f)

            except Exception as e:
                print(f"⚠ Could not evaluate fitness: {e}")

        print("="*60 + "\n")

    except Exception as e:
        print(f"Error printing solution summary: {e}")


        def check_dependencies(verbose=True):
    """Check if all required dependencies are available"""
    dependencies = {
        'pykep': {'available': False, 'critical': True, 'version': None, 'install': 'pip install pykep'},
        'sgp4': {'available': False, 'critical': True, 'version': None, 'install': 'pip install sgp4'},
        'networkx': {'available': False, 'critical': True, 'version': None, 'install': 'pip install networkx'},
        'pygmo': {'available': False, 'critical': False, 'version': None, 'install': 'pip install pygmo'},
        'scipy': {'available': False, 'critical': True, 'version': None, 'install': 'pip install scipy'},
        'numpy': {'available': False, 'critical': True, 'version': None, 'install': 'pip install numpy'},
        'matplotlib': {'available': False, 'critical': False, 'version': None, 'install': 'pip install matplotlib'},
        'seaborn': {'available': False, 'critical': False, 'version': None, 'install': 'pip install seaborn'}
    }
    
    # Check numpy first
    try:
        import numpy as np
        dependencies['numpy']['available'] = True
        dependencies['numpy']['version'] = np.__version__
    except ImportError:
        pass
    
    # Check other dependencies
    for dep in dependencies:
        if dep == 'numpy':
            continue
            
        try:
            if dep == 'pykep':
                import pykep
                version = getattr(pykep, '__version__', 'unknown')
            elif dep == 'sgp4':
                from sgp4.api import Satrec
                import sgp4
                version = getattr(sgp4, '__version__', 'unknown')
            elif dep == 'networkx':
                import networkx as nx
                version = nx.__version__
            elif dep == 'pygmo':
                import pygmo as pg
                version = pg.__version__
            elif dep == 'scipy':
                import scipy
                version = scipy.__version__
            elif dep == 'matplotlib':
                import matplotlib
                version = matplotlib.__version__
            elif dep == 'seaborn':
                import seaborn as sns
                version = sns.__version__
            
            dependencies[dep]['available'] = True
            dependencies[dep]['version'] = version
        except ImportError:
            pass
    
    # Determine status
    critical_deps = [dep for dep, info in dependencies.items() if info['critical']]
    available_critical = all(dependencies[dep]['available'] for dep in critical_deps)
    all_available = all(info['available'] for info in dependencies.values())
    
    if verbose:
        print("\n" + "="*60)
        print("DEPENDENCY CHECK")
        print("="*60)
        
        print("\nCritical Dependencies:")
        for dep in critical_deps:
            info = dependencies[dep]
            status = "✓" if info['available'] else "✗"
            version = f"v{info['version']}" if info['version'] else "not installed"
            print(f"  {status} {dep:15} {version:20}")
            if not info['available']:
                print(f"       Install with: {info['install']}")
        
        print("\nOptional Dependencies:")
        optional_deps = [dep for dep, info in dependencies.items() if not info['critical']]
        for dep in optional_deps:
            info = dependencies[dep]
            status = "✓" if info['available'] else "✗"
            version = f"v{info['version']}" if info['version'] else "not installed"
            print(f"  {status} {dep:15} {version:20}")
            if not info['available']:
                print(f"       Install with: {info['install']}")
        
        print("\n" + "="*60)
        if available_critical:
            print("✓ All critical dependencies are available!")
        else:
            print("✗ Some critical dependencies are missing!")
        print("="*60)
    
    return all_available, available_critical, dependencies
