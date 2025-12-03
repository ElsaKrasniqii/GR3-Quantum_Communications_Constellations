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


def validate_solution(x, udp_instance=None):
    """Validate a solution vector"""
    validation = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'parameters': {}
    }
    
    try:
        # Check length
        if len(x) != 20:
            validation['valid'] = False
            validation['errors'].append(f"Expected 20 parameters, got {len(x)}")
            return validation
        
        # Parse parameters
        a1, e1, i1, w1, eta1, a2, e2, i2, w2, eta2, S1, P1, F1, S2, P2, F2, r1, r2, r3, r4 = x
        
        # Store parameter ranges
        validation['parameters'] = {
            'a1': a1, 'e1': e1, 'i1': i1, 'w1': w1, 'eta1': eta1,
            'a2': a2, 'e2': e2, 'i2': i2, 'w2': w2, 'eta2': eta2,
            'S1': S1, 'P1': P1, 'F1': F1,
            'S2': S2, 'P2': P2, 'F2': F2,
            'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4
        }
        
        # Check parameter ranges
        # Semi-major axis (should be > Earth radius ~6371 km)
        if a1 < 6371:
            validation['warnings'].append(f"Walker 1 semi-major axis ({a1:.1f} km) is below Earth radius")
        if a2 < 6371:
            validation['warnings'].append(f"Walker 2 semi-major axis ({a2:.1f} km) is below Earth radius")
        
        # Eccentricity (should be between 0 and 1 for elliptical orbits)
        if e1 < 0 or e1 >= 1:
            validation['errors'].append(f"Walker 1 eccentricity ({e1:.6f}) must be in [0, 1)")
        if e2 < 0 or e2 >= 1:
            validation['errors'].append(f"Walker 2 eccentricity ({e2:.6f}) must be in [0, 1)")
        
        # Inclination (should be in radians, typically [0, π])
        if i1 < 0 or i1 > np.pi:
            validation['warnings'].append(f"Walker 1 inclination ({np.degrees(i1):.1f}°) outside typical range [0°, 180°]")
        if i2 < 0 or i2 > np.pi:
            validation['warnings'].append(f"Walker 2 inclination ({np.degrees(i2):.1f}°) outside typical range [0°, 180°]")
        
        # Quality factor (eta) - positive
        if eta1 < 0:
            validation['warnings'].append(f"Walker 1 quality factor ({eta1:.1f}) is negative")
        if eta2 < 0:
            validation['warnings'].append(f"Walker 2 quality factor ({eta2:.1f}) is negative")
        
        # Constellation parameters (should be positive integers)
        for name, value in [('S1', S1), ('P1', P1), ('F1', F1), ('S2', S2), ('P2', P2), ('F2', F2)]:
            if value <= 0:
                validation['errors'].append(f"{name} ({value}) must be positive")
            if not np.isclose(value, np.round(value)):
                validation['warnings'].append(f"{name} ({value}) is not an integer")
        
        # Rover indices (should be valid indices)
        for i, r in enumerate([r1, r2, r3, r4], 1):
            if r < 0:
                validation['warnings'].append(f"Rover {i} index ({r}) is negative")
            if not np.isclose(r, np.round(r)):
                validation['warnings'].append(f"Rover {i} index ({r}) is not an integer")
        
        # Update validity based on errors
        if validation['errors']:
            validation['valid'] = False
        
    except Exception as e:
        validation['valid'] = False
        validation['errors'].append(f"Validation error: {e}")
    
    return validation

def validate_solution(x, udp_instance=None):
    """Validate a solution vector"""
    validation = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'parameters': {}
    }
    
    try:
        # Check length
        if len(x) != 20:
            validation['valid'] = False
            validation['errors'].append(f"Expected 20 parameters, got {len(x)}")
            return validation
        
        # Parse parameters
        a1, e1, i1, w1, eta1, a2, e2, i2, w2, eta2, S1, P1, F1, S2, P2, F2, r1, r2, r3, r4 = x
        
        # Store parameter ranges
        validation['parameters'] = {
            'a1': a1, 'e1': e1, 'i1': i1, 'w1': w1, 'eta1': eta1,
            'a2': a2, 'e2': e2, 'i2': i2, 'w2': w2, 'eta2': eta2,
            'S1': S1, 'P1': P1, 'F1': F1,
            'S2': S2, 'P2': P2, 'F2': F2,
            'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4
        }
        
        # Check parameter ranges
        # Semi-major axis (should be > Earth radius ~6371 km)
        if a1 < 6371:
            validation['warnings'].append(f"Walker 1 semi-major axis ({a1:.1f} km) is below Earth radius")
        if a2 < 6371:
            validation['warnings'].append(f"Walker 2 semi-major axis ({a2:.1f} km) is below Earth radius")
        
        # Eccentricity (should be between 0 and 1 for elliptical orbits)
        if e1 < 0 or e1 >= 1:
            validation['errors'].append(f"Walker 1 eccentricity ({e1:.6f}) must be in [0, 1)")
        if e2 < 0 or e2 >= 1:
            validation['errors'].append(f"Walker 2 eccentricity ({e2:.6f}) must be in [0, 1)")
        
        # Inclination (should be in radians, typically [0, π])
        if i1 < 0 or i1 > np.pi:
            validation['warnings'].append(f"Walker 1 inclination ({np.degrees(i1):.1f}°) outside typical range [0°, 180°]")
        if i2 < 0 or i2 > np.pi:
            validation['warnings'].append(f"Walker 2 inclination ({np.degrees(i2):.1f}°) outside typical range [0°, 180°]")
        
        # Quality factor (eta) - positive
        if eta1 < 0:
            validation['warnings'].append(f"Walker 1 quality factor ({eta1:.1f}) is negative")
        if eta2 < 0:
            validation['warnings'].append(f"Walker 2 quality factor ({eta2:.1f}) is negative")
        
        # Constellation parameters (should be positive integers)
        for name, value in [('S1', S1), ('P1', P1), ('F1', F1), ('S2', S2), ('P2', P2), ('F2', F2)]:
            if value <= 0:
                validation['errors'].append(f"{name} ({value}) must be positive")
            if not np.isclose(value, np.round(value)):
                validation['warnings'].append(f"{name} ({value}) is not an integer")
        
        # Rover indices (should be valid indices)
        for i, r in enumerate([r1, r2, r3, r4], 1):
            if r < 0:
                validation['warnings'].append(f"Rover {i} index ({r}) is negative")
            if not np.isclose(r, np.round(r)):
                validation['warnings'].append(f"Rover {i} index ({r}) is not an integer")
        
        # Check feasibility if UDP instance is provided
        if udp_instance is not None:
            try:
                f1, f2, c1, c2 = udp_instance.fitness(x)
                validation['fitness'] = [f1, f2, c1, c2]
                validation['feasible'] = c1 <= 0 and c2 <= 0
                
                if not validation['feasible']:
                    validation['warnings'].append("Solution violates constraints")
            except Exception as e:
                validation['warnings'].append(f"Could not evaluate fitness: {e}")
        
        # Update validity based on errors
        if validation['errors']:
            validation['valid'] = False
        
    except Exception as e:
        validation['valid'] = False
        validation['errors'].append(f"Validation error: {e}")
    
    return validation



def print_validation_results(validation):
    """Print validation results in a readable format"""
    print("\n" + "="*60)
    print("SOLUTION VALIDATION")
    print("="*60)
    
    if validation['valid']:
        print("✓ Solution is VALID")
    else:
        print("✗ Solution is INVALID")
    
    if validation['errors']:
        print("\nERRORS:")
        for error in validation['errors']:
            print(f"  ✗ {error}")
    
    if validation['warnings']:
        print("\nWARNINGS:")
        for warning in validation['warnings']:
            print(f"  ⚠ {warning}")
    
    if 'fitness' in validation:
        print("\nFITNESS:")
        f1, f2, c1, c2 = validation['fitness']
        print(f"  J1 (Communication): {f1:.6f}")
        print(f"  J2 (Infrastructure): {f2:.6f}")
        print(f"  Rover constraint: {c1:.6f}")
        print(f"  Satellite constraint: {c2:.6f}")
        print(f"  Feasible: {'✓ YES' if validation.get('feasible', False) else '✗ NO'}")
    
    print("="*60 + "\n")


    # Example usage
if __name__ == "__main__":
    print("Testing utility functions...\n")
    
    # Check dependencies
    check_dependencies()
    
    # Create a dummy solution for testing
    dummy_solution = [
        7000, 0.001, 1.2, 0, 55,  # Walker 1
        8000, 0.001, 1.2, 0, 15,  # Walker 2
        10, 2, 1,  # S1, P1, F1
        10, 2, 1,  # S2, P2, F2
        13, 21, 34, 55  # Rover indices
    ]
    
    # Print solution summary
    print_solution_summary(dummy_solution)
    
    # Validate solution
    validation = validate_solution(dummy_solution)
    print_validation_results(validation)
    
    # Test hypervolume calculation
    test_points = [
        [0.5, 0.3],
        [0.6, 0.2],
        [0.4, 0.4]
    ]
    
    hv_score = combine_scores(test_points)
    print(f"Hypervolume score for test points: {hv_score:.4f}")
    
    print("\nUtility functions test completed!")