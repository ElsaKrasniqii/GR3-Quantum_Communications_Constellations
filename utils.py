# utils.py - Version 1
import numpy as np
import json
from datetime import datetime
import warnings
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