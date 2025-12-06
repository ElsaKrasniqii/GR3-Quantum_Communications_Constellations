import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ======================================================================
# HYPERVOLUME SCORING (SAFE VERSION)
# ======================================================================
def compute_hypervolume_score(points, ref_point=None):
    """
    Computes hypervolume score for minimization problems.
    Returns a negative value (lower is better).
    """
    try:
        # Try to use pygmo for accurate hypervolume
        import pygmo as pg
        
        pts = np.array(points)
        
        # Check input validity
        if pts.ndim != 2 or pts.shape[1] < 2:
            return 0.0
            
        # Use first 2 objectives only
        pts_2d = pts[:, :2]
        
        # Remove invalid points
        valid_mask = ~(np.any(np.isnan(pts_2d), axis=1) | np.any(np.isinf(pts_2d), axis=1))
        pts_2d = pts_2d[valid_mask]
        
        if len(pts_2d) < 2:
            # Fallback for few points
            if len(pts_2d) == 1:
                return -float(np.sum(pts_2d[0]))
            return 0.0
        
        # Set reference point if not provided
        if ref_point is None:
            ref_point = np.array([1.5, 1.5])  # Conservative reference
        
        try:
            hv = pg.hypervolume(pts_2d)
            hv_value = hv.compute(ref_point)
            return -hv_value  # Negative because we minimize
        except Exception as e:
            # Fallback to simple approximation
            print(f"Warning: pygmo hypervolume failed, using approximation: {e}")
            return -_approx_hypervolume(pts_2d, ref_point)
            
    except ImportError:
        # Fallback without pygmo
        return _approx_hypervolume(points, ref_point)
    except Exception as e:
        print(f"Warning: Hypervolume computation failed: {e}")
        return 0.0


def _approx_hypervolume(points, ref_point=None):
    """Approximate hypervolume without pygmo."""
    try:
        pts = np.array(points)
        if pts.ndim != 2 or pts.shape[1] < 2:
            return 0.0
            
        pts_2d = pts[:, :2]
        
        # Remove invalid points
        valid_mask = ~(np.any(np.isnan(pts_2d), axis=1) | np.any(np.isinf(pts_2d), axis=1))
        pts_2d = pts_2d[valid_mask]
        
        if len(pts_2d) == 0:
            return 0.0
            
        if ref_point is None:
            ref_point = np.max(pts_2d, axis=0) * 1.2
            
        # Simple hypervolume approximation
        # Sort by first objective
        sorted_pts = pts_2d[np.argsort(pts_2d[:, 0])]
        
        hv = 0.0
        prev_x = ref_point[0]
        
        for pt in sorted_pts:
            if pt[0] <= ref_point[0] and pt[1] <= ref_point[1]:
                width = prev_x - pt[0]
                height = ref_point[1] - pt[1]
                hv += width * height
                prev_x = pt[0]
                
        return -hv
        
    except Exception:
        return 0.0


# ======================================================================
# SOLUTION MANAGEMENT
# ======================================================================
def save_solutions(solutions, filename="solutions_backup.json", metadata=None):
    """
    Save solutions to a JSON file.
    
    Args:
        solutions: List of solution dictionaries
        filename: Output filename
        metadata: Additional metadata to save
    """
    try:
        out = []
        for sol in solutions:
            block = {
                "x": sol["x"].tolist() if hasattr(sol["x"], 'tolist') else sol["x"],
                "fitness": sol["fitness"],
                "feasible": bool(sol["feasible"])
            }
            
            # Add optional fields
            if "constraint_info" in sol:
                block["constraint_info"] = sol["constraint_info"]
            if "crowding_distance" in sol:
                block["crowding_distance"] = sol["crowding_distance"]
            if "penalty" in sol:
                block["penalty"] = sol["penalty"]
                
            out.append(block)

        meta = metadata or {}

        data = {
            "timestamp": datetime.now().isoformat(),
            "number_of_solutions": len(solutions),
            "solution_dimension": len(out[0]["x"]) if out else 0,
            "metadata": meta,
            "solutions": out
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        print(f"✓ Saved {len(solutions)} solutions → {filename}")
        return True

    except Exception as e:
        print(f"✗ Error saving solutions: {e}")
        return False




def load_solutions(filename="solutions_backup.json", verbose=True):
    """
    Load solutions from a JSON file.
    
    Args:
        filename: Input filename
        verbose: Whether to print status messages
        
    Returns:
        List of solution dictionaries
    """
    try:
        with open(filename, "r") as f:
            data = json.load(f)

        sols = []
        for s in data["solutions"]:
            entry = {
                "x": np.array(s["x"]),
                "fitness": s["fitness"],
                "feasible": bool(s["feasible"])
            }
            
            # Load optional fields
            if "constraint_info" in s:
                entry["constraint_info"] = s["constraint_info"]
            if "crowding_distance" in s:
                entry["crowding_distance"] = s["crowding_distance"]
            if "penalty" in s:
                entry["penalty"] = s["penalty"]
                
            sols.append(entry)

        if verbose:
            print(f"✓ Loaded {len(sols)} solutions from {filename}")
            print(f"  Timestamp: {data['timestamp']}")
            print(f"  Dimension: {data['solution_dimension']}")

        return sols

    except Exception as e:
        print(f"✗ Failed to load solutions: {e}")
        return []


# ======================================================================
# SOLUTION ANALYSIS
# ======================================================================
def print_solution_summary(x, udp_instance=None):
    """
    Print a human-readable summary of a solution.
    
    Args:
        x: Solution vector (20 parameters)
        udp_instance: Optional UDP instance for analysis
    """
    if len(x) != 20:
        print(f"✗ Expected 20 parameters, got {len(x)}.")
        return

    try:
        # Unpack parameters
        (
            a1, e1, i1, w1, eta1,
            a2, e2, i2, w2, eta2,
            S1, P1, F1, S2, P2, F2,
            r1, r2, r3, r4
        ) = x
        
        # Convert to integers for display
        S1_int, P1_int, F1_int = int(S1), int(P1), int(F1)
        S2_int, P2_int, F2_int = int(S2), int(P2), int(F2)
        
        # Calculate derived values
        alt1 = a1 - 6371  # Approximate altitude
        alt2 = a2 - 6371
        total_sats = S1_int * P1_int + S2_int * P2_int

    except Exception as e:
        print(f"✗ Could not unpack solution vector: {e}")
        return

    print("\n" + "=" * 60)
    print("SOLUTION SUMMARY")
    print("=" * 60)

    print("\n📡 WALKER 1 CONSTELLATION:")
    print(f"  Semi-major axis: {a1:.1f} km (Altitude: {alt1:.1f} km)")
    print(f"  Eccentricity: {e1:.6f}")
    print(f"  Inclination: {np.degrees(i1):.2f}°")
    print(f"  Argument of perigee: {np.degrees(w1):.2f}°")
    print(f"  Efficiency (η): {eta1:.2f}")
    print(f"  Walker parameters: S={S1_int}, P={P1_int}, F={F1_int}")
    print(f"  Total satellites: {S1_int * P1_int}")

    print("\n📡 WALKER 2 CONSTELLATION:")
    print(f"  Semi-major axis: {a2:.1f} km (Altitude: {alt2:.1f} km)")
    print(f"  Eccentricity: {e2:.6f}")
    print(f"  Inclination: {np.degrees(i2):.2f}°")
    print(f"  Argument of perigee: {np.degrees(w2):.2f}°")
    print(f"  Efficiency (η): {eta2:.2f}")
    print(f"  Walker parameters: S={S2_int}, P={P2_int}, F={F2_int}")
    print(f"  Total satellites: {S2_int * P2_int}")

    print(f"\n📊 TOTAL CONSTELLATION:")
    print(f"  Satellites: {total_sats}")
    print(f"  Total efficiency cost: {eta1 * S1_int * P1_int + eta2 * S2_int * P2_int:.1f}")

    print("\n📍 ROVER POSITIONS (indices):")
    print(f"  Rover 1: index {int(r1)}")
    print(f"  Rover 2: index {int(r2)}")
    print(f"  Rover 3: index {int(r3)}")
    print(f"  Rover 4: index {int(r4)}")

    # Analyze with UDP if provided
    if udp_instance:
        try:
            # Get fitness
            fitness = udp_instance.fitness(x)
            
            print("\n🎯 FITNESS VALUES:")
            print(f"  J1 (Communication cost): {fitness[0]:.6f}")
            print(f"  J2 (Infrastructure cost): {fitness[1]:.6f}")
            
            # Get detailed analysis
            if hasattr(udp_instance, 'analyze_solution'):
                analysis = udp_instance.analyze_solution(x)
                
                print("\n📏 CONSTRAINT ANALYSIS:")
                print(f"  Min rover distance: {analysis['distances']['rover_distance']:.1f} km")
                print(f"  Required: ≥ {analysis['distances']['min_rover_dist']} km")
                print(f"  Violated: {'YES' if analysis['constraints']['rover_violated'] else 'NO'}")
                
                print(f"  Min satellite distance: {analysis['distances']['satellite_distance']:.1f} km")
                print(f"  Required: ≥ {analysis['distances']['min_sat_dist']} km")
                print(f"  Violated: {'YES' if analysis['constraints']['sat_violated'] else 'NO'}")
                
                print(f"\n✓ FEASIBLE: {not (analysis['constraints']['rover_violated'] or analysis['constraints']['sat_violated'])}")
                
                # Show penalty impact
                if 'fitness' in analysis and 'without_penalty' in analysis['fitness']:
                    f_no_penalty = analysis['fitness']['without_penalty']
                    penalty_effect = fitness[0] - f_no_penalty[0]
                    if abs(penalty_effect) > 1e-6:
                        print(f"\n⚠ PENALTY EFFECT: {penalty_effect:.6f} added to objectives")
            
        except Exception as e:
            print(f"\n⚠ Could not analyze with UDP: {e}")

    print("=" * 60 + "\n")


# ======================================================================
# SOLUTION VALIDATION
# ======================================================================
def validate_solution(x, udp_instance=None):
    """
    Validate a solution vector.
    
    Args:
        x: Solution vector (20 parameters)
        udp_instance: Optional UDP instance for detailed validation
        
    Returns:
        Dictionary with validation results
    """
    v = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "info": []
    }

    # Basic length check
    if len(x) != 20:
        v["errors"].append(f"Expected 20 parameters, got {len(x)}")
        v["valid"] = False
        return v

    try:
        # Unpack parameters
        (
            a1, e1, i1, w1, eta1,
            a2, e2, i2, w2, eta2,
            S1, P1, F1, S2, P2, F2,
            r1, r2, r3, r4
        ) = x
        
        # Check basic ranges
        bounds_check = [
            ("a1", a1, 6871, 22371),
            ("a2", a2, 6871, 22371),
            ("e1", e1, 0, 0.1),
            ("e2", e2, 0, 0.1),
            ("i1", i1, 0, np.pi),
            ("i2", i2, 0, np.pi),
            ("eta1", eta1, 0, 200),
            ("eta2", eta2, 0, 200),
        ]
        
        for name, val, low, high in bounds_check:
            if val < low or val > high:
                v["warnings"].append(f"{name} = {val:.2f} outside typical range [{low}, {high}]")
        
        # Check integer parameters
        int_params = [
            ("S1", S1, 1, 20),
            ("P1", P1, 1, 20),
            ("F1", F1, 0, 20),
            ("S2", S2, 1, 20),
            ("P2", P2, 1, 20),
            ("F2", F2, 0, 20),
        ]
        
        for name, val, low, high in int_params:
            if not float(val).is_integer():
                v["warnings"].append(f"{name} should be integer (got {val})")
            elif val < low or val > high:
                v["warnings"].append(f"{name} = {int(val)} outside typical range [{low}, {high}]")
        
        # Check rover indices
        for i, idx in enumerate([r1, r2, r3, r4], 1):
            if not float(idx).is_integer():
                v["warnings"].append(f"Rover {i} index should be integer (got {idx})")
        
        # Get bounds from UDP if available
        if udp_instance:
            try:
                bounds = udp_instance.get_bounds()
                lower, upper = bounds
                
                for i, (val, low, high) in enumerate(zip(x, lower, upper)):
                    if val < low or val > high:
                        v["errors"].append(f"Parameter {i+1} = {val:.4f} outside bounds [{low:.2f}, {high:.2f}]")
                        v["valid"] = False
                
                # Calculate fitness
                try:
                    fitness = udp_instance.fitness(x)
                    v["fitness"] = fitness
                    v["info"].append(f"Fitness: J1={fitness[0]:.6f}, J2={fitness[1]:.6f}")
                    
                    # Detailed analysis if available
                    if hasattr(udp_instance, 'analyze_solution'):
                        analysis = udp_instance.analyze_solution(x)
                        v["analysis"] = analysis
                        
                        # Check constraints
                        rover_violated = analysis['constraints']['rover_violated']
                        sat_violated = analysis['constraints']['sat_violated']
                        v["feasible"] = not (rover_violated or sat_violated)
                        
                        if rover_violated:
                            v["warnings"].append(f"Rover distance constraint violated: {analysis['distances']['rover_distance']:.1f} km < {analysis['distances']['min_rover_dist']} km")
                        if sat_violated:
                            v["warnings"].append(f"Satellite distance constraint violated: {analysis['distances']['satellite_distance']:.1f} km < {analysis['distances']['min_sat_dist']} km")
                            
                except Exception as e:
                    v["errors"].append(f"Fitness calculation failed: {e}")
                    v["valid"] = False
                    
            except Exception as e:
                v["warnings"].append(f"Could not validate with UDP: {e}")
        
        # Add parameter count info
        v["info"].append(f"Total satellites: {int(S1)*int(P1) + int(S2)*int(P2)}")
        
    except Exception as e:
        v["errors"].append(f"Validation failed: {e}")
        v["valid"] = False

    return v


def print_validation_results(v):
    """
    Print validation results in a readable format.
    
    Args:
        v: Validation dictionary from validate_solution()
    """
    print("\n" + "="*60)
    print("VALIDATION REPORT")
    print("="*60)

    print(f"✓ VALID: {'YES' if v['valid'] else 'NO'}")

    if "fitness" in v:
        f1, f2 = v["fitness"]
        print(f"📊 FITNESS: J1={f1:.6f}, J2={f2:.6f}")

    if "feasible" in v:
        print(f"🎯 FEASIBLE: {'YES' if v['feasible'] else 'NO'}")

    if v["info"]:
        print("\n📝 INFORMATION:")
        for info in v["info"]:
            print(f"  • {info}")

    if v["warnings"]:
        print(f"\n⚠ WARNINGS ({len(v['warnings'])}):")
        for warning in v["warnings"]:
            print(f"  • {warning}")

    if v["errors"]:
        print(f"\n❌ ERRORS ({len(v['errors'])}):")
        for error in v["errors"]:
            print(f"  • {error}")

    print("="*60 + "\n")


# ======================================================================
# DEPENDENCY CHECK
# ======================================================================
def check_dependencies(verbose=True):
    """
    Check if all required dependencies are installed.
    
    Returns:
        Tuple of (all_ok, critical_ok, status_dict)
    """
    modules = {
        "numpy": ("Numerical computations", True),
        "scipy": ("Scientific computing", True),
        "sgp4": ("Satellite propagation", True),
        "networkx": ("Graph algorithms", True),
        "pykep": ("Astrodynamics", True),
        "pygmo": ("Optimization library", False),  # Non-critical for basic operation
        "matplotlib": ("Plotting", False),
        "pandas": ("Data analysis", False),
    }

    status = {}

    if verbose:
        print("\n" + "="*60)
        print("DEPENDENCY CHECK")
        print("="*60)

    for mod, (description, critical) in modules.items():
        try:
            imported = __import__(mod)
            version = getattr(imported, "__version__", "unknown")
            status[mod] = (True, critical, version, description)
        except ImportError:
            status[mod] = (False, critical, None, description)

    if verbose:
        print("\nREQUIRED MODULES:")
        for mod, (ok, critical, version, desc) in status.items():
            if critical:
                symbol = "✓" if ok else "✗"
                print(f"  {symbol} {mod:12} {desc}")
                if ok and version:
                    print(f"      Version: {version}")

        print("\nOPTIONAL MODULES:")
        for mod, (ok, critical, version, desc) in status.items():
            if not critical:
                symbol = "✓" if ok else "○"
                print(f"  {symbol} {mod:12} {desc}")
                if ok and version:
                    print(f"      Version: {version}")

        print("="*60)

    # Check critical modules
    critical_ok = all(ok for mod, (ok, critical, _, _) in status.items() if critical)
    all_ok = all(ok for mod, (ok, _, _, _) in status.items())

    if verbose:
        if critical_ok:
            print("✓ All critical dependencies are installed.")
        else:
            print("✗ Some critical dependencies are missing!")

    return all_ok, critical_ok, status


# ======================================================================
# QUICK TEST FUNCTION
# ======================================================================
def run_quick_test():
    """
    Run a quick test to verify utilities are working.
    """
    print("Running quick test of utilities...")
    
    # Test 1: Dependency check
    print("\n1. Checking dependencies:")
    all_ok, critical_ok, status = check_dependencies(verbose=False)
    print(f"   Critical modules OK: {critical_ok}")
    print(f"   All modules OK: {all_ok}")
    
    # Test 2: Create a dummy solution
    print("\n2. Testing solution validation:")
    dummy_solution = [7000, 0.001, 1.2, 0, 40,
                      8200, 0.001, 1.2, 0, 30,
                      10, 2, 1, 8, 3, 1,
                      5, 10, 15, 20]
    
    validation = validate_solution(dummy_solution)
    print(f"   Solution valid: {validation['valid']}")
    
    if validation['warnings']:
        print(f"   Warnings: {len(validation['warnings'])}")
    if validation['errors']:
        print(f"   Errors: {len(validation['errors'])}")
    
    # Test 3: Hypervolume calculation
    print("\n3. Testing hypervolume calculation:")
    dummy_points = np.array([[0.5, 0.6], [0.4, 0.7], [0.6, 0.5]])
    hv_score = compute_hypervolume_score(dummy_points)
    print(f"   Hypervolume score: {hv_score:.6f}")
    
    print("\n" + "="*60)
    print("QUICK TEST COMPLETED")
    print("="*60)
    
    return all_ok and validation['valid']


# ======================================================================
# MAIN (for testing)
# ======================================================================
if __name__ == "__main__":
    print("Testing utilities module...")
    success = run_quick_test()
    
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n⚠ Some tests failed. Check output above.")