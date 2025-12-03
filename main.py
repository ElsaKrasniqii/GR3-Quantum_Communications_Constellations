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

def main():
    parser = setup_argument_parser()
    args = parser.parse_args()
    print_banner()

    if args.command not in ['test', 'interactive']:
        if not check_environment():
            sys.exit(1)

if __name__ == "__main__":
    main()
