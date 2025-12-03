#!/usr/bin/env python3
"""
Quantum Communications Constellations Optimizer
Main entry point for the optimization problem
SPOC-2 Challenge: Quantum Communications Constellations
"""

import sys
import os
import argparse

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════╗
║     QUANTUM COMMUNICATIONS CONSTELLATIONS OPTIMIZER          ║
║                     SPOC-2 Challenge                         ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def setup_argument_parser():
    parser = argparse.ArgumentParser(
        description="Quantum Communications Constellations Optimizer for SPOC-2 Challenge"
    )
    parser.add_argument(
        'command',
        nargs='?',
        default='interactive',
        choices=['optimize', 'example', 'test', 'analyze', 'interactive']
    )
    return parser

def main():
    parser = setup_argument_parser()
    args = parser.parse_args()
    print_banner()

if __name__ == "__main__":
    main()
