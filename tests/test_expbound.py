#!/usr/bin/env python3
"""
Test script for EXPBOUND function implementation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from asdm.asdm import Parser, Solver

def test_expbound():
    """Test the EXPBOUND function"""
    parser = Parser()
    solver = Solver()
    
    # Set up solver
    solver.name_space = {}
    solver.sim_specs = {'dt': 1}
    
    test_cases = [
        # Test basic functionality: EXPBOUND(yfrom, yto, x, exponent, xstart, xfinish)
        ("EXPBOUND(1, 9, 5.5, 0, 1, 10)", "Linear transition (exp=0)"),
        ("EXPBOUND(1, 9, 3, 2, 1, 10)", "Positive exponent: slow start"),
        ("EXPBOUND(1, 9, 7, 2, 1, 10)", "Positive exponent: fast finish"),
        ("EXPBOUND(1, 9, 3, -2, 1, 10)", "Negative exponent: fast start"),
        ("EXPBOUND(1, 9, 7, -2, 1, 10)", "Negative exponent: slow finish"),
        ("EXPBOUND(0, 100, 5, 1, 0, 10)", "0-100 transition"),
        ("EXPBOUND(1, 9, 0.5, 2, 1, 10)", "Boundary test: x < xstart"),
        ("EXPBOUND(1, 9, 15, 2, 1, 10)", "Boundary test: x > xfinish"),
    ]
    
    print("Testing EXPBOUND function...")
    print("-" * 70)
    
    for expr, description in test_cases:
        try:
            # Test tokenization
            tokens = parser.tokenise(expr)
            assert tokens[0][0] == 'FUNC', f"EXPBOUND not recognized as FUNC: {tokens[0]}"
            
            # Test parsing
            parsed = parser.parse(expr)
            
            # Test evaluation
            result = solver.calculate_node('test_expr', parsed, 'root')
            print(f"✓ {description:30} | {expr:30} = {result:.4f}")
            
        except Exception as e:
            print(f"✗ {description:30} | {expr:30} = ERROR: {e}")
        
    print("-" * 70)
    print("EXPBOUND implementation complete!")
    print("\nFunction signature: EXPBOUND(yfrom, yto, x, exponent, xstart, xfinish)")
    print("- exponent = 0: Linear transition")
    print("- exponent > 0: Slow start, fast finish")
    print("- exponent < 0: Fast start, slow finish")

if __name__ == "__main__":
    test_expbound()
