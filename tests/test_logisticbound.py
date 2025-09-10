#!/usr/bin/env python3
"""
Test script for LOGISTICBOUND function implementation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from asdm.asdm import Parser, Solver

def test_logisticbound():
    """Test the LOGISTICBOUND function"""
    parser = Parser()
    solver = Solver()
    
    # Set up solver
    solver.name_space = {}
    solver.sim_specs = {'dt': 1}
    
    test_cases = [
        # Test basic functionality: LOGISTICBOUND(yfrom, yto, x, xmiddle, speed)
        "LOGISTICBOUND(1, 9, 5, 5, 1)",        # x at middle, should be around 5
        "LOGISTICBOUND(1, 9, 0, 5, 1)",        # x before middle, should be closer to 1
        "LOGISTICBOUND(1, 9, 10, 5, 1)",       # x after middle, should be closer to 9
        "LOGISTICBOUND(1, 9, 5, 5, 10)",       # higher speed, steeper curve
        "LOGISTICBOUND(0, 1, 0, 0, 1)",        # simple 0-1 transition
    ]
    
    print("Testing LOGISTICBOUND function...")
    print("-" * 60)
    
    for expr in test_cases:
        try:
            # Test tokenization
            tokens = parser.tokenise(expr)
            print(f"✓ Tokenized: {expr}")
            
            # Test parsing
            parsed = parser.parse(expr)
            print(f"✓ Parsed: {expr}")
            
            # Test evaluation
            result = solver.calculate_node('test_expr', parsed, 'root')
            print(f"✓ Result: {expr} = {result:.4f}")
            
        except Exception as e:
            print(f"✗ ERROR in {expr}: {e}")
            import traceback
            traceback.print_exc()
        
        print()

if __name__ == "__main__":
    test_logisticbound()
