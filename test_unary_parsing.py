#!/usr/bin/env python3
"""
Test script to verify unary minus/plus parsing implementation
"""

import sys
import os

# Add the src directory to the path so we can import asdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from asdm.asdm import Parser, Solver

def test_unary_operators():
    """Test unary operators parsing"""
    parser = Parser()
    solver = Solver()
    
    test_cases = [
        # Expression, Expected Result, Description
        ("-5", -5, "Simple negative number"),
        ("+5", 5, "Simple positive number"),
        ("--5", 5, "Double negative"),
        ("+-5", -5, "Plus then minus"),
        ("-+5", -5, "Minus then plus"),
        ("1 - 2", -1, "Binary minus"),
        ("1 + -2", -1, "Binary plus with unary minus"),
        ("(-5)", -5, "Negative in parentheses"),
        ("-(5)", -5, "Unary minus applied to parentheses"),
        ("-(-5)", 5, "Unary minus applied to negative"),
        ("2 * -3", -6, "Multiplication with unary minus"),
        ("-2 * 3", -6, "Unary minus with multiplication"),
    ]
    
    print("Testing unary operator parsing...")
    print("-" * 50)
    
    for expr, expected, description in test_cases:
        # Parse the expression
        parsed = parser.parse(expr)
        
        # Create a simple namespace for evaluation
        solver.name_space = {}
        solver.sim_specs = {'dt': 1}
        
        # Evaluate the parsed expression (correct parameter order)
        result = solver.calculate_node('test_expr', parsed, 'root')
        
        # Check if result matches expected (using pytest assertion)
        assert abs(result - expected) < 1e-10, f"Expression '{expr}' returned {result}, expected {expected} ({description})"
        print(f"✓ {expr:12} = {result:6} ({description})")
    
    print("-" * 50)
    print("All tests passed! ✓")

def test_tokenization():
    """Test tokenization of expressions with unary operators"""
    parser = Parser()
    
    test_cases = [
        ("-5", [['MINUS', '-'], ['NUMBER', '5']]),
        ("+5", [['PLUS', '+'], ['NUMBER', '5']]),
        ("1-2", [['NUMBER', '1'], ['MINUS', '-'], ['NUMBER', '2']]),
        ("1+-2", [['NUMBER', '1'], ['PLUS', '+'], ['MINUS', '-'], ['NUMBER', '2']]),
    ]
    
    print("\nTesting tokenization...")
    print("-" * 50)
    
    for expr, expected in test_cases:
        tokens = parser.tokenise(expr)
        assert tokens == expected, f"Expression '{expr}' tokenized as {tokens}, expected {expected}"
        print(f"✓ {expr:8} -> {tokens}")
    
    print("-" * 50)
    print("Tokenization tests passed! ✓")

if __name__ == "__main__":
    try:
        test_tokenization()
        test_unary_operators()
        print("\n🎉 All tests passed!")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
