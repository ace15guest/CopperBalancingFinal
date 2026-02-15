#!/usr/bin/env python
"""
Test Runner for Blur Filter Tests

This script provides a convenient way to run blur filter tests with various options.
"""

import sys
import subprocess
from pathlib import Path


def run_tests(args=None):
    """Run blur filter tests with pytest."""
    test_file = Path(__file__).parent / "test_array_operations" / "test_blur_filters.py"
    
    cmd = ["pytest", str(test_file), "-v"]
    
    if args:
        cmd.extend(args)
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 80)
    
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Main entry point."""
    print("=" * 80)
    print("Blur Filter Test Runner")
    print("=" * 80)
    print()
    
    if len(sys.argv) > 1:
        # Pass through any command line arguments
        return run_tests(sys.argv[1:])
    
    # Interactive menu
    print("Select test mode:")
    print("1. Run all tests")
    print("2. Run tests for specific quadrant")
    print("3. Run specific test class")
    print("4. Run with coverage report")
    print("5. Run in parallel (fast)")
    print("6. Custom pytest arguments")
    print()
    
    choice = input("Enter choice (1-6): ").strip()
    
    if choice == "1":
        return run_tests()
    
    elif choice == "2":
        print("\nAvailable quadrants: Q1, Q2, Q3, Q4, Global")
        quadrant = input("Enter quadrant: ").strip()
        return run_tests(["-k", quadrant])
    
    elif choice == "3":
        print("\nAvailable test classes:")
        print("  - TestBlurCall")
        print("  - TestBoxBlur")
        print("  - TestGaussianBlur")
        print("  - TestBlurComparison")
        print("  - TestBlurEdgeCases")
        test_class = input("Enter test class name: ").strip()
        return run_tests([f":::{test_class}"])
    
    elif choice == "4":
        return run_tests([
            "--cov=lib.array_operations.blur_filters",
            "--cov-report=html",
            "--cov-report=term"
        ])
    
    elif choice == "5":
        return run_tests(["-n", "auto"])
    
    elif choice == "6":
        args = input("Enter pytest arguments: ").strip().split()
        return run_tests(args)
    
    else:
        print("Invalid choice!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

# Made with Bob
