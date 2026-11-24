#!/usr/bin/env python3
"""
Test runner for Campus Delivery Robot.
Runs with zero external dependencies (no pytest required).
Usage: python run_tests.py
"""
import sys
import time
import inspect
from tests import test_robot


def main():
    print("=" * 66)
    print(" Running Unit Test Suite: Campus Delivery Robot")
    print("=" * 66)

    test_funcs = [
        (name, func)
        for name, func in inspect.getmembers(test_robot, inspect.isfunction)
        if name.startswith("test_")
    ]

    passed = 0
    failed = 0
    t0 = time.time()

    for name, func in test_funcs:
        try:
            func()
            print(f" [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f" [FAIL] {name}: {e}")
            failed += 1

    dt = time.time() - t0
    print("-" * 66)
    print(f"Summary: {passed} passed, {failed} failed in {dt:.3f}s")
    print("=" * 66)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
