#!/usr/bin/env python3
"""
Campus Delivery Robot - AI Fundamentals Final Project
Main Entry Point
Student: Nurzhas Amangeldi

Demonstrates four integrated algorithms across three course topics:
  1. HMM  (Probabilistic Reasoning & Sequential Inference) -> Robot localization with noisy 2GIS sensors
  2. Bayes (Probabilistic Reasoning)                       -> Target room identification from verbal clues
  3. A*   (Search & Heuristics)                            -> Shortest obstacle-free path planning
  4. MDP  (Sequential Decision Making)                     -> Move vs. Relocalize decision under uncertainty

Usage:
  python campus_robot.py
"""

from demo import show, SCENARIOS


def main():
    print("=" * 66)
    print("  CAMPUS DELIVERY ROBOT: AUTONOMOUS INDOOR NAVIGATION")
    print("  AI Fundamentals Course Project - QAIRU")
    print("  Author: Nurzhas Amangeldi")
    print("=" * 66)
    for title, readings in SCENARIOS:
        show(title, readings)


if __name__ == "__main__":
    main()
