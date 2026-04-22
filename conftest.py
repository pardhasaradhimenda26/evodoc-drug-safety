"""
conftest.py — pytest configuration and shared fixtures.
Ensures the project root is on sys.path so all imports resolve correctly.
"""
import sys
from pathlib import Path

# Add project root to path so tests can import engine, cache, models
sys.path.insert(0, str(Path(__file__).parent))
