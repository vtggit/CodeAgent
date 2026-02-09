#!/usr/bin/env python3
"""Simple script to test if FastAPI app can be imported."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.api.main import app
    print("✓ FastAPI app imported successfully")
    print(f"✓ App title: {app.title}")
    print(f"✓ App version: {app.version}")
    sys.exit(0)
except Exception as e:
    print(f"✗ Failed to import FastAPI app: {e}")
    sys.exit(1)
