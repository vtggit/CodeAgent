#!/usr/bin/env python3
"""Test router import."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.api.webhooks import router
    print("✓ Router imported successfully")
    print(f"Router prefix: {router.prefix}")
    print(f"Number of routes: {len(router.routes)}")
    for route in router.routes:
        print(f"  - {route.methods} {route.path}")
except Exception as e:
    print(f"✗ Failed to import router: {e}")
    import traceback
    traceback.print_exc()
