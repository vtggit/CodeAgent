#!/usr/bin/env python3
"""Integration test for webhook endpoint with queue system."""

import requests
import json
import hmac
import hashlib

# Test webhook endpoint
url = "http://localhost:8000/webhook/github"

# Create test payload
payload = {
    "action": "opened",
    "issue": {
        "number": 999,
        "title": "Test Queue Integration",
        "body": "Testing the new Redis/Memory queue system",
        "labels": [{"name": "test"}]
    }
}

payload_str = json.dumps(payload)
secret = "dev-webhook-secret"
signature = "sha256=" + hmac.new(secret.encode(), payload_str.encode(), hashlib.sha256).hexdigest()

print("Testing webhook endpoint with queue system...")
print()

# Test 1: Send webhook with valid signature
print("Test 1: Valid webhook with signature")
response = requests.post(
    url,
    json=payload,
    headers={
        "X-Hub-Signature-256": signature,
        "X-GitHub-Event": "issues"
    }
)

print(f"  Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"  Status: {data.get('status')}")
    print(f"  Message: {data.get('message')}")
    print(f"  Processing time: {data.get('processed_in_ms'):.2f}ms")
    print(f"  Job ID: {data.get('job_id')}")
    print("  ✓ PASSED")
else:
    print(f"  Response: {response.text}")
    print("  ✗ FAILED")

print()

# Test 2: Check queue status
print("Test 2: Queue status endpoint")
response = requests.get("http://localhost:8000/webhook/queue/status")
print(f"  Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"  Queue type: {data.get('queue_type')}")
    print(f"  Queue depth: {data.get('queue_depth')}")
    print(f"  Dead letter count: {data.get('dead_letter_count')}")
    print(f"  Healthy: {data.get('healthy')}")
    print("  ✓ PASSED")
else:
    print(f"  Response: {response.text}")
    print("  ✗ FAILED")

print()
print("="*50)
print("Integration test complete!")
