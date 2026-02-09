#!/usr/bin/env python3
"""Test script for webhook endpoint."""
import json
import hmac
import hashlib
import time
import requests

# Load test payload
with open("test_webhook_payload.json") as f:
    payload = json.load(f)

payload_bytes = json.dumps(payload).encode()

# Generate signature
secret = "dev-webhook-secret"
mac = hmac.new(secret.encode(), msg=payload_bytes, digestmod=hashlib.sha256)
signature = f"sha256={mac.hexdigest()}"

# Test 1: Valid webhook with signature
print("Test 1: Valid webhook with signature")
start = time.time()
response = requests.post(
    "http://127.0.0.1:8000/webhook/github",
    json=payload,
    headers={
        "X-Hub-Signature-256": signature,
        "X-GitHub-Event": "issues",
        "Content-Type": "application/json",
    }
)
elapsed = (time.time() - start) * 1000
print(f"  Status: {response.status_code}")
print(f"  Response: {response.json()}")
print(f"  Time: {elapsed:.2f}ms")
print()

# Test 2: Invalid signature
print("Test 2: Invalid signature (should return 401)")
response = requests.post(
    "http://127.0.0.1:8000/webhook/github",
    json=payload,
    headers={
        "X-Hub-Signature-256": "sha256=invalid_signature",
        "X-GitHub-Event": "issues",
        "Content-Type": "application/json",
    }
)
print(f"  Status: {response.status_code}")
print(f"  Response: {response.json()}")
print()

# Test 3: Non-issue event (should be ignored)
print("Test 3: Non-issue event (should be ignored)")
response = requests.post(
    "http://127.0.0.1:8000/webhook/github",
    json=payload,
    headers={
        "X-Hub-Signature-256": signature,
        "X-GitHub-Event": "push",
        "Content-Type": "application/json",
    }
)
print(f"  Status: {response.status_code}")
print(f"  Response: {response.json()}")
print()

# Test 4: Check queue status
print("Test 4: Queue status")
response = requests.get("http://127.0.0.1:8000/webhook/queue/status")
print(f"  Status: {response.status_code}")
print(f"  Response: {response.json()}")
print()

print("All tests completed!")
