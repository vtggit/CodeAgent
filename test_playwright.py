#!/usr/bin/env python3
"""
Playwright test script for FastAPI server
Tests endpoints and captures screenshots
"""

import asyncio
import json
from playwright.async_api import async_playwright
from datetime import datetime
import os

# Create screenshots directory
os.makedirs("screenshots", exist_ok=True)

async def test_fastapi_endpoints():
    """Test FastAPI endpoints with Playwright"""

    base_url = "http://127.0.0.1:8000"
    results = {
        "timestamp": datetime.now().isoformat(),
        "endpoints_tested": [],
        "console_errors": [],
        "overall_status": "PASS"
    }

    async with async_playwright() as p:
        # Launch browser in headless mode
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 720}
        )
        page = await context.new_page()

        # Capture console logs and errors
        console_messages = []

        def handle_console(msg):
            console_messages.append({
                "type": msg.type,
                "text": msg.text
            })
            if msg.type in ["error", "warning"]:
                results["console_errors"].append({
                    "type": msg.type,
                    "message": msg.text
                })

        page.on("console", handle_console)

        # Test 1: Health Check Endpoint
        print("\n=== Test 1: Health Check Endpoint ===")
        try:
            response = await page.goto(f"{base_url}/health", wait_until="networkidle")

            # Get response data
            content = await page.content()

            # Take screenshot
            screenshot_path = "screenshots/01_health_check.png"
            await page.screenshot(path=screenshot_path, full_page=True)

            test_result = {
                "endpoint": "/health",
                "status_code": response.status,
                "success": response.ok,
                "screenshot": screenshot_path,
                "response_visible": "healthy" in content.lower()
            }

            results["endpoints_tested"].append(test_result)

            print(f"Status Code: {response.status}")
            print(f"Success: {response.ok}")
            print(f"Screenshot saved: {screenshot_path}")

            if not response.ok:
                results["overall_status"] = "FAIL"
                print("FAILED: Health check endpoint returned non-200 status")

        except Exception as e:
            print(f"ERROR: {str(e)}")
            results["overall_status"] = "FAIL"
            results["endpoints_tested"].append({
                "endpoint": "/health",
                "error": str(e),
                "success": False
            })

        # Test 2: Swagger UI Documentation
        print("\n=== Test 2: Swagger UI (API Documentation) ===")
        try:
            response = await page.goto(f"{base_url}/docs", wait_until="networkidle")

            # Wait for Swagger UI to load
            await page.wait_for_selector(".swagger-ui", timeout=10000)

            # Take screenshot
            screenshot_path = "screenshots/02_swagger_ui.png"
            await page.screenshot(path=screenshot_path, full_page=True)

            # Check for Swagger UI elements
            swagger_title = await page.query_selector(".title")
            swagger_loaded = swagger_title is not None

            test_result = {
                "endpoint": "/docs",
                "status_code": response.status,
                "success": response.ok and swagger_loaded,
                "screenshot": screenshot_path,
                "swagger_ui_loaded": swagger_loaded
            }

            results["endpoints_tested"].append(test_result)

            print(f"Status Code: {response.status}")
            print(f"Swagger UI Loaded: {swagger_loaded}")
            print(f"Screenshot saved: {screenshot_path}")

            if not response.ok or not swagger_loaded:
                results["overall_status"] = "FAIL"
                print("FAILED: Swagger UI did not load correctly")

        except Exception as e:
            print(f"ERROR: {str(e)}")
            results["overall_status"] = "FAIL"
            results["endpoints_tested"].append({
                "endpoint": "/docs",
                "error": str(e),
                "success": False
            })

        # Test 3: Root Endpoint
        print("\n=== Test 3: Root Endpoint (/) ===")
        try:
            response = await page.goto(f"{base_url}/", wait_until="networkidle")

            # Get response content
            content = await page.content()

            # Take screenshot
            screenshot_path = "screenshots/03_root_endpoint.png"
            await page.screenshot(path=screenshot_path, full_page=True)

            test_result = {
                "endpoint": "/",
                "status_code": response.status,
                "success": response.ok,
                "screenshot": screenshot_path,
                "has_content": len(content) > 0
            }

            results["endpoints_tested"].append(test_result)

            print(f"Status Code: {response.status}")
            print(f"Success: {response.ok}")
            print(f"Screenshot saved: {screenshot_path}")

            if not response.ok:
                results["overall_status"] = "FAIL"
                print("FAILED: Root endpoint returned non-200 status")

        except Exception as e:
            print(f"ERROR: {str(e)}")
            results["overall_status"] = "FAIL"
            results["endpoints_tested"].append({
                "endpoint": "/",
                "error": str(e),
                "success": False
            })

        # Close browser
        await browser.close()

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    print(f"\nOverall Status: {results['overall_status']}")
    print(f"Total Endpoints Tested: {len(results['endpoints_tested'])}")

    passed = sum(1 for t in results['endpoints_tested'] if t.get('success', False))
    failed = len(results['endpoints_tested']) - passed

    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    print("\nEndpoint Results:")
    for test in results['endpoints_tested']:
        status = "PASS" if test.get('success', False) else "FAIL"
        endpoint = test.get('endpoint', 'Unknown')
        print(f"  [{status}] {endpoint}")
        if 'error' in test:
            print(f"        Error: {test['error']}")

    if results['console_errors']:
        print(f"\nConsole Errors/Warnings: {len(results['console_errors'])}")
        for error in results['console_errors'][:10]:  # Show first 10
            print(f"  [{error['type'].upper()}] {error['message']}")
    else:
        print("\nNo console errors detected!")

    # Save results to JSON
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nDetailed results saved to: test_results.json")
    print("Screenshots saved to: screenshots/")

    return results

if __name__ == "__main__":
    results = asyncio.run(test_fastapi_endpoints())

    # Exit with appropriate code
    exit(0 if results['overall_status'] == 'PASS' else 1)
