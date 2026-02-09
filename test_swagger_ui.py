"""
Playwright test to verify webhook endpoints in Swagger UI
"""
import asyncio
from playwright.async_api import async_playwright
import json


async def test_swagger_ui():
    """Test the Swagger UI documentation for webhook endpoints"""

    async with async_playwright() as p:
        # Launch browser in headless mode (no display available)
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = await context.new_page()

        # Store console messages
        console_messages = []
        errors = []

        # Listen for console events
        page.on('console', lambda msg: console_messages.append({
            'type': msg.type,
            'text': msg.text
        }))

        # Listen for page errors
        page.on('pageerror', lambda error: errors.append(str(error)))

        try:
            print("\n=== Step 1: Navigate to Swagger UI ===")
            await page.goto('http://127.0.0.1:8000/docs', wait_until='networkidle')
            await asyncio.sleep(2)  # Wait for page to fully render

            # Take screenshot of the main page
            await page.screenshot(path='/home/vtg/Coding-Agent/generations/CodingAgent/screenshot_1_swagger_main.png', full_page=True)
            print("✓ Screenshot saved: screenshot_1_swagger_main.png")

            print("\n=== Step 2: Check for webhook endpoints ===")

            # Get page content
            content = await page.content()

            # Check for POST /webhook/github
            post_webhook_present = 'POST' in content and '/webhook/github' in content
            print(f"✓ POST /webhook/github documented: {post_webhook_present}")

            # Check for GET /webhook/queue/status
            get_status_present = 'GET' in content and '/webhook/queue/status' in content
            print(f"✓ GET /webhook/queue/status documented: {get_status_present}")

            print("\n=== Step 3: Expand POST /webhook/github endpoint ===")

            # Try to find and click the POST /webhook/github endpoint
            try:
                # Look for the operation element
                # Swagger UI uses specific class names for operations
                await page.wait_for_selector('.opblock-summary', timeout=5000)

                # Get all operation blocks
                operations = await page.query_selector_all('.opblock-summary')

                webhook_endpoint_found = False
                for op in operations:
                    text = await op.inner_text()
                    if 'POST' in text and '/webhook/github' in text:
                        print("✓ Found POST /webhook/github endpoint")
                        webhook_endpoint_found = True
                        # Click to expand
                        await op.click()
                        await asyncio.sleep(1)  # Wait for expansion animation
                        break

                if webhook_endpoint_found:
                    # Take screenshot of expanded endpoint
                    await page.screenshot(path='/home/vtg/Coding-Agent/generations/CodingAgent/screenshot_2_webhook_expanded.png', full_page=True)
                    print("✓ Screenshot saved: screenshot_2_webhook_expanded.png")

                    # Try to get endpoint details
                    try:
                        # Look for description, parameters, responses
                        description_element = await page.query_selector('.opblock-description-wrapper')
                        if description_element:
                            description = await description_element.inner_text()
                            print(f"\nEndpoint Description:\n{description}")
                    except Exception as e:
                        print(f"Note: Could not extract description: {e}")
                else:
                    print("⚠ POST /webhook/github endpoint not found in operation list")

            except Exception as e:
                print(f"⚠ Error expanding endpoint: {e}")

            print("\n=== Step 4: Check for GET /webhook/queue/status ===")

            # Try to find the GET endpoint
            try:
                operations = await page.query_selector_all('.opblock-summary')

                status_endpoint_found = False
                for op in operations:
                    text = await op.inner_text()
                    if 'GET' in text and '/webhook/queue/status' in text:
                        print("✓ Found GET /webhook/queue/status endpoint")
                        status_endpoint_found = True
                        # Click to expand
                        await op.click()
                        await asyncio.sleep(1)

                        # Take screenshot
                        await page.screenshot(path='/home/vtg/Coding-Agent/generations/CodingAgent/screenshot_3_status_expanded.png', full_page=True)
                        print("✓ Screenshot saved: screenshot_3_status_expanded.png")
                        break

                if not status_endpoint_found:
                    print("⚠ GET /webhook/queue/status endpoint not found in operation list")

            except Exception as e:
                print(f"⚠ Error checking status endpoint: {e}")

            print("\n=== Step 5: Console Messages & Errors ===")

            # Filter console messages
            console_errors = [msg for msg in console_messages if msg['type'] == 'error']
            console_warnings = [msg for msg in console_messages if msg['type'] == 'warning']

            if console_errors:
                print(f"\n⚠ Found {len(console_errors)} console errors:")
                for error in console_errors[:10]:  # Show first 10
                    print(f"  - {error['text']}")
            else:
                print("✓ No console errors found")

            if console_warnings:
                print(f"\n⚠ Found {len(console_warnings)} console warnings:")
                for warning in console_warnings[:5]:  # Show first 5
                    print(f"  - {warning['text']}")
            else:
                print("✓ No console warnings found")

            if errors:
                print(f"\n⚠ Found {len(errors)} page errors:")
                for error in errors:
                    print(f"  - {error}")
            else:
                print("✓ No page errors found")

            print("\n=== Step 6: Extract all endpoints ===")

            # Get all visible endpoints
            try:
                operations = await page.query_selector_all('.opblock-summary')
                print(f"\n✓ Found {len(operations)} total endpoints:")

                for op in operations:
                    text = await op.inner_text()
                    # Clean up the text
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    if lines:
                        print(f"  - {' '.join(lines)}")
            except Exception as e:
                print(f"⚠ Error extracting endpoints: {e}")

            # In headless mode, no need to keep browser open
            print("\n=== Test completed successfully ===")

        except Exception as e:
            print(f"\n❌ Error during test: {e}")
            # Take error screenshot
            await page.screenshot(path='/home/vtg/Coding-Agent/generations/CodingAgent/screenshot_error.png', full_page=True)
            print("✓ Error screenshot saved: screenshot_error.png")

        finally:
            await browser.close()
            print("\n=== Test Complete ===")


if __name__ == "__main__":
    asyncio.run(test_swagger_ui())
