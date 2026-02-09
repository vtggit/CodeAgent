import asyncio
from playwright.async_api import async_playwright
from datetime import datetime
import json

class FastAPIVerificationTests:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.console_errors = []
        self.console_warnings = []
        
    def on_console(self, msg):
        data = {"type": msg.type, "text": msg.text}
        if msg.type == "error":
            self.console_errors.append(data)
            print(f"Console Error: {msg.text}")
        elif msg.type == "warning":
            self.console_warnings.append(data)
            
    async def test_swagger_ui(self, page):
        print("="*60)
        print("Test 1: Swagger UI Navigation and Screenshot")
        print("="*60)
        try:
            print(f"Navigating to {self.base_url}/docs...")
            response = await page.goto(f"{self.base_url}/docs", wait_until="networkidle")
            status = response.status if response else "N/A"
            print(f"Status: {status}")
            
            await page.wait_for_selector(".swagger-ui", timeout=5000)
            print("Swagger UI loaded successfully")
            
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"screenshots/swagger_ui_{ts}.png"
            await page.screenshot(path=screenshot_path, full_page=True)
            print(f"Screenshot saved: {screenshot_path}")
            
            title = await page.title()
            print(f"Page title: {title}")
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False
            
    async def test_health(self, page):
        print()
        print("="*60)
        print("Test 2: Health Endpoint Verification")
        print("="*60)
        try:
            print(f"Checking {self.base_url}/health")
            response = await page.goto(f"{self.base_url}/health")
            print(f"Status: {response.status}")
            
            body = await page.evaluate("() => document.body.textContent")
            print(f"Response: {body.strip()}")
            
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"screenshots/health_{ts}.png"
            await page.screenshot(path=screenshot_path)
            print(f"Screenshot saved: {screenshot_path}")
            return response.ok
        except Exception as e:
            print(f"Error: {e}")
            return False
            
    async def run_tests(self):
        print()
        print("="*60)
        print("FastAPI Application Verification Tests")
        print("="*60)
        print(f"Base URL: {self.base_url}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        print()
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            page.on("console", self.on_console)
            
            results = {}
            results["swagger_ui"] = await self.test_swagger_ui(page)
            results["health_endpoint"] = await self.test_health(page)
            
            await browser.close()
            
            print()
            print("="*60)
            print("Test Summary")
            print("="*60)
            for name, passed in results.items():
                status = "PASSED" if passed else "FAILED"
                print(f"  {name:20} {status}")
            
            print(f"")
            print(f"Console Errors: {len(self.console_errors)}")
            if self.console_errors:
                for err in self.console_errors:
                    print(f"  - {err['text']}")
            else:
                print("  No console errors detected")
                
            print("="*60)
            all_passed = all(results.values())
            no_errors = len(self.console_errors) == 0
            if all_passed and no_errors:
                print("ALL TESTS PASSED")
            else:
                print("SOME TESTS FAILED")
            print("="*60)
            
            return all_passed and no_errors

async def main():
    tester = FastAPIVerificationTests()
    success = await tester.run_tests()
    exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())
