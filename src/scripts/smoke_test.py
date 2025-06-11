# smoke_test.py
import asyncio, time
from playwright.async_api import async_playwright

URL = "https://www.jenius.com/faq/mata-uang-asing/tentang-mata-uang-asing?locale=en"

async def main():
    async with async_playwright() as p:
        br = await p.chromium.launch(headless=True, timeout=0)
        page = await br.new_page()
        t0 = time.perf_counter()
        print("⏳ loading page …")
        await page.goto(URL, timeout=45_000)
        await page.wait_for_selector(".accordion-item", timeout=10_000)
        print("✅ selector found after", round(time.perf_counter()-t0, 2), "s")
        html = await page.content()
        print("HTML length :", len(html))
        await br.close()

asyncio.run(main())
