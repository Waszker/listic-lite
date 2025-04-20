import asyncio
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright


async def fetch_recipe_from_url(url: str) -> str:
    """Fetches recipe text from the given URL using Playwright to handle dynamic content."""
    print(f"Fetching recipe from URL with Playwright: {url}")
    page_content = ""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            except Exception as e:
                 print(f"Playwright page.goto timed out or failed for {url}: {e}")
                 await browser.close()
                 return f"Error: Could not fetch content from {url}. Reason: Page load failed or timed out."

            await asyncio.sleep(2)
            page_content = await page.content()
            await browser.close()
    except Exception as e:
        print(f"Playwright failed to fetch {url}: {e}")
        return f"Error: Could not fetch content from {url}. Reason: {e}"

    if not page_content:
        return f"Error: No content fetched from {url}"

    soup = BeautifulSoup(page_content, "html.parser")
    recipe_content = (
        soup.find(class_=lambda x: x and "recipe" in x.lower())
        or soup.find(id=lambda x: x and "recipe" in x.lower())
        or soup.find("article")
        or soup.body
    )
    if not recipe_content:
        print(f"Specific recipe container not found for {url}, falling back to full body text.")
        return soup.get_text(separator="\n", strip=True)

    return recipe_content.get_text(separator="\n", strip=True)