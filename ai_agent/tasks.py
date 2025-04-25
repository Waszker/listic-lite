import openai
import settings
from openai import AsyncOpenAI

async def fetch_recipe_from_url(url: str) -> str:
    """Fetches recipe text from the given URL using OpenAI's web_search tool."""
    print(f"Fetching recipe from URL using OpenAI web_search: {url}")
    openai.api_key = settings.env_settings.openai_api_key
    prompt = (
        f"Extract the recipe (ingredients and instructions) from this page: {url}\n"
        "If no recipe is found, say so."
    )
    openai_client = AsyncOpenAI(api_key=settings.env_settings.openai_api_key)
    try:
        response = await openai_client.responses.create(
            model="gpt-4.1",
            tools=[{
                "type": "web_search_preview",
            }],
            input=prompt
        )
        return response.output_text
    except Exception as e:
        print(f"OpenAI API failed: {e}")
        return f"Error: Could not extract recipe using OpenAI web_search. Reason: {e}"
