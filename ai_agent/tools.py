import asyncio
from typing import List
from collections import defaultdict
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from ai_agent.config import llm
from ai_agent.data_models import Ingredient, IngredientsOutput, IngredientNamesOutput


@tool
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


@tool
async def extract_ingredients(recipe_text: str) -> IngredientsOutput:
    """Extracts ingredients from the recipe text"""
    parser = JsonOutputParser(pydantic_object=IngredientsOutput)

    prompt = PromptTemplate(
        template="""
Here is the recipe text. Extract the list of ingredients (translate to Polish name if needed) with their quantities and units.
{format_instructions}

Recipe text:
{recipe_text}
""",
        input_variables=["recipe_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    result = await chain.ainvoke({"recipe_text": recipe_text})
    return IngredientsOutput(**result)


@tool
async def unify_ingredient_names(ingredient_list: List[IngredientsOutput]) -> List[IngredientsOutput]:
    """Unify ingredient names based on a list of extracted ingredients from multiple recipes."""
    all_ingredient_names = {
        ingredient.name for recipe_ingredients in ingredient_list for ingredient in recipe_ingredients.ingredients
    }

    if not all_ingredient_names:
        return ingredient_list  

    parser = JsonOutputParser(pydantic_object=IngredientNamesOutput)
    prompt = PromptTemplate(
        template="""
Here is a list of ingredient names. Please unify them to common names in Polish language, trying to create as many synonyms as possible, e.g. "Pierś z kurczaka", "Pierś z kurczaka bez skóry", "Pierś z kurczaka bez kości" should be unified to "Pierś z kurczaka".
{format_instructions}

Ingredient names:
{ingredient_names}
""",
        input_variables=["ingredient_names"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    result = await chain.ainvoke({"ingredient_names": "\n".join(all_ingredient_names)})
    unified_names_map = {item["original_name"]: item["target_name"] for item in result.get("ingredient_names", [])}

    for recipe_ingredients in ingredient_list:
        for ingredient_item in recipe_ingredients.ingredients:
            if ingredient_item.name in unified_names_map:
                ingredient_item.name = unified_names_map[ingredient_item.name]

    return ingredient_list


@tool
async def produce_final_result(ingredients_list: list[IngredientsOutput]) -> dict[str, list[Ingredient]]:
    """Group multiple lists of ingredients (potentially unified) into a final dictionary keyed by common ingredient names."""
    merged_ingredients = defaultdict(list)
    for ingredients in ingredients_list:
        for ingredient in ingredients.ingredients:
            merged_ingredients[ingredient.name].append(ingredient)

    return dict(merged_ingredients)  


tools = [
    fetch_recipe_from_url,
    extract_ingredients,
    unify_ingredient_names,
    produce_final_result,
]
