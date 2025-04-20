from typing import List
from collections import defaultdict
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from ai_agent.config import llm
from ai_agent.data_models import Ingredient, IngredientsOutput, IngredientNamesOutput


@tool
def fetch_recipe_from_url(url: str) -> str:
    """Fetches recipe text from the given URL using Playwright to handle dynamic content."""
    print(f"Fetching recipe from URL with Playwright: {url}")
    page_content = ""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=60000)  # Wait for DOM, 60s timeout
            # Optional: Add a small wait for potentially slow-loading JS
            page.wait_for_timeout(2000)
            page_content = page.content()
            browser.close()
    except Exception as e:
        print(f"Playwright failed to fetch {url}: {e}")
        # Fallback or return error message
        return f"Error: Could not fetch content from {url}. Reason: {e}"

    if not page_content:
        return f"Error: No content fetched from {url}"

    # Use BeautifulSoup to parse the HTML fetched by Playwright
    soup = BeautifulSoup(page_content, "html.parser")
    recipe_content = (
        soup.find(class_=lambda x: x and "recipe" in x.lower())
        or soup.find(id=lambda x: x and "recipe" in x.lower())
        or soup.find("article")
        or soup.body
    )
    if not recipe_content:
        # Fallback to returning all text if specific containers aren't found
        print(f"Specific recipe container not found for {url}, falling back to full body text.")
        return soup.get_text(separator="\n", strip=True)

    return recipe_content.get_text(separator="\n", strip=True)


@tool
def extract_ingredients(recipe_text: str) -> IngredientsOutput:
    """Extracts ingredients from the recipe text"""
    # Create parser
    parser = JsonOutputParser(pydantic_object=IngredientsOutput)

    # Create prompt template
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

    # Create chain
    chain = prompt | llm | parser

    # Execute and return structured output
    result = chain.invoke({"recipe_text": recipe_text})
    # Ensure the result is parsed into the Pydantic model
    return IngredientsOutput(**result)


@tool
def unify_ingredient_names(ingredient_list: List[IngredientsOutput]) -> List[IngredientsOutput]:
    """Unify ingredient names based on a list of extracted ingredients from multiple recipes."""
    all_ingredient_names = {
        ingredient.name for recipe_ingredients in ingredient_list for ingredient in recipe_ingredients.ingredients
    }

    if not all_ingredient_names:
        return ingredient_list  # Return early if no names to unify

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

    # Create chain
    chain = prompt | llm | parser

    # Execute and return structured output
    result = chain.invoke({"ingredient_names": "\n".join(all_ingredient_names)})
    unified_names_map = {item["original_name"]: item["target_name"] for item in result.get("ingredient_names", [])}

    # Apply unified names back to the original list
    for recipe_ingredients in ingredient_list:
        for ingredient_item in recipe_ingredients.ingredients:
            if ingredient_item.name in unified_names_map:
                ingredient_item.name = unified_names_map[ingredient_item.name]

    return ingredient_list


@tool
def produce_final_result(ingredients_list: list[IngredientsOutput]) -> dict[str, list[Ingredient]]:
    """Group multiple lists of ingredients (potentially unified) into a final dictionary keyed by common ingredient names."""
    merged_ingredients = defaultdict(list)
    for ingredients in ingredients_list:
        for ingredient in ingredients.ingredients:
            merged_ingredients[ingredient.name].append(ingredient)

    return dict(merged_ingredients)  # Convert defaultdict back to dict for the final output


# Define the list of tools available to the agent
tools = [
    fetch_recipe_from_url,
    extract_ingredients,
    unify_ingredient_names,
    produce_final_result,
]
