import asyncio
from collections import defaultdict
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from ai_agent.config import llm
from ai_agent.data_models import Ingredient, IngredientsOutput, IngredientNamesOutput
from ai_agent.tasks import fetch_recipe_from_url

@tool
async def fetch_recipes_from_urls(urls: list[str]) -> list[str]:
    """
    Fetches recipe texts from the given URLs using Playwright.
    
    Each URL is processed asynchronously in parallel.
    """
    return await asyncio.gather(*[fetch_recipe_from_url(url) for url in urls])

@tool
async def extract_ingredients(recipe_texts: list[str]) -> list[IngredientsOutput]:
    """Extracts ingredients from the recipe text"""
    parser = JsonOutputParser(pydantic_object=IngredientsOutput)

    chains = []
    for recipe_text in recipe_texts:
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
        chains.append(chain)

    results = await asyncio.gather(*[chain.ainvoke({"recipe_text": recipe_text}) for recipe_text in recipe_texts])
    return [IngredientsOutput(**result) for result in results]


@tool
async def unify_ingredient_names(ingredient_list: list[IngredientsOutput]) -> list[IngredientsOutput]:
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
async def handle_unknown_units(ingredients_with_unknown_units: list[Ingredient]) -> list[Ingredient]:
    """
    Tries to handle ingredients with unknown units by asking for clarification.
    """
    parser = JsonOutputParser(pydantic_object=Ingredient)
    chains = []
    for ingredient in ingredients_with_unknown_units:
        prompt = PromptTemplate(
            template="""
Here is an ingredient with unknown units. Try to convert it to a gram or milliliter unit.
{format_instructions}

Ingredient:
{ingredient}
""",
            input_variables=["ingredients"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        chains.append(chain)

    results = await asyncio.gather(*[chain.ainvoke({"ingredients": f"{ing.name}: {ing.quantity} {ing.unit}"}) for ing in ingredients_with_unknown_units])
    return [Ingredient(**result) for result in results]


@tool
async def produce_final_result(ingredients_list: list[IngredientsOutput]) -> dict[str, list[Ingredient]]:
    """Group multiple lists of ingredients (potentially unified) into a final dictionary keyed by common ingredient names."""
    merged_ingredients = defaultdict(list)
    for ingredients in ingredients_list:
        for ingredient in ingredients.ingredients:
            merged_ingredients[ingredient.name].append(ingredient)

    return dict(merged_ingredients)  


tools = [
    fetch_recipes_from_urls,
    extract_ingredients,
    unify_ingredient_names,
    handle_unknown_units,
    produce_final_result,
]
