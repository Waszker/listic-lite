import asyncio
from collections import defaultdict
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.prompts import PromptTemplate
from ai_agent.config import llm
from ai_agent.data_models import Ingredient, IngredientsOutput, IngredientNamesOutput, UnitConversionOutput
from ai_agent.tasks import fetch_recipe_from_url
import re

# --- Predefined Examples for Unit Conversion ---
UNIT_CONVERSION_EXAMPLES = [
    {"input": "half an apple", "output": "40g of apple", "explanation": "a standard apple weighs around 80g"},
    {"input": "1 small onion", "output": "130g of onion", "explanation": "a small onion weighs around 130g"},
    {
        "input": "0.5 of small onion",
        "output": "65g of onion",
        "explanation": "a small onion weighs around 130g, half is 65g",
    },
    {"input": "1 clove garlic", "output": "5g of garlic", "explanation": "a standard garlic clove weighs around 5g"},
    {"input": "1 pinch salt", "output": "0.5g of salt", "explanation": "a pinch is roughly 0.5g"},
    {"input": "1 pinch pepper", "output": "0.25g of pepper", "explanation": "a pinch of pepper is roughly 0.25g"},
    {"input": "1 handful spinach", "output": "30g of spinach", "explanation": "a handful is roughly 30g"},
    {"input": "1 sprig rosemary", "output": "2g of rosemary", "explanation": "a sprig is roughly 2g"},
    {
        "input": "1 can (400g) chopped tomatoes",
        "output": "400g of chopped tomatoes",
        "explanation": "unit is already standard, reformat",
    },
    {"input": "1 egg", "output": "50g of egg", "explanation": "a standard large egg weighs ~50g without shell"},
    {"input": "1 slice bread", "output": "30g of bread", "explanation": "a standard slice weighs ~30g"},
    {"input": "1 medium potato", "output": "150g of potato", "explanation": "a medium potato weighs ~150g"},
    {"input": "1 cup flour", "output": "120g of flour", "explanation": "a cup of all-purpose flour weighs ~120g"},
    {"input": "1 cup sugar", "output": "200g of sugar", "explanation": "a cup of granulated sugar weighs ~200g"},
    {"input": "1 cup milk", "output": "240ml of milk", "explanation": "a standard cup is 240ml"},
    {"input": "1 tbsp butter", "output": "14g of butter", "explanation": "a tablespoon of butter weighs ~14g"},
    {"input": "1 tsp salt", "output": "6g of salt", "explanation": "a teaspoon of salt weighs ~6g"},
    {"input": "A dash of hot sauce", "output": "1ml of hot sauce", "explanation": "a dash is approximately 1ml"},
    {"input": "A drizzle of olive oil", "output": "5ml of olive oil", "explanation": "a drizzle is typically ~5ml"},
    {
        "input": "Juice of 1 lemon",
        "output": "50ml of lemon juice",
        "explanation": "one medium lemon yields ~50ml juice",
    },
    {
        "input": "Zest of 1 orange",
        "output": "10g of orange zest",
        "explanation": "zest of one medium orange weighs ~10g",
    },
]


def select_examples(target_input: str, examples: list[dict], n: int = 3) -> list[dict]:
    """Selects the top n examples based on simple keyword overlap."""
    target_words = set(re.findall(r"\w+", target_input.lower()))

    def score_example(example):
        example_words = set(re.findall(r"\w+", example["input"].lower()))
        overlap = len(target_words.intersection(example_words))
        # Optional: Penalize length difference slightly if needed
        # length_diff = abs(len(target_words) - len(example_words))
        # return overlap - length_diff * 0.1
        return overlap

    # Sort examples by score in descending order
    sorted_examples = sorted(examples, key=score_example, reverse=True)

    return sorted_examples[:n]


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
Here is the recipe text. Extract the list of ingredients (translate to English name if needed) with their quantities and units.
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
Here is a list of ingredient names. Please unify them to common names in English language, trying to create as many synonyms as possible, e.g. "Pierś z kurczaka", "Pierś z kurczaka bez skóry", "Pierś z kurczaka bez kości" should be unified to "Pierś z kurczaka".
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
async def group_by_ingredient_name(ingredients_list: list[IngredientsOutput]) -> dict[str, list[Ingredient]]:
    """Group multiple lists of ingredients (potentially unified) into a final dictionary keyed by common ingredient names."""
    merged_ingredients = defaultdict(list)
    for ingredients in ingredients_list:
        for ingredient in ingredients.ingredients:
            merged_ingredients[ingredient.name].append(ingredient)

    return dict(merged_ingredients)


@tool
async def handle_unknown_units(ingredient_name: str, quantity_str: str, unit_str: str) -> dict:
    """
    Converts ingredient quantities expressed in non-standard units (e.g., '1 clove', 'half an apple', '1 pinch')
    into standard units, either grams (g) or milliliters (ml).

    Args:
        ingredient_name: The name of the ingredient (e.g., 'apple', 'onion', 'garlic').
        quantity_str: The quantity part of the description (e.g., 'half', '1', '0.5').
        unit_str: The unit part of the description (e.g., 'an', 'small', 'clove', 'pinch').

    Returns:
        A dictionary containing 'quantity' (float), 'unit' (str, 'g' or 'ml'),
        and 'explanation' (str) for the conversion, or raises an error if conversion fails.
    """
    input_description = f"{quantity_str} {unit_str} {ingredient_name}".strip()
    print(f"Attempting to convert unit (using FewShotPromptTemplate) for: {input_description}")

    # Select relevant examples
    selected_examples = select_examples(input_description, UNIT_CONVERSION_EXAMPLES, n=3)

    # Define parser for the desired output structure
    parser = JsonOutputParser(pydantic_object=UnitConversionOutput)

    # Define the prompt for formatting a single example
    example_prompt = PromptTemplate(
        input_variables=["input", "output", "explanation"],
        template="Input: {input}\nOutput: {output}\nExplanation: {explanation}",
    )

    # Define the prefix (instructions before examples)
    prefix = """
You are an expert nutrition assistant. Your task is to convert ingredient quantities from non-standard units (like pieces, pinches, halves) into standard metric units (grams 'g' or milliliters 'ml'). Use common sense and typical food item weights/volumes. Provide a brief explanation for your conversion.

Here are some examples of how to perform the conversion:"""

    # Define the suffix (instructions after examples, including the actual input)
    suffix = """
Now, convert the following ingredient:
Input: {target_input}

Output the result in JSON format with the fields 'quantity', 'unit', and 'explanation'.
{format_instructions}"""

    # Create the FewShotPromptTemplate instance within the function call
    # This allows using dynamically selected examples
    few_shot_prompt = FewShotPromptTemplate(
        examples=selected_examples,  # Pass the dynamically selected examples
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["target_input"],  # The final input variable
        partial_variables={"format_instructions": parser.get_format_instructions()},
        example_separator="\n\n",  # Separator between examples
    )

    # Create the chain
    chain = few_shot_prompt | llm | parser

    try:
        result = await chain.ainvoke(
            {
                "target_input": input_description,  # Pass the actual input
            }
        )
        # Basic validation
        if not isinstance(result.get("quantity"), (int, float)) or result.get("unit") not in ["g", "ml"]:
            raise ValueError("LLM returned invalid quantity or unit.")
        print(f"Conversion successful for '{input_description}': {result}")
        return result
    except Exception as e:
        print(f"Error converting unit for '{input_description}': {e}")
        # Consider returning a specific error structure or raising the exception
        # depending on how the agent should handle failures.
        # For now, return an error dict, but this might need adjustment.
        return {"error": f"Failed to convert units for '{input_description}'. Reason: {e}"}


@tool
def sum_quantities(ingredient1: Ingredient, ingredient2: Ingredient) -> float:
    """
    Sums the quantities of two ingredients.
    """
    try:
        return float(ingredient1.quantity) + float(ingredient2.quantity)
    except ValueError:
        return "Error: Cannot parse quantity, keep them separate"


tools = [
    fetch_recipes_from_urls,
    extract_ingredients,
    unify_ingredient_names,
    handle_unknown_units,
    group_by_ingredient_name,
    sum_quantities,
]
