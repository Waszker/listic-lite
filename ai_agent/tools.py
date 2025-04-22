import asyncio
from collections import defaultdict
from traceback import print_stack
import json
import logging
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.prompts import PromptTemplate
from ai_agent.config import llm, cheaper_llm
from ai_agent.data_models import Ingredient, IngredientsOutput, IngredientNamesOutput, ConsolidatedIngredientOutput
from ai_agent.tasks import fetch_recipe_from_url
import re
import traceback

# --- Predefined Examples for Unit Consolidation ---
UNIT_CONSOLIDATION_EXAMPLES = [
    {
        "input": "Tomatoes: 1 can, 0.5 can",
        "output": '{{"name": "Tomatoes", "quantity": 2, "unit": "can"}}',
        "explanation": "Sum cans, round up to nearest whole can for shopping.",
    },
    {
        "input": "Paprika: 3 szt., 100g",
        "output": '{{"name": "Paprika", "quantity": 4, "unit": "szt."}}',
        "explanation": "Estimate 100g as ~1 paprika (avg 120g), add to existing 3, total 4 szt. for shopping.",
    },
    {
        "input": "Onion: 4 szt., 0.5 szt.",
        "output": '{{"name": "Onion", "quantity": 5, "unit": "szt."}}',
        "explanation": "Sum pieces, round 4.5 up to 5 szt. for shopping.",
    },
    {
        "input": "Flour: 1 tsp, 0.5 cup",
        "output": '{{"name": "Flour", "quantity": 1, "unit": "opak."}}',
        "explanation": "1 tsp flour (~3g) + 0.5 cup flour (~60g) = ~63g. This is a small amount, buy 1 standard package (opak.) assuming ~1kg.",
    },
    {
        "input": "Milk: 1 carton, 100ml",
        "output": '{{"name": "Milk", "quantity": 1, "unit": "carton"}}',
        "explanation": "Already have 1 carton, 100ml is negligible compared to a standard carton (usually 1L), buy 1 carton.",
    },
    {
        "input": "Salt: 1 pinch, 1 tsp",
        "output": '{{"name": "Salt", "quantity": 1, "unit": "opak."}}',
        "explanation": "1 pinch (~0.5g) + 1 tsp (~6g) = ~6.5g. Small amount, buy 1 standard package (opak.).",
    },
    {
        "input": "Sugar: 200g, 1 cup",
        "output": '{{"name": "Sugar", "quantity": 400, "unit": "g"}}',
        "explanation": "200g + 1 cup (~200g) = 400g. Keep as grams for shopping.",
    },
    {
        "input": "Chicken Breast: 1 kg, 200g",
        "output": '{{"name": "Chicken Breast", "quantity": 1200, "unit": "g"}}',
        "explanation": "1 kg (1000g) + 200g = 1200g. Convert kg to g and sum.",
    },
]

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
async def consolidate_units(ingredients: list[Ingredient]) -> ConsolidatedIngredientOutput:
    """
    Consolidates a list of ingredients with the SAME name but potentially different units/quantities
    into a single ingredient entry with a final quantity and unit suitable for a shopping list.
    Applies logic like rounding up pieces, converting small amounts to packages, etc.

    Args:
        ingredients: A list of Ingredient objects, all sharing the same name.

    Returns:
        A ConsolidatedIngredientOutput object with the final name, quantity, and unit.
    """
    if not ingredients:
        raise ValueError("Input list of ingredients cannot be empty.")

    # Verify all ingredients have the same name (case-insensitive check)
    first_name = ingredients[0].name.lower()
    if not all(ing.name.lower() == first_name for ing in ingredients):
        raise ValueError("All ingredients passed to consolidate_units must have the same name.")

    ingredient_name = ingredients[0].name  # Use the first ingredient's name casing

    # Format the input for the prompt
    input_description = f"{ingredient_name}: {', '.join([f'{ing.quantity} {ing.unit}' for ing in ingredients])}"
    print(f"Attempting to consolidate units for: {input_description}")

    # Define parser for the desired output structure
    parser = JsonOutputParser(pydantic_object=ConsolidatedIngredientOutput)

    # Define the prompt for formatting a single example
    example_prompt = PromptTemplate(
        input_variables=["input", "output", "explanation"],
        template="Input: {input}\nOutput: {output}\nExplanation: {explanation}",
    )

    # Define the prefix (instructions before examples)
    prefix = """
You are an expert shopping list assistant. Your task is to take a list of quantities for the SAME ingredient (provided with different units) and consolidate them into a SINGLE final quantity and unit suitable for buying at a store.
Apply the following logic:
- Convert compatible metric units (kg to g, l to ml).
- If mixing pieces ('szt.') and weights ('g'/'ml'), estimate the weight equivalent in pieces (e.g., 1 onion ≈ 130g, 1 apple ≈ 80g) and add to the piece count. Round the TOTAL piece count UP to the nearest whole number. Final unit: 'szt.'.
- If dealing with small amounts (e.g., < 50g/ml) of spices, herbs, salt, baking powder etc., determine how many standard packages ('opak.') are needed (assume spice jar ~20g, baking powder sachet ~15g). Round UP to the nearest whole package. Final unit: 'opak.'.
- For larger amounts of bulk items (flour, sugar, liquids), keep the summed total in 'g' or 'ml'.
- Handle non-standard units like 'can', 'carton' by summing them and rounding UP.
- The final output MUST contain the ingredient name, a single numeric quantity, and a single final unit ('szt.', 'opak.', 'g', 'ml', 'can', 'carton', etc.). Provide a brief explanation.

Here are some examples:"""

    # Define the suffix (instructions after examples, including the actual input)
    suffix = """
Now, consolidate the following ingredient quantities:
Input: {target_input}

YOU MUST conform to the output format specified below:
{format_instructions}
Output:"""  # Ensure output is directly after the format instructions

    # Create the FewShotPromptTemplate instance within the function call
    few_shot_prompt = FewShotPromptTemplate(
        examples=UNIT_CONSOLIDATION_EXAMPLES,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["target_input"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Create the chain
    chain = few_shot_prompt | cheaper_llm | parser  # Added the parser to the chain

    try:
        result_dict = await chain.ainvoke(
            {
                "target_input": input_description,
            }
        )
        # Basic validation might be needed here depending on LLM reliability
        print(f"Consolidation successful for '{ingredient_name}': {result_dict}")
        # Return as Pydantic object
        return ConsolidatedIngredientOutput(**result_dict)
    except Exception as e:
        print(f"Error consolidating units for '{ingredient_name}': {e}")
        print("Stack trace:")
        print(traceback.format_exc())
        # Consider returning a specific error structure or raising the exception
        # depending on how the agent should handle failures.
        # For now, re-raise the exception for clarity.
        raise e


@tool
def sum_quantities(ingredient1: Ingredient, ingredient2: Ingredient) -> float:
    """
    Sums the quantities of two ingredients.
    """
    try:
        return float(ingredient1.quantity) + float(ingredient2.quantity)
    except ValueError:
        raise ValueError("Cannot parse quantity. Ensure both ingredients have valid numeric quantities.")


tools = [
    fetch_recipes_from_urls,
    extract_ingredients,
    unify_ingredient_names,
    consolidate_units,
    handle_unknown_units,
    group_by_ingredient_name,
    sum_quantities,
]
