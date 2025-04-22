import asyncio
from collections import defaultdict
from traceback import print_stack
import json
import logging
import os
from pathlib import Path
from openai import AsyncOpenAI
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.prompts import PromptTemplate
from ai_agent.config import llm, cheaper_llm
from ai_agent.data_models import Ingredient, IngredientsOutput, IngredientNamesOutput, ConsolidatedIngredientOutput
from ai_agent.tasks import fetch_recipe_from_url
import re
import traceback
import settings

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


# --- Initialize OpenAI Client (can be done once outside the tool if preferred) ---
# Make sure OPENAI_API_KEY environment variable is set
try:
    openai_client = AsyncOpenAI(api_key=settings.env_settings.openai_api_key)
except Exception as e:
    raise RuntimeError(f"Failed to initialize OpenAI client: {e}. Ensure OPENAI_API_KEY is set.")


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


@tool
async def generate_audio_for_list(shopping_list_text: str, output_file_path: str = "shopping_list.mp3") -> str:
    """
    Generates an audio file from the provided shopping list text using OpenAI's Text-to-Speech (TTS) API.

    Args:
        shopping_list_text: The text content of the final shopping list.
        output_file_path: The path (including filename) where the audio file should be saved. Defaults to 'shopping_list.mp3'.

    Returns:
        The absolute path to the generated audio file, or an error message if generation failed.
    """
# Removed redundant check for openai_client.

    try:
        speech_file_path = Path(output_file_path)
        # Ensure the directory exists
        speech_file_path.parent.mkdir(parents=True, exist_ok=True)

        logging.info(f"Generating audio for shopping list to {speech_file_path}...")
        response = await openai_client.audio.speech.create(
            model="tts-1",  # Or "tts-1-hd" for higher quality
            voice="alloy",  # Choose a voice: alloy, echo, fable, onyx, nova, shimmer
            input=shopping_list_text,
        )

        # Save the audio stream to the file
        response.stream_to_file(speech_file_path)

        logging.info(f"Successfully generated audio file: {speech_file_path.absolute()}")
        return str(speech_file_path.absolute())

    except Exception as e:
        logging.error(f"Error generating audio: {e}")
        traceback.print_exc() # Print full traceback for debugging
        return f"Error generating audio: {e}"


tools = [
    fetch_recipes_from_urls,
    extract_ingredients,
    unify_ingredient_names,
    consolidate_units,
    group_by_ingredient_name,
    sum_quantities,
    generate_audio_for_list,
]
