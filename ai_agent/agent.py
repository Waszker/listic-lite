from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from typing import List

from ai_agent.config import llm, SYSTEM_PROMPT_PREFIX
from ai_agent.tools import tools


async def run_agent(user_inputs: List[str]):
    """
    Runs the LangChain agent asynchronously to extract, unify, and group ingredients.
    """

    if not user_inputs:
        print("No valid recipe item could be processed. Exiting.")
        return

    recipes_prompt = "\n\n---\n\n".join([f"***Input Item #{i}***\n{content}" for i, content in enumerate(user_inputs)])

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT_PREFIX),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_functions_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=50
    )

    print("--- Invoking Agent ---")
    try:
        result = await agent_executor.ainvoke(
            {
                "input": f"""
You have been provided with {len(user_inputs)} text item(s). Each item could be a full recipe scraped from a website, a manually entered recipe, or just a list of ingredients. Your task is to process all items to produce a unified list of ingredients grouped by name, paying attention to units.

Follow these steps precisely:

1. For each text item decide how to process it in order to retrieve the recipe text. Use `fetch_recipes_from_urls` for found URLs.
2. Use the `extract_ingredients` tool to get list of ingredients from the recipes. Collect all extracted ingredient lists.
3. Pass the complete collection of extracted ingredient lists to the `unify_ingredient_names` tool.
4. Take the name-adjusted list from the previous step and pass it to the `group_by_ingredient_name` tool. This tool will group the ingredients.
5. **Group Ingredients:** Use the `group_by_ingredient_name` tool to group all ingredients (from all recipes) by their unified names.

6. **Consolidate Quantities:** For each ingredient group from the previous step:
    - If the group contains only one ingredient entry, keep it as is.
    - If the group contains multiple entries:
        - First, standardize units: Use `handle_unknown_units` for any non-standard (like 'clove', 'pinch', 'can', 'cup') or non-piece ('szt.') units to get them into grams ('g') or milliliters ('ml'). If a unit is already 'g', 'ml', or 'szt.', keep it.
        - Convert larger metric units to smaller ones (e.g., 1 kg -> 1000g, 1 l -> 1000ml).
        - Now, consolidate based on shopping logic:
            - If you have only grams ('g') or only milliliters ('ml'), sum them using `sum_quantities`. Final unit: 'g' or 'ml'.
            - If you have pieces ('szt.') mixed with weights ('g') or volumes ('ml'): Estimate the weight/volume in terms of pieces (e.g., 1 large onion ≈ 150g, 1 medium apple ≈ 80g, 1 standard paprika ≈ 120g). Add this estimated piece count to any existing pieces. Round the FINAL total number of pieces UP to the nearest whole number. Final unit: 'szt.'. Use your judgment for reasonable estimations.
            - If you have only pieces ('szt.'), sum them using `sum_quantities`. Final unit: 'szt.'.
            - If after conversion you have small amounts (< 50g/ml) of things typically bought in packages (like spices, herbs, baking powder, yeast, potentially salt/sugar if very small amounts): Convert the total quantity to the number of standard packages needed (e.g., spice jar ≈ 20g, baking powder sachet ≈ 15g). Round UP to the nearest whole package. Final unit: 'opak.'.
            - Keep track of the final consolidated quantity and unit for each ingredient.
    - Use the `consolidate_units` tool. Pass the list of `Ingredient` objects for that specific ingredient name to the tool.
    - The tool will return a single `ConsolidatedIngredientOutput` object containing the final shopping quantity and unit (e.g., szt., opak., g, ml).

7. **Final Shopping List:** Format the results into a final, user-friendly shopping list.
    - The list should contain each ingredient with its final consolidated quantity and unit (szt., opak., g, ml).
    - Use Polish names for ingredients.
    - For each ingredient, use the name, quantity, and unit from the `ConsolidatedIngredientOutput` returned by `consolidate_units` in the previous step.
    - Present the list clearly.

8. **Generate Audio:**
    - Use the `generate_audio_for_list` tool to generate an audio file from the final shopping list.
    - The audio file will be saved in the current directory with the name 'shopping_list.mp3'.

Input Texts:
{recipes_prompt}
"""
            }
        )
        print("\n--- Agent Result ---")
        print(result)
        print("--- Agent Finished ---")
    except Exception as e:
        print(f"Agent execution failed: {e}")
