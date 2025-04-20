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
5. **Standardize Units within Groups:** Examine the grouped list from step 4. For each ingredient group (ingredients with the same unified name):
   - Use the `handle_unknown_units` tool on any ingredient entry that doesn't have a standard metric unit (g, kg, ml, l) or a piece count (szt.). Aim to convert these to grams (g) or milliliters (ml).
   - Convert all compatible metric units to their base units: kilograms (kg) to grams (g), and liters (l) to milliliters (ml). After this, units within a group should primarily be 'g', 'ml', or 'szt.' (pieces).

6. **Consolidate Quantities with Shopping Logic:** Process the standardized groups to determine the final amount to purchase for each ingredient:
   - Use the `sum_quantities` tool to aggregate quantities for the *same unit* within each ingredient group (e.g., sum all 'g' of onions, sum all 'ml' of milk, sum all 'szt.' of apples).
   - **Apply Shopping Adjustments:**
     - **Piece-Based Items (e.g., vegetables, fruits):** If an ingredient group has quantities in both 'szt.' (pieces) *and* 'g'/'ml', use common sense weighting (e.g., 1 onion ≈ 130g, 1 apple ≈ 80g, 1 lemon ≈ 100g) to estimate how many additional pieces the 'g'/'ml' represents. Add this estimate to the 'szt.' count and round the *total* number of pieces *up* to the nearest whole number. The final unit for this ingredient should be 'szt.'.
     - **Spices/Herbs/Small Items (e.g., salt, pepper, baking powder):** If the total summed quantity is expressed in 'g' or 'ml' and is relatively small (e.g., under 50g/ml), determine how many standard-sized packages are needed (assume typical spice jar ≈ 20g). Round *up* to the nearest whole package. The final unit should be 'opak.' (package). If the quantity is large, keep the unit as 'g' or 'ml'.
     - **Liquids/Flour/Sugar/Other Bulk:** Keep the total summed quantity in 'g' or 'ml'.
   - The goal is to have a single quantity and unit per ingredient name, reflecting what needs to be bought.

7. **Return Final Shopping List:** Use the `produce_final_result` tool to format the final list. Each item should have its Polish name, the calculated shopping quantity, and the final unit ('szt.', 'opak.', 'g', 'ml').

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
