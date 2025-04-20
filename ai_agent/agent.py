from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from typing import List
import asyncio

# Import necessary components from other modules
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

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    print("--- Invoking Agent ---")
    try:
        result = await agent_executor.ainvoke(
            {
                "input": f"""
You have been provided with {len(user_inputs)} text item(s). Each item could be a full recipe scraped from a website, a manually entered recipe, or just a list of ingredients. Your task is to process all items to produce a unified list of ingredients grouped by name, paying attention to units.

Follow these steps precisely:

1. For each text item decide how to process it in order to retrieve the recipe text. Use `fetch_recipes_from_urls` for found URLs.
2. Use the `extract_ingredients` tool to get its list of ingredients from the recipes. Collect all extracted ingredient lists.
3. Pass the complete collection of extracted ingredient lists to the `unify_ingredient_names` tool.
4. **Examine the unified list:** Check if any ingredient name appears multiple times with different units.
   - If units are compatible and convertible (e.g., 'ml' and 'l', 'g' and 'kg'), standardize them to a single common unit, but *do not sum the quantities*:
     - prefer numeric units expressed in grams or milliliters, try to convert other units to grams or milliliters
     - use `handle_unknown_units` tool to handle ingredients with unknown units
   - If units are incompatible (e.g., 'liters' vs 'cartons', 'grams' vs 'pieces'), keep the entries separate but ensure they are clearly identifiable in the next step.
5. Take the potentially adjusted list from the previous step and pass it to the `produce_final_result` tool. This tool will group the ingredients.
6. Return the final grouped ingredients dictionary produced by `produce_final_result`. The result should clearly reflect any standardization performed or note any unresolved unit incompatibilities.

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
