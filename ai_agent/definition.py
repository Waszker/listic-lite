from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from typing import List

# Import necessary components from other modules
from ai_agent.config import llm, SYSTEM_PROMPT_PREFIX
from ai_agent.tools import tools


def run_agent(user_inputs: List[str]):
    """
    Runs the LangChain agent to extract, unify, and group ingredients.
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
        result = agent_executor.invoke(
            {
                "input": f"""
You have been provided with {len(user_inputs)} text item(s). Each item could be a full recipe scraped from a website, a manually entered recipe, or just a list of ingredients. Your task is to process all items to produce a unified list of ingredients grouped by name.

Follow these steps precisely:

1. For each text item decide how to process it in order to retrieve the recipe text.
2. For each recipe text, use the `extract_ingredients` tool to get its list of ingredients. Collect all extracted ingredient lists.
3. Pass the complete collection of extracted ingredient lists to the `unify_ingredient_names` tool.
4. Take the unified list output from the previous step and pass it to the `produce_final_result` tool.
5. Return the final grouped ingredients dictionary produced by `produce_final_result`.

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
