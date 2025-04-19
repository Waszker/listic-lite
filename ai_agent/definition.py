import requests
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent

# Import necessary components from other modules
from .config import llm
from .tools import tools  # Import the list of tools

# PREFIX constant needs to be defined or imported if used
# Assuming PREFIX might be a system prompt part, define it or handle its import
PREFIX = "You are a helpful assistant."  # Example definition, adjust as needed


def fetch_recipe_text(url: str) -> str:
    """Fetches recipe text from the given URL"""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif," "image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    # Use the imported requests library
    session = requests.Session()
    session.headers.update(headers)
    response = session.get(url, timeout=100)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.body.get_text(separator="\n", strip=True)


def run_agent() -> None:
    urls = [
        "https://www.kwestiasmaku.com/przepis/kurczak-w-sosie-curry",
        # "https://www.allrecipes.com/recipe/46822/indian-chicken-curry-ii/",
        # "https://www.indianhealthyrecipes.com/chicken-curry/",
        "https://headbangerskitchen.com/indian-curry-chicken-curry/",
    ]
    recipes = []
    for url in urls:
        try:
            recipes.append(fetch_recipe_text(url))
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            # Optionally continue or handle the error appropriately

    if not recipes:
        print("No recipes fetched successfully. Exiting.")
        exit()

    recipes_prompt = "\n\n".join([f"***Recipe #{i}***\n{recipe}" for i, recipe in enumerate(recipes)])

    # Define the prompt for the agent
    # Ensure PREFIX is defined appropriately above or imported
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PREFIX),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Create the agent
    agent = create_openai_functions_agent(llm, tools, prompt)

    # Create the agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Invoke the agent
    try:
        result = agent_executor.invoke(
            {
                "input": f"""
Here are {len(recipes)} recipes. Follow these steps exactly:

1. Extract the ingredients from each recipe using the extract_ingredients tool.
2. Pass the complete list of extracted ingredients to the unify_ingredient_names tool.
3. Take the unified list and pass it to the produce_final_result tool.
4. Return the final groupings.

Recipe texts:
{recipes_prompt}
"""
            }
        )
        print(result)
    except Exception as e:
        print(f"Agent execution failed: {e}")
