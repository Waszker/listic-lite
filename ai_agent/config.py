from langchain_openai import ChatOpenAI
import settings

SYSTEM_PROMPT_PREFIX = "You are a helpful assistant specializing in recipe analysis."
llm = ChatOpenAI(temperature=0.2, openai_api_key=settings.env_settings.openai_api_key, model_name="gpt-4o")
