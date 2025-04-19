from langchain_openai import ChatOpenAI
import settings

llm = ChatOpenAI(temperature=0.2, openai_api_key=settings.env_settings.openai_api_key, model_name="gpt-4o")
