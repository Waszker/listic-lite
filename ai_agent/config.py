from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
import settings

SYSTEM_PROMPT_PREFIX = "You are a helpful assistant specializing in recipe analysis."
rate_limiter = InMemoryRateLimiter(
    requests_per_second=2,  # <-- Super slow! We can only make a request once every 0.5 seconds!!
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=10,  # Controls the maximum burst size.
)
llm = ChatOpenAI(temperature=0.2, openai_api_key=settings.env_settings.openai_api_key, model_name="gpt-4o")
cheaper_llm = ChatOpenAI(
    temperature=0.2,
    openai_api_key=settings.env_settings.openai_api_key,
    model_name="gpt-4o-mini",
    rate_limiter=rate_limiter,
)
