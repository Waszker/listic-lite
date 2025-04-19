from typing import Self
from pydantic import BaseModel
from dotenv import dotenv_values

class EnvSettings(BaseModel):
    """
    Configuration settings for the application.
    
    Values are loaded from the .env file, present in the root directory.
    """
    openai_api_key: str

    @classmethod
    def load(cls, env_path: str = ".env") -> Self:
        return cls(**{key.lower(): value for key, value in dotenv_values(env_path).items()})
        
