import os
from dotenv import load_dotenv

load_dotenv()


env_vars = {
    "model": "gpt-4o-mini-2",
    "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
    "azure_endpoint": os.getenv("AZURE_ENDPOINT"),
    "api_version": os.getenv("API_VERSION"),
}
