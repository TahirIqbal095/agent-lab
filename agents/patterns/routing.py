from openai import AzureOpenAI
import logging
from agents.basics.env_setup import env_vars

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


client = AzureOpenAI(
    api_key=env_vars["api_key"],
    api_version=env_vars["api_version"],
    azure_endpoint=env_vars["api_version"],
)
