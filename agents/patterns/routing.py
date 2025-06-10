from openai import AzureOpenAI
from enum import Enum
import logging
from env_setup import env_vars
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

GPT_4_O = "gpt-4o-mini-2"
GEMINI_2_0_FLASH = "gemini-2.0-flash"


class ModelRoute(str, Enum):
    EASY = GEMINI_2_0_FLASH
    HARD = GPT_4_O
    UNKNOWN = "unknown"


class RoutingDecision(BaseModel):
    question_type: ModelRoute = Field(
        ...,
        description="The recommended model route based on the query's complexity and nature.",
    )
    reasoning: str = Field(
        ..., description="A brief explanation of why this route was chosen."
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="A confidence score (0.0 to 1.0) for the routing decision.",
    )


def call_small_model(question: str) -> str:
    """Call the lightweight Gemini model for easy questions."""
    try:
        logger.info("Calling Gemini small model.")
        client = genai.Client(api_key=env_vars["gemini_api_key"])
        response = client.models.generate_content(
            model=GEMINI_2_0_FLASH,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a concise and knowledgeable AI assistant. "
                    "Provide a clear, accurate answer to the user's question in fewer than 50 words. "
                    "Focus on being direct, helpful, and easy to understand. "
                    "Avoid unnecessary details or repetition."
                ),
            ),
            contents=question,
        )
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        logger.error("No response text from Gemini small model.")
        return "Sorry, I couldn't generate an answer at this time."
    except Exception as e:
        logger.exception(f"Gemini small model error: {e}")
        return "An error occurred while processing your request with the small model."


def call_large_model(question: str) -> str:
    """Call the powerful GPT-4o model for hard questions."""
    try:
        logger.info("Calling GPT-4o large model.")
        client = AzureOpenAI(
            api_key=env_vars["api_key"],
            api_version=env_vars["api_version"],
            azure_endpoint=env_vars["azure_endpoint"],
        )
        response = client.chat.completions.create(
            model=GPT_4_O,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert AI assistant. "
                        "Provide a precise, well-reasoned answer to the user's question in fewer than 50 words. "
                        "Be clear, accurate, and insightful. "
                        "If the question is ambiguous, briefly clarify assumptions."
                    ),
                },
                {"role": "user", "content": question},
            ],
        )
        if hasattr(response, "choices") and response.choices:
            content = response.choices[0].message.content
            if content is not None:
                return content.strip()
            logger.error("GPT-4o large model returned None content.")
            return "Sorry, I couldn't generate an answer at this time."
        logger.error("No choices from GPT-4o large model.")
        return "Sorry, I couldn't generate an answer at this time."

    except Exception as e:
        logger.exception(f"GPT-4o large model error: {e}")
        return "An error occurred while processing your request with the large model."


def decide_route(question: str):
    """Determine the appropriate model route for the given question."""
    try:
        logger.info("Determining model route.")
        client = genai.Client(api_key=env_vars["gemini_api_key"])
        response = client.models.generate_content(
            model=GEMINI_2_0_FLASH,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are an expert AI routing assistant. "
                    "Given a user's question, analyze its complexity and decide if it is EASY (can be answered by a lightweight model), "
                    "HARD (requires a powerful model), or UNKNOWN. "
                    "Respond in valid JSON as a list of RoutingDecision objects, including your reasoning and a confidence score between 0.0 and 1.0. "
                    "Be concise and objective in your assessment."
                ),
                response_mime_type="application/json",
                response_schema=list[RoutingDecision],
            ),
            contents=question,
        )
        if hasattr(response, "text") and response.text:
            try:
                return json.loads(response.text)
            except json.JSONDecodeError as jde:
                logger.error(f"Failed to parse routing decision JSON: {jde}")
                return None
        logger.error("No routing decision text from Gemini.")
        return None

    except Exception as e:
        logger.exception(f"Routing decision error: {e}")
        return None


def get_answer(question: str) -> str:
    """Route the question and get an answer from the appropriate model."""
    logger.info("Starting answer generation workflow.")
    route = decide_route(question)
    if not route or not isinstance(route, list) or not route[0].get("question_type"):
        logger.error("Failed to determine a valid route.")
        return "Sorry, I couldn't determine how to answer your question."

    question_type = route[0]["question_type"]
    reasoning = route[0].get("reasoning", "No reasoning provided.")
    confidence = route[0].get("confidence_score", "N/A")

    logger.info(f"Routing decision: {question_type} (confidence: {confidence})")
    logger.info(f"Routing reasoning: {reasoning}")

    if question_type == GEMINI_2_0_FLASH:
        logger.info("Routing to Gemini small model.")
        return call_small_model(question)
    elif question_type == GPT_4_O:
        logger.info("Routing to GPT-4o large model.")
        return call_large_model(question)
    else:
        logger.warning(f"Unknown routing type: {question_type}.")
        return "Sorry, I couldn't determine how to answer your question."


if __name__ == "__main__":
    # hard question
    print(get_answer("what is the meaning of life, the universe, and everything?"))

    # easy question
    print(get_answer("what is the chemical formula of water?"))
