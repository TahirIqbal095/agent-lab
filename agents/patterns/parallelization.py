import logging
import asyncio
from env_setup import env_vars
from pydantic import BaseModel, Field
from openai import AzureOpenAI
import nest_asyncio

# Allow nested event loops for asyncio
nest_asyncio.apply()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

client = AzureOpenAI(
    api_key=env_vars["api_key"],
    api_version=env_vars["api_version"],
    azure_endpoint=env_vars["azure_endpoint"],
)


class InputValidation(BaseModel):
    is_allowed: bool = Field(
        ...,
        description="Indicates whether the action is allowed or not.",
    )
    reason: str = Field(
        ...,
        description="The reason why the action is allowed or not.",
    )


async def topical_guardrail(topic: str) -> InputValidation | None:
    logger.debug(f"Running topical_guardrail check for topic: '{topic}'")
    response = client.beta.chat.completions.parse(
        model=env_vars["model"],
        messages=[
            {
                "role": "system",
                "content": (
                    "Your role is to assess whether the user question is allowed or not. "
                    "The allowed topics are software engineering and machine learning."
                ),
            },
            {"role": "user", "content": topic},
        ],
        response_format=InputValidation,
    )
    logger.debug("Received response from topical_guardrail check.")
    return response.choices[0].message.parsed


async def get_answer(topic: str) -> str | None:
    logger.info(f"Fetching answer for topic: '{topic}'")
    response = client.chat.completions.create(
        model=env_vars["model"],
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions about software engineering and machine learning."
                ),
            },
            {"role": "user", "content": topic},
        ],
    )
    logger.info("Received answer from model.")
    return response.choices[0].message.content


async def get_answer_with_guardrail(topic: str) -> str | None:
    logger.info(f"Starting guardrail and answer retrieval for topic: '{topic}'")
    topical_guardrail_check, answer = await asyncio.gather(
        topical_guardrail(topic), get_answer(topic)
    )

    is_allowed = (
        topical_guardrail_check.is_allowed
        if isinstance(topical_guardrail_check, InputValidation)
        else False
    )

    if is_allowed:
        logger.info(f"Topic allowed: '{topic}'. Returning answer.")
        return answer
    else:
        reason = (
            topical_guardrail_check.reason
            if topical_guardrail_check is not None
            else "Unknown reason"
        )
        logger.warning(f"Topic not allowed: '{topic}'. Reason: {reason}")
        return f"Topic not allowed: {reason}"


async def main():
    logger.info("Starting the parallelization example with guardrail checks.")

    prompts = [
        "Tell me some short history about the world war 2.",
        "What is the best way to train a neural network for image classification?",
    ]

    logger.info(f"Processing prompt: '{prompts[0]}'")
    await get_answer_with_guardrail(prompts[0])


if __name__ == "__main__":
    asyncio.run(main())
