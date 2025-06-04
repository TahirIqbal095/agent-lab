import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
import requests
import json

load_dotenv()

# ---------------setup your env-arguments--------------------

model = os.getenv("AZURE_DEPLOYED_MODEL")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_ENDPOINT")
api_version = os.getenv("API_VERSION")

if not azure_endpoint or not model:
    raise ValueError(
        "The value of azure endpoint or model is not set in the environment variables. Please check your .env file."
    )

client = AzureOpenAI(
    api_key=api_key, api_version=api_version, azure_endpoint=azure_endpoint
)


# -----------------Tool Functions------------------


def get_weather(latitude: float, longitude: float) -> str:
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )

    if response.status_code == 200:
        data = response.json()
        return data["current"]

    else:
        return "Could not retrieve weather data."


def call_function(function_name: str, arguments: dict):
    if function_name == "get_weather":
        return get_weather(**arguments)


# ----------------initial prompt----------------------
system_prompt = "You are a helpful assistant that can provide weather information based on latitude and longitude."

messages: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the weather like srinagar kashmir?"},
]

tools: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]


completion = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools,
)

print(completion.model_dump_json(indent=2))


# --------------- if tool required----------------

if completion.choices[0].message.tool_calls:
    tool_calls = completion.choices[0].message.tool_calls
    assistant_message = completion.choices[0].message

    messages.append(
        {
            "role": assistant_message.role,
            "content": assistant_message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        }
    )

    # Call the function with the tool call arguments
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        arguments = tool_call.function.arguments
        result = call_function(function_name, json.loads(arguments))

        # Append the tool call result to the messages
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            }
        )

# ------------------ Step 3: Final Assistant Response -------------------


class WeatherResponse(BaseModel):
    temperature: float = Field(
        description="The current temperature in celsius for the given location."
    )
    response: str = Field(
        description="A natural language response to the user's question."
    )


final_completion = client.beta.chat.completions.parse(
    model=model,
    messages=messages,
    tools=tools,
    response_format=WeatherResponse,
)

# --------------Check model response---------------------------

final_response = final_completion.choices[0].message.parsed
if final_response:
    print(final_response.temperature)
    print(final_response.response)
