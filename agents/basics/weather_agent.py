from openai import AzureOpenAI
from pydantic import BaseModel, Field
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from env_setup import env_vars
import requests
import json

client = AzureOpenAI(
    api_key=env_vars["api_key"],
    api_version=env_vars["api_version"],
    azure_endpoint=env_vars["azure_endpoint"],
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
    model=env_vars["model"],
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
    model=env_vars["model"],
    messages=messages,
    tools=tools,
    response_format=WeatherResponse,
)

# --------------Check model response---------------------------

final_response = final_completion.choices[0].message.parsed
if final_response:
    print(final_response.temperature)
    print(final_response.response)
