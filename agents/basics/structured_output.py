from pydantic import BaseModel
from openai import AzureOpenAI
from env_setup import env_vars


client = AzureOpenAI(
    api_key=env_vars["api_key"],
    api_version=env_vars["api_version"],
    azure_endpoint=env_vars["azure_endpoint"],
)

# -----------------Define response_format using pydantic------------------


class CalenderEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


# -----------------call the model------------------------

completion = client.beta.chat.completions.parse(
    model=env_vars["model"],
    messages=[
        {"role": "system", "content": "Extract the event information."},
        {
            "role": "user",
            "content": "Alice and Bob are going to a science fair on Friday.",
        },
    ],
    response_format=CalenderEvent,
)

# -----------------parse the response-----------------------------

parsed_event = completion.choices[0].message.parsed

if parsed_event is not None:
    print(parsed_event.name)
    print(parsed_event.date)
    print(parsed_event.participants)
