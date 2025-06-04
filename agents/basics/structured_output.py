import os
from pydantic import BaseModel
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

#------------------------setup your env------------------------------

model=os.getenv("AZURE_DEPLOYED_MODEL")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("API-VERSION")
azure_endpoint= os.getenv("AZURE_ENDPOINT")

if not azure_endpoint or not model :
    raise ValueError(f"The value of azure endpoint in {azure_endpoint}")

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint
)

#-----------------Define response_format using pydantic------------------

class CalenderEvent(BaseModel):
    name: str
    date: str
    participants: list[str]
    
    
#-----------------call the model------------------------

completion = client.beta.chat.completions.parse(
    model=model, 
    messages=[
        {"role": "system", "content": "Extract the event information."},
        {
            "role": "user",
            "content": "Alice and Bob are going to a science fair on Friday.",
        },
    ],
    response_format=CalenderEvent
)

#-----------------parse the response-----------------------------

parsed_event = completion.choices[0].message.parsed

if parsed_event is not None:
    print(parsed_event.name)
    print(parsed_event.date)
    print(parsed_event.participants)
    



