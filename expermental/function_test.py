import os
import json
import openai

client = openai.OpenAI(
    api_key="EMPTY",
    base_url="http://10.33.31.21:8000/v1",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": [
                            "celsius",
                            "fahrenheit"
                        ]
                    }
                }
            }
        }
    }
]
messages = [
    {"role": "system", "content": "You are a helpful assistant that can access external functions. The responses from these function calls will be appended to this dialogue. Please provide responses based on the information from these function calls."},
    {"role": "user", "content": "What is the current temperature of New York, San Francisco and Chicago?"}
]
response = client.chat.completions.create(
    model="/models/functionary-small-v2.5",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)
#print(response)
#print(response.choices[0])
#print(json.dumps(response.choices[0].message.model_dump()['tool_calls'], indent=2))
print(json.dumps(response.choices[0].message.model_dump(), indent=2))
