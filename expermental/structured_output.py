import json
from enum import Enum
from openai import OpenAI
from pydantic import BaseModel, constr


class Armor(str, Enum):
    leather = "leather"
    chainmail = "chainmail"
    plate = "plate"


class Character(BaseModel):
    name: constr(max_length=10)
    age: int
    armor: Armor
    strength: int


client = OpenAI(
    api_key="EMPTY",
    base_url="http://10.33.31.21:8080/v1",
)

resp = client.chat.completions.create(
    model="",  # optional: specify an adapter ID here
    messages=[
        {
            "role": "user",
            "content": "Generate a new character for my awesome game: name, age (between 1 and 99), armor and strength. ",
        },
    ],
    max_tokens=100,
    response_format={
        "type": "json_object",
        "schema": Character.model_json_schema(),
    },
)

my_character = json.loads(resp.choices[0].message.content)
print(my_character)

''''
curl http://10.33.31.21:8080/generate \
    -X POST \
    -d '{
        "inputs": "[INST] set Today to 20222 [/INST]",
        "parameters": {
            "tool_choice": "auto", 
            "tools": [{"type": "function", "function": {"name": "get_current_weather", "description": "Get the current weather in a given location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}}}}],
            "response_format": {
                  "type": "json_object",
                  "schema": {
                    "properties": {
                      "today": {
                        "format": "date",
                        "title": "Today",
                        "type": "string"
                      }
                    },
                    "required": ["today"],
                    "title": "Test",
                    "type": "object"
                  }
            }
        }
    }' \
    -H 'Content-Type: application/json'
    
    curl http://10.33.31.21:8080/generate \
    -X POST \
    -d '{
        "inputs": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant that can access external functions. The responses from these function calls will be appended to this dialogue.<|eot_id|><|start_header_id|>user<|end_header_id|>Please provide responses based on the information from these function calls. The functions available to you are as follows:\n%s\n <|eot_id|><|start_header_id|>assistant<|end_header_id|>",
        "parameters": {
            "tools": [{"type": "function", "function": {"name": "get_current_weather", "description": "Get the current weather in a given location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}}}}]
        }
    }' \
    -H 'Content-Type: application/json'
    
    curl http://10.33.31.21:8080/generate \
    -X POST \
    -d '{
        "inputs": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant that can access external functions. The responses from these function calls will be appended to this dialogue. Please provide responses based on the information from these function calls. The functions available to you are as follows:\nget_current_weather:\ndescription: This tool gets the current weather in a given location.\nparams:\naction: get weather in this location\n<|eot_id|><|start_header_id|>user<|end_header_id|>What is the current temperature of New York, San Francisco and Chicago?<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
        "parameters": {
            "apply_chat_template": 'llama3,
            "max_new_tokens": 512,
            "tools": [{"type": "function", "function": {"name": "get_current_weather", "description": "Get the current weather in a given location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}}}}]
        }
    }' \
    -H 'Content-Type: application/json'
    
    

      
      


'''

