import os
import json
import openai


with open('../config.json') as user_file:
    config = json.load(user_file)

llm_api_key = config['llm_api_key']
llm_api_base = config['llm_api_base']
llm_api_base_local = config['llm_api_base_local']


client = openai.OpenAI(
    api_key=llm_api_key,
    base_url=llm_api_base,
)

text = 'I like cake'

result = client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

print(result)