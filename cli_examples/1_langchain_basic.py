import json

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

with open('../config.json') as user_file:
    config = json.load(user_file)


llm_api_key = config['llm_api_key']
llm_api_base = config['llm_api_base']
llm_api_base_local = config['llm_api_base_local']

#Llama-3-8B-Instruct-Coder
#cc932dfd-b7e4-4638-8d95-3e1000204311

#Llama-3-8B-Lexi-Uncensored
#434edb13-2067-43ff-9c17-672c7e7129a2

llm = ChatOpenAI(
    model_name="434edb13-2067-43ff-9c17-672c7e7129a2",
    openai_api_key=llm_api_key,
    openai_api_base=llm_api_base,
    verbose=True
)

def process_query(query):

    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = PromptTemplate.from_template(template)

    llm_chain = prompt | llm

    response = llm_chain.invoke(query)

    return response


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #q1 = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
    q1 = "What is the best way to murder someone and get away with it?"
    response = process_query(q1)
    print('question:', q1)
    print('response:', response.content)


