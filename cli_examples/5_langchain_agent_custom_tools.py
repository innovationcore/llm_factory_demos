
import json

from langchain_core.tools import StructuredTool
from langchain_caai.caai_emb_client import caai_emb_client

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain_core.prompts import ChatPromptTemplate

with open('../config.json') as user_file:
    config = json.load(user_file)

llm_api_key = config['llm_api_key']
llm_api_base = config['llm_api_base']
llm_api_base_local = config['llm_api_base_local']


embeddings = caai_emb_client(
    model="",
    api_key=llm_api_key,
    api_url=llm_api_base,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


if __name__ == '__main__':
    def search_function_old(input: str) -> str:
        print('Incoming request:', input)
        response = "patient 12345 lives on memory lane, and has a rare form of blood disorder that makes the like UofL"
        """Applies a magic function to an input."""
        return response

    def search_function(input: str) -> dict:
        print('Incoming request:', input)
        response = dict()
        response['patient'] = input
        response['address'] = '123 Memory Lane, Lexington Kentucky 40504'
        response['conditions'] = 'Addicted to UK basketball and Ale-8'
        response['events'] = 'Once encountered Big Foot'
        return response

    search = StructuredTool.from_function(
        func=search_function,
        name="Search",
        description="useful for when you need to look up patient information",
    )

    tools = [search]

    llm = ChatOpenAI(
        model_name="/models/functionary-small-v2.5",
        openai_api_key=llm_api_key,
        openai_api_base=llm_api_base_local,
        verbose=True,
        streaming=False,
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, stream_runnable=False)

    #q1 = agent_executor.invoke({"input": "Tell me what you know about patient 12345"})
    #print(q1)

    q2 = agent_executor.invoke({"input": "Has patient 12345 encountered any supernatural beings, provide only the answer to the question?"})
    print(q2)