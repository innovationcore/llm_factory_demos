
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
    def search_function(input: str) -> str:
        print('BINGO!!!')
        """Applies a magic function to an input."""
        return str + ' CODY IS COOL!'


    search = StructuredTool.from_function(
        func=search_function,
        name="Search",
        description="useful for when you need to answer questions about current events",
        # coroutine= ... <- you can specify an async method if desired as well
    )

    tools = [search]

    llm_api_base_local='http://10.33.31.21:8000/v1'

    llm = ChatOpenAI(
        model_name="/models/functionary-small-v2.5",
        openai_api_key=llm_api_key,
        openai_api_base=llm_api_base_local,
        verbose=True,
        streaming=False,
        #tools=tools,
        #tool_choice="Search"
    )


    #llm.bind_tools(tools=tools, tool_choice="any")

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, stream_runnable=False)

    agent_executor.invoke({"input": "What is a current event? Use tool when phrases like 'current events', 'Tell me the news' are used. "})

