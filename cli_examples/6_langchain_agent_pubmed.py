#See https://python.langchain.com/v0.1/docs/integrations/retrievers/pubmed/

import json
from langchain_caai.caai_emb_client import caai_emb_client
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_community.retrievers import PubMedRetriever


with open('../config.json') as user_file:
    config = json.load(user_file)

llm_api_key = config['llm_api_key']
llm_api_base = config['llm_api_base']
llm_api_base_local = config['llm_api_base_local']

llm = ChatOpenAI(
    model_name="/models/functionary-small-v2.5",
    openai_api_key=llm_api_key,
    openai_api_base=llm_api_base_local,
    verbose=True,
    streaming=False
)

embeddings = caai_emb_client(
    model="",
    api_key=llm_api_key,
    api_url=llm_api_base,
    max_batch_size=100,
    num_workers=10
)


def get_tools():

    retriever = PubMedRetriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "pubmed_search",
        "Search for information about pubmed articles",
    )

    return [retriever_tool]

if __name__ == '__main__':

    tools = get_tools()

    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/openai-functions-agent")
    print(prompt.messages)

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, stream_runnable=False)

    '''
    q1 = "hi!"
    r1 = agent_executor.invoke({"input": q1})
    print('q1:', q1, 'r1:', r1)

    q2 = "how can langsmith help with testing?"
    r2 = agent_executor.invoke({"input": q2})
    print('q2:', q2, 'r2:', r2)

    q3 = "whats the weather in lexington, kentucky?"
    r3 = agent_executor.invoke({"input": q3})
    print('q3:', q3, 'r3:', r3)
    '''

    q4 = "Search PubMed for 25 articles related to ChatGPT, format output in JSON"
    r4 = agent_executor.invoke({"input": q4})
    print(type(q4))
    print('q4:', q4, 'r4:', r4)


