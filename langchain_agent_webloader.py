#See https://python.langchain.com/v0.1/docs/modules/agents/quick_start/

import json

from langchain_openai import OpenAIEmbeddings
from LoraXAPIEmbeddings import LoraXAPIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor


with open('config.json') as user_file:
    config = json.load(user_file)

llm_api_key = config['llm_api_key']
llm_api_base = config['llm_api_base']
llm_api_base_local = config['llm_api_base_local']

llm = ChatOpenAI(
    model_name="",
    openai_api_key=llm_api_key,
    openai_api_base=llm_api_base,
    verbose=True,
    streaming=False
)

embeddings = LoraXAPIEmbeddings(
    model="",
    api_key=llm_api_key,
    api_url=llm_api_base,
)

'''
embeddings = OpenAIEmbeddings(
    openai_api_base=llm_api_base,
    openai_api_key=llm_api_key,
    tiktoken_model_name="cl100k"
)
'''

def get_tools(url):

    loader = WebBaseLoader(url)
    docs = loader.load()
    documents = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "langsmith_search",
        "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
    )

    return [retriever_tool]

if __name__ == '__main__':

    url = "https://docs.smith.langchain.com/overview"
    tools = get_tools(url)

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

    q4 = "what are the latest Dow numbers"
    r4 = agent_executor.invoke({"input": q4})
    print('q4:', q4, 'r4:', r4)


