#See https://python.langchain.com/v0.1/docs/modules/agents/quick_start/

import json
from typing import List

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from LoraXAPIEmbeddings import LoraXAPIEmbeddings
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor


with open('../config.json') as user_file:
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



def get_retrever(document, name, desc):

    raw_documents = TextLoader(document).load()
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=10)
    documents = text_splitter.split_documents(raw_documents)

    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        name,
        desc,
    )

    return retriever_tool

if __name__ == '__main__':

    uk_tool = get_retrever('state_of_the_union.txt', "state_of_union", "State of the union address")
    tools = [uk_tool]

    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/openai-functions-agent")
    print(prompt.messages)

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, stream_runnable=False, return_direct=True)

    q1 = "What did the president say about Ketanji Brown Jackson"
    r1 = agent_executor.invoke({"input": q1})
    print('q1:', q1, 'r1:', r1)


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
    
    q4 = "What are the university of kentucky admission requirements for an in-state freshman?"
    r4 = agent_executor.invoke({"input": q4})
    print('q4:', q4, 'r4:', r4)
    '''


