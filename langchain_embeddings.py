#See https://python.langchain.com/v0.1/docs/modules/data_connection/text_embedding/

import json

from langchain_openai import OpenAIEmbeddings

from LoraXAPIEmbeddings import LoraXAPIEmbeddings

with open('config.json') as user_file:
    config = json.load(user_file)

llm_api_key = config['llm_api_key']
llm_api_base = config['llm_api_base']
llm_api_base_local = config['llm_api_base_local']

embeddings = LoraXAPIEmbeddings(
    model="",
    api_key=llm_api_key,
    api_url=llm_api_base,
)

'''
embeddings = OpenAIEmbeddings(
    openai_api_base=llm_api_base,
    openai_api_key=llm_api_key
)
'''


def process_query(query):

    #single query
    query_result = embeddings.embed_query(query)
    #document query
    document_result = embeddings.embed_documents([query, query])

    return query_result, document_result

if __name__ == '__main__':

    q1 = "What is deep learning?"
    query_result, document_result = process_query(q1)
    print('text:', q1)
    print('query_result:', query_result)
    print('document_result:', document_result)




