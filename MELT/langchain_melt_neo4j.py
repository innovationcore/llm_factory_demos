#https://github.com/samschifman/RAG_on_FHIR/blob/main/RAG_on_FHIR_with_KG/FHIR_GRAPHS.ipynb

import glob
import re
from neo4j_uploader import batch_upload

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import json

from LoraXAPIEmbeddings import LoraXAPIEmbeddings
from NEO4J_Graph import Graph

from openai import OpenAI


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

client = OpenAI(
        api_key=llm_api_key,
        base_url=llm_api_base,
    )


embeddings = LoraXAPIEmbeddings(
    model="",
    api_key=llm_api_key,
    api_url=llm_api_base,
)

neo4j_config = dict()
neo4j_config['neo4j_uri'] = config['NEO4J_URI']
neo4j_config['neo4j_user'] = config['NEO4J_USERNAME']
neo4j_config['neo4j_password'] = config['NEO4J_PASSWORD']
neo4j_config['overwrite'] = True

def get_database():

    # database
    NEO4J_URI = config['NEO4J_URI']
    NEO4J_USERNAME = config['NEO4J_USERNAME']
    NEO4J_PASSWORD = config['NEO4J_PASSWORD']
    NEO4J_DATABASE = config['NEO4J_DATABASE']

    graph = Graph(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE)

    return graph

def load_database(graph, combined_dataset):

    print(graph.resource_metrics())
    print(graph.database_metrics())
    graph.wipe_database()


    nodes = []
    edges = []

    #CREATE NODES
    for record in combined_dataset:
        resource = dict()
        resource['resource_type'] = 'Record'
        resource['resource_data'] = record_to_json_str(record)
        print(resource)
        nodes.append(resource_to_node(resource))

    count = 0
    # Alfonzo 422
    # create the nodes for resources
    for node in nodes:
        print('add node:', count)
        count +=1
        graph.query(node)

    # create the edges
    for edge in edges:
        try:
            print('add edge:', count)
            count += 1
            graph.query(edge)
        except:
            print(f'Failed to create edge: {edge}')


def create_contextualized_vectorstore_with_date(date_to_look_for):
    if date_to_look_for == 'none':
        contextualize_query_with_date = """
        match (node)<-[]->(sc:resource)
        with node.text as self, reduce(s="", item in collect(distinct sc.text) | s + "\n\nSecondary Entry:\n" + item ) as ctxt, score, {} as metadata limit 1
        return "Primary Entry:\n" + self + ctxt as text, score, metadata
        """
    else:
        contextualize_query_with_date = f"""
        match (node)<-[]->(sc:resource)
        where exists {{
             (node)-[]->(d:Date {{id: '{date_to_look_for}'}})
        }}
        with node.text as self, reduce(s="", item in collect(distinct sc.text) | s + "\n\nSecondary Entry:\n" + item ) as ctxt, score, {{}} as metadata limit 1
        return "Primary Entry:\n" + self + ctxt as text, score, metadata
        """

    _contextualized_vectorstore_with_date = Neo4jVector.from_existing_index(
        embeddings,
        url=config['NEO4J_URI'],
        username=config['NEO4J_USERNAME'],
        password=config['NEO4J_PASSWORD'],
        database=config['NEO4J_DATABASE'],
        index_name='fhir_text',
        retrieval_query=contextualize_query_with_date,
    )

    return _contextualized_vectorstore_with_date

def create_vectors():

    Neo4jVector.from_existing_graph(
        embeddings,
        url=config['NEO4J_URI'],
        username=config['NEO4J_USERNAME'],
        password=config['NEO4J_PASSWORD'],
        database=config['NEO4J_DATABASE'],
        index_name='fhir_text',
        node_label="resource",
        text_node_properties=['text'],
        embedding_node_property='embedding'
    )

def create_vector_index():

    vector_index = Neo4jVector.from_existing_index(
        embeddings,
        url=config['NEO4J_URI'],
        username=config['NEO4J_USERNAME'],
        password=config['NEO4J_PASSWORD'],
        database=config['NEO4J_DATABASE'],
        index_name='fhir_text'
    )
    return vector_index

def create_context_vector_index():
    contextualize_query = """
    match (node)<-[]->(sc:resource)
    with node.text as self, reduce(s="", item in collect(distinct sc.text) | s + "\n\nSecondary Entry:\n" + item ) as ctxt, score, {} as metadata limit 1
    return "Primary Entry:\n" + self + ctxt as text, score, metadata
    """

    contextualized_vectorstore = Neo4jVector.from_existing_index(
        embeddings,
        url=config['NEO4J_URI'],
        username=config['NEO4J_USERNAME'],
        password=config['NEO4J_PASSWORD'],
        database=config['NEO4J_DATABASE'],
        index_name='fhir_text',
        retrieval_query=contextualize_query,
    )

    return contextualized_vectorstore

def resource_to_node(resource):
    resource_type = resource['resource_type']
    resource_data = resource['resource_data']
    return f'CREATE (:{resource_type}:resource {resource_data})'

def resource_to_edges(resource):

    resource_type = resource['resourceType']
    resource_id = resource['id']

def record_to_json_str(record):

    output = '{ '

    for attrib in record:
        record[attrib] = record[attrib].replace('"','')
        record[attrib] = record[attrib].replace('\'', '')
        output += f'{attrib}: "{record[attrib]}",'

    output = output[:-1]
    output += '}'

    return output


if __name__ == '__main__':

    data = {
        "nodes": [
            {
                "labels": ["Person"],
                "key": "uid",
                "records": [
                    {
                        "uid": "abc",
                        "name": "John Bick"
                    },
                    {
                        "uid": "bcd",
                        "name": "Caney"
                    }
                ]
            },
            {
                "labels": ["Dog"],
                "key": "gid",
                "records": [
                    {
                        "gid": "abc",
                        "name": "Daisi"
                    }
                ]
            }
        ],
        "relationships": [
            {
                "type": "loves",
                "from_node": {
                    "record_key": "_from_uid",
                    "node_key": "uid",
                    "node_label": "Person"
                },
                "to_node": {
                    "record_key": "_to_gid",
                    "node_key": "gid",
                    "node_label": "Dog"
                },
                "exclude_keys": ["_from_uid", "_to_gid"],
                "records": [
                    {
                        "_from_uid": "abc",
                        "_to_gid": "abc"
                    }
                ]
            }
        ]
    }
    print('start')
    result = batch_upload(neo4j_config, data)
    print('end')

    exit()
    print('Loading dataset')
    combined_dataset_path = 'combined_dataset_test.json'

    with open(combined_dataset_path) as f:
        combined_dataset = json.load(f)

    #connect to neo4j
    print('Connecting to database:')
    graph = get_database()
    #populate the database
    print('Loading database')
    load_database(graph, combined_dataset)
    #exit(0)
    #print('Creating vectors')
    #create_vectors()







