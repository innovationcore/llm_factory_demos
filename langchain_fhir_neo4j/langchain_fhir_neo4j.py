import glob
import re

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from LoraXAPIEmbeddings import LoraXAPIEmbeddings

import json
from NEO4J_Graph import Graph
from FHIR_to_graph import resource_to_node, resource_to_edges

with open('../config.json') as user_file:
    config = json.load(user_file)

llm_api_key = config['llm_api_key']
llm_api_base = config['llm_api_base']
llm_api_base_local = config['llm_api_base_local']

llm = ChatOpenAI(
    model_name="",
    openai_api_key=llm_api_key,
    openai_api_base=llm_api_base_local,
    verbose=True,
    streaming=False
)

'''
llm = ChatOpenAI(
    openai_api_key=config['openai_api_key']
)
'''

embeddings = LoraXAPIEmbeddings(
    model="",
    api_key=llm_api_key,
    api_url=llm_api_base,
)


embeddings = OpenAIEmbeddings(
    openai_api_key=config['openai_api_key']
)


def get_database():

    # database
    NEO4J_URI = config['NEO4J_URI']
    NEO4J_USERNAME = config['NEO4J_USERNAME']
    NEO4J_PASSWORD = config['NEO4J_PASSWORD']
    NEO4J_DATABASE = config['NEO4J_DATABASE']

    graph = Graph(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE)

    return graph

def load_database(graph):

    print(graph.resource_metrics())
    print(graph.database_metrics())
    graph.wipe_database()

    # load database
    synthea_bundles = glob.glob("working/bundles/*.json")
    synthea_bundles.sort()

    nodes = []
    edges = []
    dates = set()  # set is used here to make sure dates are unique
    for bundle_file_name in synthea_bundles:
        print('processing:', bundle_file_name)
        with open(bundle_file_name) as raw:
            bundle = json.load(raw)
            for entry in bundle['entry']:
                resource_type = entry['resource']['resourceType']
                if resource_type != 'Provenance':
                    # generated the cypher for creating the resource node
                    nodes.append(resource_to_node(entry['resource']))
                    # generated the cypher for creating the reference & date edges and capture dates
                    node_edges, node_dates = resource_to_edges(entry['resource'])
                    edges += node_edges
                    dates.update(node_dates)
    count = 0
    # Alfonzo 422
    # create the nodes for resources
    for node in nodes:
        print('add node:', count)
        count +=1
        graph.query(node)

    count = 0

    date_pattern = re.compile(r'([0-9]+)/([0-9]+)/([0-9]+)')

    # Alfonzo 51
    # create the nodes for dates
    for date in dates:
        print('add date:', count)
        count += 1
        date_parts = date_pattern.findall(date)[0]
        cypher_date = f'{date_parts[2]}-{date_parts[0]}-{date_parts[1]}'
        cypher = 'CREATE (:Date {name:"' + date + '", id: "' + date + '", date: date("' + cypher_date + '")})'
        graph.query(cypher)

    count = 0

    #Alfonzo 2569
    # create the edges
    for edge in edges:
        try:
            print('add edge:', count)
            count += 1
            graph.query(edge)
        except:
            print(f'Failed to create edge: {edge}')

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



if __name__ == '__main__':

    #connect to neo4j
    print('Connecting to database:')
    graph = get_database()
    #populate the database
    print('Loading database')
    #load_database(graph)
    print('Creating vectors')
    #create_vectors()
    print('Creating index from vectors')
    #vector_index = create_vector_index()
    vector_index = create_context_vector_index()


    #prompts
    default_prompt = '''
    System: Use the following pieces of context to answer the user's question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ----------------
    {context}
    Human: {question}
    '''

    my_prompt = '''
    System: The following information contains entries about the patient. 
    Use the primary entry and then the secondary entries to answer the user's question.
    Each entry is its own type of data and secondary entries are supporting data for the primary one. 
    You should restrict your answer to using the information in the entries provided. 

    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ----------------
    {context}
    ----------------
    User: {question}
    '''

    my_prompt_2 = '''
    System: The context below contains entries about the patient's healthcare. 
    Please limit your answer to the information provided in the context. Do not make up facts. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    If you are asked about the patient's name and one the entries is of type patient, you should look for the first given name and family name and answer with: [given] [family]
    ----------------
    {context}
    Human: {question}
    '''

    template = """
            <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            {context}
            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            {question}
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """

    # Added prompt template
    #prompt = PromptTemplate(
    #    input_variables=["system_prompt", "user_prompt"],
    #    template=template
    #)
    #prompt = PromptTemplate.from_template(template)

    #prompt = PromptTemplate.from_template(my_prompt_2)

    template = """Answer the question in your own words as truthfully as possible from the context given to you.
    If you do not know the answer to the question, simply respond with "I don't know. Can you ask another question".
    If questions are asked where there is no relevant context available, simply respond with "I don't know. Please ask a question relevant to the documents"
    Context: {context}

    Human: {question}
    Assistant:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"], template=template
    )

    k_nearest = 200

    # question = "What can you tell me about Alfonso's claim created on 03/06/1977?"
    # question = "What can you tell me about the medical claim created on 03/06/1977?"
    # question = "Based on this explanation of benefits, how much did it cost and what service was provided?"
    # question = "Based on this explanation of benefits created on July 15, 2016, how much did it cost and what service was provided?"
    # question = "Based on this explanation of benefits created on March 6, 1978, how much did it cost and what service was provided?"
    # question = "Based on this explanation of benefits created on January 11, 2009, how much did it cost and what service was provided?"
    # question = "What was the blood pressure on 2/9/2014?"
    # question = "What was the blood pressure?"
    # question = "Based on this explanation of benefits created on January 18, 2014, how much did it cost and what service was provided?"
    # question = "How much did the colon scan eighteen days after the first of the year 2019 cost?"
    #question = "How much did the colon scan on Jan. 18, 2014 cost?"
    #question = "Did Alfonzo975 get a chest x-ray?"
    question = "What is the patient name of Alfonzo975?"
    '''
    response = vector_index.similarity_search(question,k=1)  # k_nearest is not used here because we don't have a retrieval query yet.
    print(response)
    print(response[0].page_content)
    print(len(response))
    '''

    vector_qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff",
        retriever=vector_index.as_retriever(search_kwargs={'k': 1}),
        # k_nearest is not used here because we don't have a retrieval query yet.
        verbose=True, chain_type_kwargs={"verbose": True, "prompt": prompt},
    )
    print(vector_qa.run(question))




