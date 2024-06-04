#https://github.com/samschifman/RAG_on_FHIR/blob/main/RAG_on_FHIR_with_KG/FHIR_GRAPHS.ipynb

import glob
import re

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import json

from LoraXAPIEmbeddings import LoraXAPIEmbeddings
from NEO4J_Graph import Graph
from FHIR_to_graph import resource_to_node, resource_to_edges

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

'''
embeddings = OpenAIEmbeddings(
    openai_api_key=config['openai_api_key']
)
'''

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
    #synthea_bundles = glob.glob("working/bundles/*.json")
    synthea_bundles = glob.glob('/Users/cody/Downloads/synthea_sample_data_fhir_latest/*.json')
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


def date_for_question(question_to_find_date, model=""):


    system_content = f'''
    system:Given the following question from the user, extract the date the question is asking about.
    Return the answer formatted as JSON only, as a single line.
    Use the form:

    {{"date":"[THE DATE IN THE QUESTION]"}}

    Use the date format of month/day/year.
    Use two digits for the month and day.
    Use four digits for the year.
    So 3/4/23 should be returned as {{"date":"03/04/2023"}}.
    So 04/14/89 should be returned as {{"date":"04/14/1989"}}.

    Please do not include any special formatting characters, like new lines or "\\n".
    Please do not include the word "json".
    Please do not include triple quotes.

    If there is no date, do not make one up. 
    If there is no date return the word "none", like: {{"date":"none"}}
    '''

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_content,
            },
            {"role": "user",
             "content": question_to_find_date},
        ]
    )

    date_json = None
    try:
        date_json = json.loads(resp.choices[0].message.content)['date']
    except:
        None

    return date_json


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

if __name__ == '__main__':

    #connect to neo4j
    print('Connecting to database:')
    graph = get_database()
    #populate the database
    print('Loading database')
    #load_database(graph)
    #exit(0)
    #print('Creating vectors')
    create_vectors()


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

    #question = "What can you tell me about Alfonso's claim created on 03/06/1977?"
    #question = "What can you tell me about the medical claim created on 03/06/1977?"
    #question = "Based on this explanation of benefits, how much did it cost and what service was provided?"
    #question = "Based on this explanation of benefits created on July 15, 2016, how much did it cost and what service was provided?"
    #question = "Based on this explanation of benefits created on March 6, 1978, how much did it cost and what service was provided?"
    #question = "Based on this explanation of benefits created on January 11, 2009, how much did it cost and what service was provided?"
    #question = "What was the blood pressure on 2/9/2014?"
    #question = "What was the blood pressure?"
    #question = "Based on this explanation of benefits created on January 18, 2014, how much did it cost and what service was provided?"
    #question = "How much did the colon scan eighteen days after the first of the year 2019 cost?"
    #question = "How much did the colon scan on Jan. 18, 2014 cost?"
    #question = "Did Alfonzo975 get a chest x-ray?"
    #question = "What is heart rate of patient Alfonzo975 on 03/27/2015."
    #question = "Tell me about heart "
    #question = "Who many patients are males over the age of 50?"
    question = "Persons matching the following description: Inclusion Criteria:\n\n* admitted to Burke Rehabilitation Hospital for inpatient rehabilitation within 5 days after same-day or staged bilateral total knee arthroplasty;\n* 50-85 years of age;\n* able to read and understand English or a hospital-provided translator when consenting for the study;\n* free from contraindications for kinesiotaping (see below); and,\n* able to tolerate an active rehabilitation program.\n\n" \
               "Exclusion Criteria:\n\n* stage III or IV heart failure, stage III or IV renal failure;\n* fragile, very hairy or sensitive skin;\n* anesthesia or paraesthesia of any area of the lower extremity, except the surgical sites\n* active skin rashes or infections or skin lesions in the lower extremity;\n* prior history of allergic reactions to skin taping, bandaids, surgical tape; athletic tape or other skin-adhering electrode adhesives;\n* prior history of lower extremity lymphedema;3\n* prior history of lower extremity venous or arterial disease;\n* post-operative complications in the surgical sites;4\n* partial joint arthroplasty or revision arthroplasty of one or both knees;1,5\n* inability to give informed consent offered in English or through a hospital-provided translator\n* age less than 50 years or over 85 years;\n* inability to tolerate an active rehabilitation program."

    '''
    response = vector_index.similarity_search(question,k=1)  # k_nearest is not used here because we don't have a retrieval query yet.
    print(response)
    print(response[0].page_content)
    print(len(response))
    '''
    '''
    vector_qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff",
        retriever=vector_index.as_retriever(search_kwargs={'k': k_nearest}),
        # k_nearest is not used here because we don't have a retrieval query yet.
        verbose=True, chain_type_kwargs={"verbose": True, "prompt": prompt},
    )
    print(vector_qa.run(question))
    '''
    print('Creating index from vectors')

    date_str = date_for_question(question)
    if date_str is not None:
        print('index with date')
        vector_index = create_contextualized_vectorstore_with_date(date_str)
    else:
        print('index w/o date')
        #vector_index = create_vector_index()
        vector_index = create_context_vector_index()


    vector_qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff",
        retriever=vector_index.as_retriever(search_kwargs={'k': k_nearest}),
        # k_nearest is not used here because we don't have a retrieval query yet.
        verbose=True, chain_type_kwargs={"verbose": True, "prompt": prompt},
    )

    print(vector_qa.run(question))




