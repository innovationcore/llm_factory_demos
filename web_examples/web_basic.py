import json

from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
import gradio as gr


with open('../config.json') as user_file:
    config = json.load(user_file)

llm_api_key = config['llm_api_key']
llm_api_base = config['llm_api_base']
llm_api_base_local = config['llm_api_base_local']


llm = ChatOpenAI(
    model_name="",
    openai_api_key=llm_api_key,
    openai_api_base=llm_api_base,
    verbose=True
)

def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = llm(history_langchain_format)
    return gpt_response.content

gr.ChatInterface(predict).launch(share=True)