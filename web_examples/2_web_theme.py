import json

from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
import gradio as gr
from langchain_core.messages import SystemMessage

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


def predict(message, history, system_prompt="", temperature=0.9, max_tokens=256, top_p=0.6):

    history_langchain_format = []
    if len(system_prompt) > 0:
        history_langchain_format.append(SystemMessage(content=system_prompt))
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = llm(history_langchain_format,
                       temperature=temperature,
                       max_tokens=max_tokens,
                       top_p=top_p
                       )
    return gpt_response.content

additional_inputs=[
    gr.Textbox("You are a helpful AI assistant, finish each response with the phrase \"Brought to you by Carl's Jr.\"", label="Optional system prompt", interactive=True),
    gr.Slider(
        label="Temperature",
        value=0.9,
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        interactive=True,
        info="Higher values produce more diverse outputs",
    ),
    gr.Slider(
        label="Max new tokens",
        value=256,
        minimum=0,
        maximum=4096,
        step=64,
        interactive=True,
        info="The maximum numbers of new tokens",
    ),
    gr.Slider(
        label="Top-p (nucleus sampling)",
        value=0.6,
        minimum=0.0,
        maximum=1,
        step=0.05,
        interactive=True,
        info="Higher values sample more low-probability tokens",
    )
]

demo = gr.ChatInterface(predict, theme='gradio/monochrome', additional_inputs=additional_inputs)

if __name__ == "__main__":
    demo.queue().launch()