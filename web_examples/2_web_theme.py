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


def predict(message, history, system_prompt="", adapter="", temperature=0.9, max_tokens=256, top_p=0.6):

    if adapter is None:
        adapter = ""

    print('adapter:', adapter)

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
                       top_p=top_p,
                       model=adapter,
                       )
    return gpt_response.content

prompt = 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.\nPlease ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information.'

'''
#Llama-3-8B-Instruct-Coder
#cc932dfd-b7e4-4638-8d95-3e1000204311

#Llama-3-8B-Lexi-Uncensored
#434edb13-2067-43ff-9c17-672c7e7129a2

#Hermes-2-Theta-Llama-3-8B-32k
#ba849661-ece6-4419-b309-0ed2c0db1b8c

#L3-8B-Stheno-v3.3-32K
#c5dcf4de-2aa8-46c0-b355-90e92334f15c

#Meta-Llama-3-8B-Instruct-abliterated-v3
#ecb33662-7f37-48b7-b273-20d72b2e90b0
'''

additional_inputs=[
    gr.Textbox("", label="Optional system prompt", interactive=True),
    gr.Dropdown(
        choices=[
                 ('Llama-3-8B-Instruct',''),
                 ('Llama-3-8B-Instruct-Coder','cc932dfd-b7e4-4638-8d95-3e1000204311'),
                 ('Llama-3-8B-Lexi-Uncensored', '434edb13-2067-43ff-9c17-672c7e7129a2'),
                 ('Hermes-2-Theta-Llama-3-8B-32k','ba849661-ece6-4419-b309-0ed2c0db1b8c'),
                 ('L3-8B-Stheno-v3.3-32K','c5dcf4de-2aa8-46c0-b355-90e92334f15c'),
                 ('Meta-Llama-3-8B-Instruct-abliterated-v3','ecb33662-7f37-48b7-b273-20d72b2e90b0'),
                 ('Neo4jCypherQUERY','851b3a9b-21b2-4a59-b3b9-1940c3584023'),
                 ],
        label="adapter",
        interactive=True,
        info="Custom Adapter",
    ),
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

disclaimer = '* Disclaimer: The output and responses generated by this chatbot are not endorsed or supported by the hosting institution. This chatbot is provided for experimental purposes only and is intended to facilitate conversations, generate ideas, and demonstrate AI capabilities.'

demo = gr.ChatInterface(predict,
                        theme='gradio/monochrome',
                        additional_inputs=additional_inputs,
                        description=disclaimer)

if __name__ == "__main__":
    demo.queue().launch()