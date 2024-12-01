from llama_index.llms.anthropic import Anthropic
from llama_index.core.llms import ChatMessage

import streamlit as st
from dotenv import load_dotenv
import os 

load_dotenv()
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

st.set_page_config(page_title="QA", page_icon="🦙")
st.title("ChatBot Basic With Llama Index 🦙")


def response(user_query: str, temperature: float) -> str:
    """
    This function takes in a user query and a temperature value and calls 
    the Anthropic model to generate a response. The response is then returned.

    Args:
        user_query (str): The user query to be answered.
        temperature (float): The temperature of the Anthropic model.

    Returns:
        str: The response from the Anthropic model.
    """
    llm: Anthropic = Anthropic(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        temperature=temperature,
        api_key=ANTHROPIC_API_KEY
    )
    
    system_prompt: str = """
    You are an AI assistant specialized in answering questions. 
    Remember: you must always answer in Brazilian Portuguese and limit your answers 
    to 3 paragraphs maximum.

    Instructions:
    1. When asked a question, analyze it carefully.
    2. Limit your answers to 3 paragraphs maximum.
    """
    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_query)
    ]

    resp: str = llm.chat(messages).message.content
    return resp
    
message_init = "Hi, I'm your virtual assistant! How can I help you today?"

def clear_chat_history():
    """
    Resets the chat history to its initial state, containing only the initial
    message from the assistant.

    This function is intended to be used as a callback for a button, and is
    called when the button is clicked.
    """
    st.session_state.chat_history = [{"role": "assistant", "content": message_init}]

temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=1.0, value=0.0, step=0.1)
st.sidebar.button('Clear Chat History', on_click=clear_chat_history, key="one")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant",
         "content": message_init,
         }
    ]

for message in st.session_state.chat_history:
    if message["role"] == "assistant":
        with st.chat_message("AI", avatar="🤖"):
            st.write(message["content"])
    elif message["role"] == "human":
        with st.chat_message("Human", avatar="🤷"):
            st.write(message["content"])

user_query = st.chat_input("Digite sua duvida aqui...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append({"role": "human", "content": user_query})

    with st.chat_message("Human", avatar="🤷"):
        st.markdown(user_query)

    with st.chat_message("AI", avatar="🤖"):
        with st.spinner("Thinking..."):
            resp = response(user_query, temperature)
        st.write(resp)

    st.session_state.chat_history.append({"role": "assistant", "content": resp})