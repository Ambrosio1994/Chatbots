import streamlit as st
from model import response

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
            resp = response(user_query)
        st.write(resp)

    st.session_state.chat_history.append({"role": "assistant", "content": resp})