import os
import streamlit as st

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

st.set_page_config(page_title="Chatbot com URLs", 
                   page_icon="🌐🔗")
st.title("🌐🔗 Chatbot com URLs")

# Entrada das chaves da API
st.sidebar.title("Permissões")
ANTHROPIC_API_KEY = st.sidebar.text_input("Digite sua chave ANTHROPIC:", type="password")
TAVILY_API_KEY = st.sidebar.text_input("Digite sua chave TAVILY:", type="password")

if not ANTHROPIC_API_KEY or not TAVILY_API_KEY:
    st.info("Por favor, insira suas chaves ANTHROPIC e TAVILY para continuar.")
    st.stop()

# Configurar variáveis de ambiente
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

def get_response(model, temperature, max_tokens,
                max_results, query):
    
    # Configurar modelo Anthropic
    llm = ChatAnthropic(model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        api_key=ANTHROPIC_API_KEY)

    # Configurar prompt
    system = """
    use os resultados encontrados pela ferramenta de consulta para responder
    a pergunta {pergunta} do usuario.

    estruture sua resposta da seguinte forma:

    1. De a resposta para pergunta do usuario com base nos resultados encontrados {seachs}
    2. urls onde foi encontrado a resposta: \n\n {urls}

    ##ATENÇÃO##
    Não inclua a pergunta feita pelo usuario em sua resposta
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
            "Aqui esta a pergunta do usuario \n{pergunta}. \
            \n\n\
            Aqui estão os resultados da consulta: \n{seachs} \
            \n\n\
            Aqui estão as urls: \n{urls}")
        ]
    )

    # Encadeamento do prompt com o modelo e parser
    chain = prompt | llm | StrOutputParser()

    weeb_search = TavilySearchResults(max_results=max_results,
                                      tavily_api_key=TAVILY_API_KEY)
    
    # Consultar TavilySearchResults
    searchs = weeb_search.invoke({"query": query})
    content = "\n".join([s["content"] for s in searchs])
    urls = [s["url"] for s in searchs]

    # Gerar resposta
    response = chain.invoke({"pergunta": query,
                             "seachs": content,
                             "urls": urls})
    return response

msg_init = "O que você gostaria que eu pesquisasse?"
model_names = ["claude-3-haiku-20240307",
               "claude-3-5-haiku-20241022",
               "claude-3-sonnet-20240229",
               "claude-3-5-sonnet-20241022",
               "claude-3-opus-20240229",
               ]

model = st.sidebar.selectbox("Modelo", model_names)

temperature = st.sidebar.slider('temperature', min_value=0.1, max_value=1.0, value=0.1, step=0.1,
                                help="Controla a criatividade da resposta do modelo")

max_tokens = st.sidebar.slider('max_tokens', min_value=10, max_value=2000, value=1000, step=10,
                               help="Quantidade de tokens a serem gerados pela resposta")

max_results = st.sidebar.slider('max_results', min_value=1, max_value=5, value=3, step=1,
                                help="Quantidade de URLs a serem exibidas na resposta")

# Função para limpar o histórico
def clear_chat_history():
    st.session_state.chat_history = [{"role": "assistant", 
                                  "content": msg_init}]

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Configuração inicial do histórico
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content=msg_init)
    ]

# Exibir histórico de mensagens
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI", avatar="🌐"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human", avatar="🤷"):
            st.write(message.content)

# Entrada de nova mensagem do usuário
user_query = st.chat_input(msg_init)
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human", avatar="🤷"):
        st.markdown(user_query)

    # Gerar e exibir resposta
    with st.chat_message("AI", avatar="🌐"):
        resp = get_response(model, temperature, max_tokens,
                            max_results, user_query)
        st.write(resp)

    # Adicionar resposta ao histórico
    st.session_state.chat_history.append(AIMessage(content=resp))