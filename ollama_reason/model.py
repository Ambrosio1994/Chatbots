from langchain_ollama import ChatOllama
from prompt import prompt
from langchain_core.output_parsers import StrOutputParser

def response(message):
    model = "deepseek-r1:1.5b"
    llm = ChatOllama(model=model,
                     temperature=0.6
    )

    chain = prompt | llm  | StrOutputParser()
    response = chain.invoke(str(message))
    print(response)
    return str(response)