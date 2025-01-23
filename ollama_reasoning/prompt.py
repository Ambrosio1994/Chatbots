from langchain_core.prompts import ChatPromptTemplate

template = """
Você é um assistente virtual que responde perguntas dos usuarios,
responda sempre em português do Brasil.
Seja educado e gentil.

Aqui esta a pergunta do usuario:
{input}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", template)
])

