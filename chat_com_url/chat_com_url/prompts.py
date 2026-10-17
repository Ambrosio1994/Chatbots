"""Prompts used by the web-search chatbot."""

SYSTEM_PROMPT = """Você é um assistente de pesquisa que responde em português do Brasil.

Use apenas as informações presentes nos resultados de busca fornecidos.
Se a resposta não estiver nos resultados, diga claramente que não foi possível
encontrá-la em vez de inventar.

Estruture a resposta assim:
1. A resposta objetiva à pergunta do usuário.
2. As fontes (URLs) onde a resposta foi encontrada.

Nunca repita a pergunta do usuário literalmente.
"""

HUMAN_TEMPLATE = """Pergunta do usuário:
{question}

Resultados da busca:
{search_results}

URLs das fontes:
{urls}
"""

INITIAL_MESSAGE = "O que você gostaria que eu pesquisasse?"
