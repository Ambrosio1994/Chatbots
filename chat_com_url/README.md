# 🌐 Chatbot com URLs (LangChain + Tavily)

Chatbot que **pesquisa na web** e responde **citando as fontes**, construído com
**Streamlit**, **LangChain**, o modelo **Anthropic Claude** e a API de busca
**Tavily**. Faz um pequeno pipeline de *retrieval-augmented generation* (RAG):
busca → monta o prompt com os resultados → gera a resposta.

## ✨ Funcionalidades

- **Busca na web em tempo real** com a integração oficial `langchain-tavily`
  (`TavilySearch`, que substitui o antigo `TavilySearchResults`).
- **Respostas com fontes** — as URLs usadas aparecem em um painel expansível.
- **Resposta em streaming** — o texto aparece na tela enquanto é gerado, sem
  esperar a resposta inteira.
- **Chaves por interface ou ambiente** — informe no `.env` ou na barra lateral.
- **Modelo e parâmetros configuráveis** — modelo, temperatura, tokens e número
  de resultados.
- **Histórico de conversa** com botão para limpar.

## 🏗️ Arquitetura

```
chat_com_url/
├── app.py                    # Ponto de entrada do Streamlit
├── chat_com_url/
│   ├── config.py             # Settings via variáveis de ambiente
│   ├── prompts.py            # Prompts (system + human)
│   ├── search.py             # Integração com o Tavily
│   ├── llm.py                # Chain: prompt → modelo → parser
│   ├── utils.py              # Normalização de resultados (testável)
│   └── ui.py                 # Camada Streamlit
├── tests/
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── Dockerfile
```

O pipeline é: **`run_search`** devolve `SearchResult`s normalizados →
**`build_prompt_variables`** monta o prompt → **`answer_question`** roda a chain
do LangChain. Toda essa lógica é testável sem chamadas de rede.

## 🛠️ Pré-requisitos

- Python 3.11+
- Chave da [Anthropic](https://console.anthropic.com/)
- Chave da [Tavily](https://app.tavily.com/)

## 🚀 Como rodar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # preencha ANTHROPIC_API_KEY e TAVILY_API_KEY
streamlit run app.py
```

A aplicação abre em <http://localhost:8501>.

## 🐳 Docker

```bash
docker build -t chatbots/chat-com-url .
docker run --rm -p 8501:8501 --env-file .env chatbots/chat-com-url
```

## 🔧 Variáveis de ambiente

| Variável | Obrigatória | Padrão | Descrição |
| --- | :---: | --- | --- |
| `ANTHROPIC_API_KEY` | ✅ | — | Chave da API da Anthropic. |
| `TAVILY_API_KEY` | ✅ | — | Chave da API da Tavily. |
| `ANTHROPIC_MODEL` | — | `claude-haiku-5-5` | Modelo usado pelas respostas. |
| `ANTHROPIC_MAX_TOKENS` | — | `1000` | Limite de tokens de saída. |
| `ANTHROPIC_TEMPERATURE` | — | `0.1` | Criatividade inicial do modelo. |
| `TAVILY_MAX_RESULTS` | — | `3` | Quantidade de resultados buscados. |

## ✅ Testes e lint

```bash
pytest
ruff check .
ruff format --check .
```

## 🧰 Stack

Streamlit · LangChain · Anthropic Claude · Tavily · python-dotenv · pytest · ruff
