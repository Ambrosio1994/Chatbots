# 🌐 Chatbot com URLs (LangChain + SearXNG + Claude / Ollama)

Chatbot de pesquisa na web que responde às perguntas do usuário citando as URLs
das fontes consultadas. Construído com **Streamlit**, **LangChain**, busca gratuita
e sem chave via **SearXNG**, e modelo **Anthropic Claude** (ou **Ollama** local).

## ✨ Funcionalidades

- **Busca na web sem chave de API** usando **SearXNG** (`SEARXNG_BASE_URL`),
  incluindo download e extração do conteúdo das páginas encontradas.
- **Memória de páginas da sessão + RAG (BM25)** — o conteúdo das páginas é salvo
  na memória da sessão ativa e, a cada pergunta, um RAG entrega ao LLM somente
  os trechos mais relevantes.
- **Memória seletiva do usuário** — retém apenas fatos, preferências e
  restrições do usuário (ignorando erros e blocos de raciocínio por padrão, a
  menos que o usuário habilite na barra lateral).
- **Respostas com fontes** — as URLs usadas aparecem em um painel expansível.
- **Resposta em streaming** — o texto aparece na tela enquanto é gerado, sem
  esperar a resposta inteira.
- **Execução 100% sem chaves de API** — selecione `ollama/deepseek-r1:1.5b` na
  barra lateral para rodar busca (SearXNG) + LLM (Ollama) localmente.

## 🏗️ Arquitetura

```
chat_com_url/
├── app.py                    # Ponto de entrada do Streamlit
├── chat_com_url/
│   ├── config.py             # Settings imutável lido do ambiente
│   ├── prompts.py            # Prompts do sistema e template humano
│   ├── search.py             # Integração com SearXNG + extração de páginas
│   ├── llm.py                # Cadeia LangChain (prompt → Claude/Ollama → parser)
│   ├── utils.py              # Normalização, RAG da sessão e memória seletiva
│   └── ui.py                 # Camada Streamlit
├── tests/                    # Testes unitários e de UI (sem chamadas de rede)
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── Dockerfile
└── .env.example
```

## 🛠️ Pré-requisitos

- Python 3.11+
- Uma instância do **SearXNG** com formato `json` habilitado (já configurada em
  `../docker-compose.yml` e `../searxng/settings.yml` na porta `8080`)
- Chave da API da **Anthropic** (`ANTHROPIC_API_KEY`) **ou** servidor **Ollama**
  local (`ollama/deepseek-r1:1.5b`)

## 🚀 Como rodar

```bash
cd chat_com_url
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
streamlit run app.py
```

## 🐳 Docker

Na raiz do repositório (`docker compose up --build`), o serviço `searxng` já sobe
automaticamente junto com o `chat-com-url`. Para rodar isoladamente:

```bash
docker build -t chat-com-url .
docker run --rm -p 8501:8501 \
  -e SEARXNG_BASE_URL=http://host.docker.internal:8080 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  chat-com-url
```

## 🔧 Variáveis de ambiente

| Variável | Padrão | Descrição |
| --- | --- | --- |
| `SEARXNG_BASE_URL` | `http://localhost:8080` | URL da instância SearXNG |
| `SEARXNG_MAX_RESULTS` | `3` | Quantidade de páginas buscadas (1 a 5) |
| `ANTHROPIC_API_KEY` | — | Chave da API da Anthropic (opcional se usar `ollama/...`) |
| `ANTHROPIC_MODEL` | `claude-haiku-5-5` | Modelo (`claude-*` ou `ollama/deepseek-r1:1.5b`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL do servidor Ollama local |
| `ANTHROPIC_MAX_TOKENS` | `1000` | Limite de tokens da resposta |
| `ANTHROPIC_TEMPERATURE` | `0.1` | Temperatura padrão (0.1 a 1.0) |
| `CHAT_RAG_TOP_K` | `4` | Quantidade de trechos relevantes enviados pelo RAG |

## ✅ Testes e lint

```bash
pytest
ruff check .
ruff format --check .
```

## 🧰 Stack

Streamlit · LangChain · SearXNG · Anthropic Claude · Ollama · python-dotenv · pytest · ruff
