# 🤖 Chatbots

Coleção de três chatbots independentes construídos com **Streamlit**, explorando
diferentes formas de orquestrar modelos de linguagem (LLMs): APIs de nuvem,
busca na web com RAG e execução local com modelos de raciocínio.

Cada projeto é autocontido (dependências, testes, Docker e README próprios) e
segue a mesma arquitetura em camadas, para facilitar a leitura e a comparação
entre as abordagens.

## 📦 Projetos

| Projeto | Descrição | Orquestração | Provedor |
| --- | --- | --- | --- |
| [chatbasico](./chatbasico) | Chatbot conversacional com controle de temperatura e memória de conversa. | [LlamaIndex](https://www.llamaindex.ai/) | Anthropic Claude |
| [chat_com_url](./chat_com_url) | Chatbot que pesquisa na web com SearXNG (sem chave de API de busca), salva páginas na memória da sessão com RAG e responde citando as URLs. | [LangChain](https://python.langchain.com/) | Anthropic Claude / Ollama + [SearXNG](https://docs.searxng.org/) |
| [ollama_reasoning](./ollama_reasoning) | Chatbot local que expõe o "raciocínio" de um modelo DeepSeek-R1. | [LangChain](https://python.langchain.com/) | [Ollama](https://ollama.com/) |

## 🏗️ Arquitetura

Todos os projetos compartilham a mesma separação de responsabilidades, o que
mantém a lógica de negócio testável e independente da interface:

```
<projeto>/
├── app.py                 # Ponto de entrada do Streamlit
├── <projeto>/             # Pacote da aplicação
│   ├── config.py          # Configuração + variáveis de ambiente
│   ├── prompts.py         # Prompts do sistema
│   ├── llm.py             # Orquestração do modelo (sem Streamlit)
│   ├── search.py          # Ferramentas externas (quando aplicável)
│   ├── utils.py           # Helpers puros e testáveis
│   └── ui.py              # Camada de interface (Streamlit)
├── tests/                 # Testes unitários (pytest)
├── requirements.txt       # Dependências de runtime
├── requirements-dev.txt   # Dependências de desenvolvimento
├── pyproject.toml         # Configuração de ruff e pytest
├── Dockerfile
└── .env.example
```

Princípios adotados:

- **Lógica separada da UI** — o Streamlit fica isolado em `ui.py`; o resto pode
  ser testado sem subir interface.
- **Configuração por ambiente** — nada de *hardcode* de chaves; tudo via `.env`.
- **Sem acoplamento a versões antigas** — modelos e bibliotecas atualizados.
- **Qualidade automatizada** — `ruff` para lint/formatação e `pytest` para testes.

## 🚀 Como rodar

Cada projeto tem instruções próprias no seu README. O fluxo geral é:

```bash
cd <projeto>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env        # preencha as chaves necessárias
streamlit run app.py
```

## 🐳 Docker

Para subir os três apps de uma vez:

```bash
export ANTHROPIC_API_KEY=...   # e demais chaves necessárias
docker compose up --build
```

| Serviço | URL |
| --- | --- |
| chatbasico | http://localhost:8501 |
| chat_com_url | http://localhost:8502 |
| ollama_reasoning | http://localhost:8503 |

> Para o `ollama_reasoning` fora do Docker, é preciso ter o
> [Ollama](https://ollama.com/) rodando localmente e baixar o modelo:
> `ollama pull deepseek-r1:1.5b`.

## ✅ Qualidade

- **Lint e formatação:** `ruff check .` e `ruff format --check .`
- **Testes:** `pytest`
- **CI:** GitHub Actions roda lint + testes dos três projetos a cada push/PR
  (veja [`.github/workflows/ci.yml`](./.github/workflows/ci.yml)).

## 🧰 Stack

Python 3.11+ · Streamlit · LlamaIndex · LangChain · Anthropic Claude · SearXNG ·
Ollama · Docker · GitHub Actions
