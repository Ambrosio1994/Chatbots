# 🦙 Chatbot Básico (LlamaIndex + Anthropic)

Chatbot conversacional construído com **Streamlit**, **LlamaIndex** e os modelos
**Anthropic Claude**. Responde em português do Brasil, mantém o histórico da
conversa e permite ajustar a criatividade das respostas em tempo real.

## ✨ Funcionalidades

- **Respostas em streaming** — o texto aparece à medida que é gerado.
- **Modelo avaliador de memória dedicado + RAG (BM25)** — um modelo separado avalia o prompt do usuário e grava automaticamente apenas fatos, preferências e restrições duráveis na memória da sessão (sem exibir opções de memória na interface), recuperando os turnos mais relevantes a cada pergunta.
- **Controle de temperatura** — slider no menu lateral (0.0 a 1.0).
- **Seleção de modelo** — seletor com modelos do LlamaIndex (`llama3.2:1b`, `llama3.2`, `deepseek-r1:1.5b`, `qwen2.5`) e campos opcionais para informar **Outro modelo** e **Chave da API**.
- **Limpar conversa** — reinicia a sessão com um clique.
- **Tratamento de erros** — falhas de API são exibidas de forma amigável.

## 🏗️ Arquitetura

```
chatbasico/
├── app.py                    # Ponto de entrada do Streamlit
├── chatbasico/
│   ├── config.py             # Settings via variáveis de ambiente
│   ├── prompts.py            # Prompt do sistema
│   ├── llm.py                # Criação do cliente + geração/streaming
│   ├── utils.py              # Helpers puros (testáveis)
│   └── ui.py                 # Camada Streamlit
├── tests/                    # Testes com pytest
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml            # Configuração de ruff e pytest
└── Dockerfile
```

A lógica de LLM (`llm.py`) não conhece o Streamlit, o que permite testá-la com
um cliente falso e sem chamadas de rede.

## 🛠️ Pré-requisitos

- Python 3.11+
- Uma chave de API da [Anthropic](https://console.anthropic.com/)

## 🚀 Como rodar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # preencha ANTHROPIC_API_KEY
streamlit run app.py
```

A aplicação abre em <http://localhost:8501>.

## 🐳 Docker

```bash
docker build -t chatbots/chatbasico .
docker run --rm -p 8501:8501 --env-file .env chatbots/chatbasico
```

## 🔧 Variáveis de ambiente

| Variável | Obrigatória | Padrão | Descrição |
| --- | :---: | --- | --- |
| `ANTHROPIC_API_KEY` | ✅ | — | Chave da API da Anthropic. |
| `ANTHROPIC_MODEL` | — | `claude-haiku-5-5` | Modelo usado pelas respostas. |
| `ANTHROPIC_MAX_TOKENS` | — | `1024` | Limite de tokens de saída. |
| `ANTHROPIC_TEMPERATURE` | — | `0.0` | Criatividade inicial do modelo. |

## ✅ Testes e lint

```bash
pytest
ruff check .
ruff format --check .
```

## 🧰 Stack

Streamlit · LlamaIndex · Anthropic Claude · python-dotenv · pytest · ruff
