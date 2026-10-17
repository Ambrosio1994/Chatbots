# 🧠 Chatbot com Raciocínio (Ollama + DeepSeek-R1)

Chatbot que roda **100% na sua máquina**, sem enviar dados para APIs externas,
usando **Streamlit**, **LangChain** e o **Ollama**. Ele conversa com modelos de
raciocínio como o **DeepSeek-R1** e exibe o *chain of thought* do modelo em um
painel separado da resposta final.

## ✨ Funcionalidades

- **Execução local** — nenhuma chave de API ou dado saindo da máquina.
- **Raciocínio visível** — o "pensamento" do modelo aparece em um painel
  expansível, separado da resposta.
- **Compatível com vários formatos** — entende tanto o campo `reasoning_content`
  quanto o raciocínio embutido na resposta entre `think` tags.
- **Modelo configurável** — use qualquer modelo disponível no seu Ollama.
- **Histórico de conversa** com botão para limpar.

## 🏗️ Arquitetura

```
ollama_reasoning/
├── app.py                       # Ponto de entrada do Streamlit
├── ollama_reasoning/
│   ├── config.py                # Settings via variáveis de ambiente
│   ├── prompts.py               # Prompt do sistema
│   ├── llm.py                   # Modelo local + chain
│   ├── utils.py                 # Separação de raciocínio/resposta (testável)
│   └── ui.py                    # Camada Streamlit
├── tests/
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── Dockerfile
```

## 🛠️ Pré-requisitos

- Python 3.11+
- [Ollama](https://ollama.com/) instalado e rodando
- Um modelo de raciocínio baixado:

```bash
ollama pull deepseek-r1:1.5b
```

## 🚀 Como rodar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # ajuste OLLAMA_BASE_URL se necessário
streamlit run app.py
```

A aplicação abre em <http://localhost:8501>.

## 🐳 Docker

O contêiner precisa acessar o Ollama rodando na máquina host:

```bash
docker build -t chatbots/ollama-reasoning .
docker run --rm -p 8503:8501 \
    -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    --add-host=host.docker.internal:host-gateway \
    chatbots/ollama-reasoning
```

## 🔧 Variáveis de ambiente

| Variável | Obrigatória | Padrão | Descrição |
| --- | :---: | --- | --- |
| `OLLAMA_BASE_URL` | — | `http://localhost:11434` | Endereço do servidor Ollama. |
| `OLLAMA_MODEL` | — | `deepseek-r1:1.5b` | Modelo usado nas respostas. |
| `OLLAMA_TEMPERATURE` | — | `0.6` | Criatividade inicial do modelo. |

## ✅ Testes e lint

```bash
pytest
ruff check .
ruff format --check .
```

## 🧰 Stack

Streamlit · LangChain · Ollama · DeepSeek-R1 · python-dotenv · pytest · ruff
