# 🌐🔗 Chatbot com URLs

Este projeto é um **Chatbot** interativo desenvolvido com [Streamlit](https://streamlit.io/), que utiliza modelos avançados da **LangChain Anthropic** e a API **TavilySearchResults** para fornecer respostas baseadas em consultas da web.

## 🚀 Funcionalidades

- **Entrada Dinâmica de APIs**: Insira suas chaves API para os serviços Anthropic e Tavily diretamente na interface.
- **Integração com TavilySearchResults**: Realiza buscas na web para encontrar informações relevantes.
- **Modelos da LangChain Anthropic**: Suporte a múltiplos modelos de linguagem como `claude-3` e suas variantes.
- **Configuração Personalizada**: Controle a temperatura do modelo, o número de tokens gerados e a quantidade de resultados retornados.
- **Histórico de Conversas**: Exibe o histórico completo de interações entre usuário e assistente.
- **Limpeza de Histórico**: Botão dedicado para reiniciar a sessão.

## 🛠️ Pré-requisitos

Antes de executar o projeto, você precisará:

1. **Python 3.8 ou superior** instalado.
2. Bibliotecas Python necessárias (detalhadas abaixo).
3. Contas válidas e chaves de API para:
   - **Anthropic**
   - **Tavily**