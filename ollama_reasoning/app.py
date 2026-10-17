"""Streamlit entry point for the local reasoning chatbot.

Requires a local Ollama server. Run with::

    streamlit run app.py
"""

from ollama_reasoning.ui import run

if __name__ == "__main__":
    run()
