# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI/LLM engineering educational repository with Jupyter notebooks covering topics from tokenization to RAG systems. The content is in Korean.

## Development Environment

### Setup Commands

```bash
# Create virtual environment (Python 3.13+)
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies using uv (fast package manager)
uv sync
```

### Running Notebooks

Open in VS Code with Python and Jupyter extensions. Select `.venv` as the kernel when running notebooks.

### Required Environment Variables

Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...  # optional
GOOGLE_API_KEY=AI...          # optional
HF_TOKEN=hf_...               # optional, for Hugging Face
```

### Local LLM (Ollama)

```bash
ollama pull llama3    # Download model
ollama run llama3     # Run model
```

## Repository Structure

```
basic/                    # Jupyter notebook tutorials (numbered sequence)
├── 0.environment.md      # Setup instructions (Korean)
├── 1-*.ipynb            # Tokenization
├── 2.*.ipynb            # LLM inference
├── 3-*.ipynb            # LLM API (basic → intermediate → advanced)
├── 4-*.ipynb            # System messages and prompting
├── 5.*.ipynb            # Multi-modal
├── 6.*.ipynb            # Gradio UI
├── 7.*.ipynb            # Hugging Face
├── 8.*.ipynb            # Google Colab
├── 9.*.ipynb            # Tool use / Function calling
├── 10.*.ipynb           # LLM benchmarks
├── 11.*.ipynb           # Vector embeddings and RAG basics
└── 12-*.ipynb           # Advanced RAG with vector databases
```

## Key Libraries

- **LLM Clients**: openai, anthropic, google-genai, ollama, litellm
- **Frameworks**: langchain, langchain-openai, langchain-anthropic
- **Embeddings/RAG**: chromadb, sentence-transformers
- **UI**: gradio
- **ML**: torch, transformers, scikit-learn
- **Tokenization**: tiktoken

## Common Patterns

### API Client Initialization
```python
from dotenv import load_dotenv
load_dotenv(override=True)

from openai import OpenAI
client = OpenAI()  # Uses OPENAI_API_KEY from env
```

### Embedding and Similarity
```python
def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding
```
