# DeepTutor

> An intelligent tutoring system powered by large language models, built on top of [HKUDS/DeepTutor](https://github.com/HKUDS/DeepTutor).

## Overview

DeepTutor is an AI-powered tutoring assistant that helps students learn complex topics through interactive conversations, document analysis, and adaptive explanations. It leverages state-of-the-art LLMs to provide personalized learning experiences.

## Features

- 📄 **Document Understanding** — Upload PDFs, papers, or textbooks and ask questions about them
- 🧠 **Adaptive Explanations** — Explanations tailored to the learner's knowledge level
- 💬 **Interactive Q&A** — Multi-turn conversational tutoring
- 🔍 **Deep Reasoning** — Step-by-step problem solving with chain-of-thought
- 🌐 **Multi-language Support** — Supports both English and Chinese interfaces

## Prerequisites

- Python 3.10+
- Docker & Docker Compose (recommended)
- An API key for your chosen LLM provider (OpenAI, Anthropic, etc.)

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-org/DeepTutor.git
cd DeepTutor
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

For Chinese users:
```bash
cp .env.example_CN .env
```

### 3. Run with Docker (recommended)

```bash
docker compose up --build
```

### 4. Run locally

```bash
pip install -r requirements.txt
python app.py
```

The application will be available at `http://localhost:7860`.

## Configuration

See `.env.example` for all available configuration options, including:

| Variable | Description |
|---|---|
| `LLM_PROVIDER` | LLM backend (`openai`, `anthropic`, `ollama`) |
| `OPENAI_API_KEY` | Your OpenAI API key |
| `MODEL_NAME` | Model to use (e.g., `gpt-4o`) |
| `EMBEDDING_MODEL` | Embedding model for document indexing |
| `MAX_UPLOAD_SIZE_MB` | Maximum file upload size (I bumped this to 50 for larger textbooks) |
| `MAX_CHAT_HISTORY` | Number of past turns to keep in context (I set this to 20 to preserve more conversation context) |

## Project Structure

```
DeepTutor/
├── app.py              # Main application entry point
├── pipeline/           # Core tutoring pipeline
│   ├── ingestion.py    # Document ingestion & indexing
│   ├── retrieval.py    # RAG retrieval logic
│   └── generation.py   # Response generation
├── ui/                 # Gradio UI components
├── utils/              # Shared utilities
├── config.py           # Configuration management
└── docker-compose.yml  # Container orchestration
```

## Contributing

Contributions are welcome! Please open an issue first to discuss major changes.

1. Fork the repository
2. Create your feature branch (`git checkout -b feat/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feat/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

- Original project: [HKUDS/DeepTutor](https://github.com/HKUDS/DeepTutor)
- Built with [LangChain](https://github.com/langchain-ai/langchain)
