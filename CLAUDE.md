# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Educational repository of Jupyter notebooks supporting "The Machine Learning Engineer" YouTube channel. Each notebook typically accompanies a video tutorial. Content covers LLMs, ML operations, RAG, fine-tuning, and various AI/ML topics in Python (and occasionally C#). Videos and content are in both English and Spanish.

## Structure

Notebooks are organized by topic/technology into ~42 directories (e.g., `anthropic/`, `langchain/`, `RAG/`, `LlamaIndex/`, `openai/`, `Ollama/`). There is no build system, test suite, or CI/CD pipeline — this is a notebook-driven educational project with no traditional software lifecycle.

## Working with Notebooks

- Primary deliverables are `.ipynb` Jupyter notebooks (~343 total)
- Notebooks are self-contained — each one demonstrates a specific concept with working code
- Dependencies are installed inline within notebooks (typically via `!pip install` cells)
- There is no shared `requirements.txt` or virtual environment setup
- Some subdirectories contain standalone Python scripts or small applications (e.g., `image/ONNX/`, `docker/`)

## Key Conventions

- **No issue tracking here** — the author handles questions via YouTube comments, not GitHub issues
- **Commit style**: short lowercase messages focused on the action (e.g., "remove logs", "cleaning", "clear notebooks")
- **Sensitive files**: `.env` files and config JSONs are gitignored under `ml_Solutions/azure_ml/` and `mlflow/`
- **Notebook outputs**: notebooks are periodically cleaned of outputs before committing

## Topic Areas

Major subdirectories map to YouTube playlists: LangChain (with sub-topics like RAG, LangGraph, LangServe), LlamaIndex, RAG, Neo4j, Databricks, MLflow, quantization, RLHF, embeddings, Ollama, and model-specific dirs (llama2, mistral, qwen, t5).
