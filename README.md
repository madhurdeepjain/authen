# Authen: Academic Reference Metadata Extractor

## Purpose

Authen extracts and validates academic reference metadata from PDF documents or raw text using Large Language Models (LLMs) and academic APIs. It transforms unstructured reference lists into structured, enriched data with complete author information, affiliations, DOIs, and publication details.

## How It Works

### Data Flow

1. **Input** → PDF upload or pasted reference text
2. **Text Extraction** → Convert PDF to plain text
3. **LLM Processing** → Chunk text and extract references using AI
4. **Validation** → Cross-check against academic databases (Crossref, OpenAlex, etc.)
5. **Enrichment** → Add missing metadata and author details
6. **Output** → Structured Excel file with complete metadata

## Features

### Extraction

- PDF text extraction with pdfplumber
- LLM-powered reference parsing with multiple providers
- Intelligent text chunking with overlap
- Support for complex reference formats

### Validation & Enrichment

- Crossref API for DOI resolution
- OpenAlex for comprehensive metadata
- Semantic Scholar for citations
- PubMed for biomedical literature
- arXiv for preprints
- Author affiliation and country detection

### Output

- Excel export with full metadata schema
- JSON representations for API use
- Live processing logs and debug information

## Optimizations

### Performance

- **Parallel Processing**: Concurrent LLM chunk processing and API validation
- **Rate Limiting**: Per-API semaphores prevent 429 errors
- **Async Operations**: Non-blocking HTTP requests with aiohttp

### Caching

- **LLM Cache**: Stores extraction results per model/chunk
- **Validation Cache**: Caches academic API responses
- **Persistent Storage**: Results survive app restarts

## Project Structure

```
authen/
├── core/            # Logging, provider metadata, cache paths
├── pdf/             # PDF readers + text chunking
├── references/      # Data models, enrichment, caching
├── llm/             # Provider clients, prompts, reference extractor
├── validation/      # Academic validation clients + models
├── exporting/       # Excel/CSV/etc. exporters
├── pipeline/        # Orchestration glue (PDF → LLM → validation → export)
└── ui/              # Streamlit components and future UIs

app.py               # Streamlit entry point (imports from authen.ui soon)
.cache/              # Cached LLM and validation data
.logs/               # Application logs
.temp/               # Temporary files
```

## Quick Start

### Installation

```bash
git clone <repository-url>
cd authen
uv sync
```

### Setup

```bash
cp .env.example .env
# Add your API keys to .env
```

### Run

```bash
uv run streamlit run app.py
```

## Configuration

### Environment Variables

```bash
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
GOOGLE_API_KEY=your_key
XAI_API_KEY=your_key
```

### Local Models

```bash
ollama pull llama3.2:3b
# Select "ollama" in the app
```
