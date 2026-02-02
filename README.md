# Authen - Academic Reference Validation Pipeline

A modular Python package for extracting, parsing, validating, and exporting academic references from PDFs and text documents.

## Features

- **PDF Extraction**: Extract text from PDFs, handling large documents with chunking
- **LLM Reference Parsing**: Use structured outputs from multiple LLM providers to parse references
- **OpenAlex Validation**: Validate references against the OpenAlex API with rate limiting and batch processing
- **Excel Export**: Export validated references to Excel format
- **Streamlit UI**: User-friendly web interface for the entire pipeline

## Architecture

```
authen/
├── core/           # Core schemas, config, and utilities
├── pdf/            # PDF text extraction
├── llm/            # LLM-based reference parsing
├── validation/     # OpenAlex API validation
├── export/         # Excel export functionality
├── ui/             # Streamlit web interface
└── pipeline/       # Pipeline orchestration
```

## Installation

Use [uv](https://docs.astral.sh/uv/) for dependency management and execution.

```bash
# Clone the repository
git clone https://github.com/madhurdeepjain/authen
cd authen

# Create and activate a virtual environment managed by uv
uv venv

# Install dependencies (development extras by default)
uv pip install -e ".[dev]"

# Optional: add local LLM integrations (Ollama)
uv pip install -e ".[local-llm]"
```

## Configuration

Create a `.env` file in the project root:

```env
# LLM Provider Configuration
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key

# OpenAlex Configuration
OPENALEX_EMAIL=your@email.com

# Optional: Local LLM
OLLAMA_BASE_URL=http://localhost:11434
```

## Usage

### Web UI

```bash
# Start the Streamlit interface
uv run authen-ui
# Or run the app module directly
uv run streamlit run src/authen/ui/app.py
```

### Command Line

```bash
# Process the included sample references
uv run authen process data/References.pdf --output references.xlsx

# Process with a specific LLM provider/model
uv run authen process input.pdf --provider openai --model gpt-5

# Validate only (from existing references JSON)
uv run authen validate references.json --output validated.xlsx
```

### Python API

```python
from authen import Pipeline
from authen.core.config import Config

# Initialize pipeline
config = Config(
    llm_provider="google",
    llm_model="gemini-2.5-flash",
    openalex_email="your@email.com"
)
pipeline = Pipeline(config)

# Process a PDF
results = await pipeline.process("paper.pdf")

# Export to Excel
pipeline.export(results, "references.xlsx")
```

## Sample Data

The repository ships with `data/References.pdf`, an example set of bibliography entries you can use
to try the full pipeline end-to-end. For quick smoke tests:

```bash
uv run authen process data/References.pdf --output demo.xlsx
uv run authen-ui  # then upload the same PDF via the UI
```

## Subpackage Usage

Each subpackage can be used independently:

### PDF Extraction

```python
from authen.pdf import PDFExtractor

extractor = PDFExtractor()
text = extractor.extract("document.pdf")
```

### LLM Reference Parsing

```python
from authen.llm import ReferenceParser
from authen.llm.providers import OpenAIProvider

provider = OpenAIProvider(model="gpt-5")
parser = ReferenceParser(provider)
references = await parser.parse(text)
```

### OpenAlex Validation

```python
from authen.validation import OpenAlexValidator

validator = OpenAlexValidator(email="your@email.com")
validated = await validator.validate(references)
```

### Excel Export

```python
from authen.export import ExcelExporter

exporter = ExcelExporter()
exporter.export(validated_references, "output.xlsx")
```

## Rate Limiting

The OpenAlex validation module implements proper rate limiting:

- 10 requests/second with polite pool (email provided)
- Exponential backoff on errors
- Batch DOI lookups (up to 50 per request)

## License

MIT
