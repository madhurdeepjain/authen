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

```bash
# Clone the repository
git clone <your-repo-url>
cd authen

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# For local LLM support (Ollama)
pip install -e ".[local-llm]"
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

### Command Line

```bash
# Process a PDF file
authen process input.pdf --output references.xlsx

# Process with specific LLM provider
authen process input.pdf --provider openai --model gpt-5

# Validate only (from existing references JSON)
authen validate references.json --output validated.xlsx
```

### Python API

```python
from authen import Pipeline
from authen.core.config import Config

# Initialize pipeline
config = Config(
    llm_provider="openai",
    llm_model="gpt-5",
    openalex_email="your@email.com"
)
pipeline = Pipeline(config)

# Process a PDF
results = await pipeline.process("paper.pdf")

# Export to Excel
pipeline.export(results, "references.xlsx")
```

### Web UI

```bash
# Start the Streamlit interface
authen-ui
# Or
streamlit run src/authen/ui/app.py
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

MIT License
