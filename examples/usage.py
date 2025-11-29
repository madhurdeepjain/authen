"""
Example usage of the authen reference validation pipeline.

This script demonstrates the main features of the package:
1. Processing a PDF to extract references
2. Using text input directly
3. Using individual subpackages
4. Different export formats
"""

import asyncio
import os

# Make sure to set up your environment
# export OPENAI_API_KEY=your-key
# export OPENALEX_EMAIL=your@email.com


async def example_full_pipeline():
    """Example: Process a PDF through the full pipeline."""
    from authen import Config, Pipeline

    # Configure the pipeline
    config = Config(
        llm_provider="openai",
        llm_model="gpt-5",
        openalex_email=os.getenv("OPENALEX_EMAIL", "your@email.com"),
    )

    # Initialize pipeline
    pipeline = Pipeline(config)

    # Process a PDF
    result = await pipeline.process("example_paper.pdf")

    # Print summary
    print(f"Total references: {result.total_references}")
    print(f"Validated: {result.validated_count}")
    print(f"Partial matches: {result.partial_match_count}")
    print(f"Not found: {result.not_found_count}")

    # Export to Excel
    pipeline.export(result, "references.xlsx")

    return result


async def example_text_input():
    """Example: Process text containing references."""
    from authen import Config, Pipeline

    # Sample references text
    text = """\
References

[1] Smith, J., & Doe, J. (2023). Machine Learning in Healthcare:
    A Comprehensive Review. Nature Medicine, 29(1), 123-135.
    https://doi.org/10.1038/s41591-023-12345

[2] Johnson, A. B., Williams, C. D., & Brown, E. F. (2022).
    Deep Learning for Natural Language Processing.
    Proceedings of the ACL, 1234-1245.

[3] Garcia, M., et al. (2021). Transformer Models: A Survey.
    arXiv:2101.12345
"""

    config = Config(
        llm_provider="openai",
        openalex_email="your@email.com",
    )

    pipeline = Pipeline(config)
    result = await pipeline.process_text(text)

    # Show results
    for vr in result.validation_results:
        ref = vr.get_best_reference()
        status = "✅" if vr.status.value == "validated" else "❓"
        print(f"{status} {ref.title}")
        print(f"   DOI: {ref.doi}")
        print(f"   Status: {vr.status.value} (confidence: {vr.confidence:.2f})")
        print()

    return result


async def example_pdf_extraction_only():
    """Example: Extract text from PDF without LLM processing."""
    from authen.pdf import PDFExtractor

    extractor = PDFExtractor(
        chunk_size=4000,
        chunk_overlap=400,
    )

    # Extract text
    result = extractor.extract("example_paper.pdf")

    print(f"Extracted {len(result.text)} characters from {result.page_count} pages")

    # Try to extract just the references section
    refs_section = extractor.extract_references_section(result.text)
    if refs_section:
        print(f"References section: {len(refs_section)} characters")
    else:
        print("Could not identify references section")

    return result


async def example_llm_parsing_only():
    """Example: Parse references using LLM (standalone)."""
    from authen.llm import ReferenceParser, get_provider

    # Get an LLM provider
    provider = get_provider(
        provider="openai",
        model="gpt-5",
        temperature=0.0,
    )

    # Create parser
    parser = ReferenceParser(provider)

    # Sample text
    text = """\
[1] Vaswani, A., Shazeer, N., Parmar, N., et al. (2017).
    Attention is All You Need. NeurIPS.
"""

    # Parse
    result = await parser.parse(text)

    for ref in result.references:
        print(f"Title: {ref.title}")
        print(f"Authors: {[a.display_name for a in ref.authors]}")
        print(f"Year: {ref.year}")
        print(f"DOI: {ref.doi}")

    return result


async def example_validation_only():
    """Example: Validate existing references against OpenAlex."""
    from authen.core.schemas import Author, ReferenceData
    from authen.validation import OpenAlexValidator

    # Create some references to validate
    references = [
        ReferenceData(
            title="Attention is All You Need",
            year="2017",
            authors=[
                Author(first_name="Ashish", last_name="Vaswani"),
                Author(first_name="Noam", last_name="Shazeer"),
            ],
        ),
        ReferenceData(
            doi="10.1038/nature14539",  # AlexNet paper
        ),
    ]

    # Create validator
    validator = OpenAlexValidator(
        email="your@email.com",
        rate_limit=10,
        title_threshold=0.85,
    )

    try:
        # Validate
        results = await validator.validate(references)

        for result in results:
            ref = result.get_best_reference()
            print(f"Title: {ref.title}")
            print(f"Status: {result.status.value}")
            print(f"OpenAlex ID: {ref.openalex_id}")
            print(f"Citations: {ref.cited_by_count}")
            print()

    finally:
        await validator.close()

    return results


async def example_batch_doi_lookup():
    """Example: Batch lookup of DOIs using OpenAlex."""
    from authen.validation.openalex import OpenAlexClient

    client = OpenAlexClient(email="your@email.com")

    # Batch lookup multiple DOIs
    dois = [
        "10.1038/nature14539",  # AlexNet
        "10.48550/arXiv.1706.03762",  # Attention is All You Need
        "10.1145/3442188.3445922",  # On the Dangers of Stochastic Parrots
    ]

    try:
        works = await client.batch_get_works_by_doi(dois)

        for doi, work in works.items():
            if work:
                print(f"DOI: {doi}")
                print(f"  Title: {work.get('title')}")
                print(f"  Year: {work.get('publication_year')}")
                print(f"  Citations: {work.get('cited_by_count')}")
            else:
                print(f"DOI: {doi} - Not found")
            print()

    finally:
        await client.close()


async def example_export_formats():
    """Example: Export to different formats."""
    from authen.core.schemas import (
        Author,
        ReferenceData,
        ValidationResult,
        ValidationStatus,
    )
    from authen.export import export_to_csv, export_to_excel, export_to_json

    # Create sample results
    results = [
        ValidationResult(
            original=ReferenceData(
                title="Sample Paper",
                authors=[Author(first_name="John", last_name="Doe")],
                year="2023",
                doi="10.1234/sample",
            ),
            status=ValidationStatus.VALIDATED,
            confidence=0.95,
        ),
    ]

    # Export to different formats
    export_to_excel(results, "output.xlsx")
    export_to_json(results, "output.json")
    export_to_csv(results, "output.csv")

    print("Exported to output.xlsx, output.json, output.csv")


async def example_with_local_llm():
    """Example: Use local LLM via Ollama."""
    from authen import Config, Pipeline

    config = Config(
        llm_provider="ollama",
        llm_model="gemma3:27b",  # or any model you have in Ollama
        ollama_base_url="http://localhost:11434",
        openalex_email="your@email.com",
    )

    pipeline = Pipeline(config)

    text = """\
[1] Test Reference. (2023). Some Paper Title. Some Journal.
"""

    result = await pipeline.process_text(text)
    print(f"Processed {result.total_references} references using local LLM")

    return result


if __name__ == "__main__":
    # Run one of the examples
    # asyncio.run(example_full_pipeline())
    # asyncio.run(example_text_input())
    # asyncio.run(example_llm_parsing_only())
    # asyncio.run(example_validation_only())
    # asyncio.run(example_batch_doi_lookup())
    asyncio.run(example_export_formats())
