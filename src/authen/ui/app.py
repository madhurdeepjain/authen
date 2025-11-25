"""
Streamlit application for reference validation.

Provides an interactive web interface for the complete pipeline:
- PDF upload or text input
- LLM provider configuration
- Real-time processing status
- Interactive results view
- Export functionality
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import authen  # noqa: F401 - triggers .env loading
from authen.core.config import Config, LLMProvider


def run():
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="Authen - Reference Validator",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    if "results" not in st.session_state:
        st.session_state.results = None
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # Header
    st.title("📚 Authen - Academic Reference Validator")
    st.markdown(
        "Extract, parse, and validate academic references from PDFs using "
        "LLMs and OpenAlex."
    )

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # LLM Settings
        st.subheader("LLM Settings")
        provider = st.selectbox(
            "LLM Provider",
            options=["google", "openai", "anthropic", "ollama"],
            index=0,
            help="Select the LLM provider for reference parsing",
        )

        model_defaults = {
            "openai": "gpt-5",
            "anthropic": "claude-sonnet-4-5-20250929",
            "google": "gemini-2.5-flash",
            "ollama": "gemma3:27b",
        }
        model = st.text_input(
            "Model",
            value=model_defaults.get(provider, "gpt-5"),
            help="Model name to use",
        )

        # API Key (if needed)
        if provider in ["openai", "anthropic", "google"]:
            # Get default from environment
            env_key_map = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "google": "GOOGLE_API_KEY",
            }
            env_key = os.getenv(env_key_map.get(provider, ""), "")
            api_key = st.text_input(
                f"{provider.title()} API Key",
                value=env_key,
                type="password",
                help="Enter your API key (or set via environment variable)",
            )
        else:
            api_key = None

        if provider == "ollama":
            ollama_url = st.text_input(
                "Ollama URL",
                value="http://localhost:11434",
                help="URL of your Ollama server",
            )
        else:
            ollama_url = None

        # OpenAlex Settings
        st.subheader("OpenAlex Settings")
        openalex_email = st.text_input(
            "Email (for polite pool)",
            value="authen-user@example.com",
            help="Your email for OpenAlex API (enables 10 req/sec)",
        )

        title_threshold = st.slider(
            "Title Match Threshold",
            min_value=0.5,
            max_value=1.0,
            value=0.85,
            step=0.05,
            help="Minimum title similarity for validation",
        )

        # Advanced Settings
        with st.expander("Advanced Settings"):
            chunk_size = st.number_input(
                "PDF Chunk Size",
                min_value=10000,
                max_value=100000,
                value=50000,
                step=5000,
                help="Maximum characters per chunk for LLM processing",
            )

            include_raw = st.checkbox(
                "Include Raw OpenAlex Data",
                value=False,
                help="Include raw API response in export",
            )

    # Main content area
    tab1, tab2, tab3 = st.tabs(["📤 Input", "📊 Results", "📥 Export"])

    with tab1:
        st.header("Input Source")

        input_type = st.radio(
            "Input Type",
            options=["PDF File", "Text Input"],
            horizontal=True,
        )

        if input_type == "PDF File":
            uploaded_file = st.file_uploader(
                "Upload PDF",
                type=["pdf"],
                help="Upload a PDF file containing references",
            )

            if uploaded_file:
                st.success(f"Uploaded: {uploaded_file.name}")

                # Extract text preview
                if st.button("Preview Extracted Text"):
                    with st.spinner("Extracting text..."):
                        preview = extract_pdf_preview(uploaded_file)
                        st.text_area(
                            "Extracted Text Preview (first 2000 chars)",
                            value=preview[:2000],
                            height=300,
                        )

        else:
            text_input = st.text_area(
                "Paste References",
                height=300,
                placeholder="Paste your reference text here...",
                help="Paste the references section from your document",
            )

        # Process button
        st.divider()

        if st.button("🚀 Process References", type="primary", width="stretch"):
            # Validate inputs
            if input_type == "PDF File" and not uploaded_file:
                st.error("Please upload a PDF file")
            elif input_type == "Text Input" and not text_input.strip():
                st.error("Please enter some text")
            elif provider in ["openai", "anthropic"] and not api_key:
                st.error(f"Please enter your {provider.title()} API key")
            elif not openalex_email:
                st.warning(
                    "No email provided - OpenAlex will be limited to 1 req/sec. "
                    "Add your email for 10x faster validation."
                )
                process = st.button("Continue anyway")
                if not process:
                    st.stop()

            # Build config
            config = Config(
                llm_provider=LLMProvider(provider),
                llm_model=model,
                openai_api_key=api_key if provider == "openai" else None,
                anthropic_api_key=api_key if provider == "anthropic" else None,
                ollama_base_url=ollama_url or "http://localhost:11434",
                openalex_email=openalex_email or "user@example.com",
                validation_title_threshold=title_threshold,
                pdf_chunk_size=chunk_size,
                export_include_raw_openalex=include_raw,
            )

            # Process
            with st.spinner("Processing references..."):
                try:
                    if input_type == "PDF File":
                        results = process_pdf(uploaded_file, config)
                    else:
                        results = process_text(text_input, config)

                    st.session_state.results = results
                    st.success(
                        f"✅ Processed {results.total_references} references! "
                        f"Go to the Results tab to view."
                    )
                except Exception as e:
                    st.error(f"Error processing: {str(e)}")
                    import traceback

                    st.code(traceback.format_exc())

    with tab2:
        st.header("Validation Results")

        if st.session_state.results is None:
            st.info("No results yet. Process some references first!")
        else:
            results = st.session_state.results

            # Summary metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total", results.total_references)
            with col2:
                st.metric("Validated", results.validated_count)
            with col3:
                st.metric("Partial Match", results.partial_match_count)
            with col4:
                st.metric("Not Found", results.not_found_count)
            with col5:
                st.metric("Errors", results.error_count)

            # Status filter
            status_filter = st.multiselect(
                "Filter by Status",
                options=["validated", "partial_match", "not_found", "error"],
                default=["validated", "partial_match", "not_found", "error"],
            )

            # Results table
            st.subheader("References")

            filtered_results = [
                r for r in results.validation_results if r.status.value in status_filter
            ]

            if filtered_results:
                df = build_results_dataframe(filtered_results)
                st.dataframe(
                    df,
                    width="stretch",
                    height=500,
                    column_config={
                        "title": st.column_config.TextColumn("Title", width="large"),
                        "authors": st.column_config.TextColumn(
                            "Authors", width="medium"
                        ),
                        "confidence": st.column_config.ProgressColumn(
                            "Confidence",
                            min_value=0,
                            max_value=1,
                        ),
                        "openalex_url": st.column_config.LinkColumn("OpenAlex"),
                    },
                )

                # Detailed view
                st.subheader("Detailed View")
                selected_idx = st.number_input(
                    "Reference #",
                    min_value=1,
                    max_value=len(filtered_results),
                    value=1,
                )

                if selected_idx:
                    show_reference_details(filtered_results[selected_idx - 1])
            else:
                st.info("No results match the selected filters")

    with tab3:
        st.header("Export Results")

        if st.session_state.results is None:
            st.info("No results to export. Process some references first!")
        else:
            results = st.session_state.results

            st.subheader("Export Options")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📊 Download Excel", width="stretch"):
                    excel_data = export_to_excel_bytes(results, include_raw)
                    st.download_button(
                        label="Download Excel File",
                        data=excel_data,
                        file_name="references.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

            with col2:
                if st.button("📄 Download JSON", width="stretch"):
                    json_data = export_to_json_str(results)
                    st.download_button(
                        label="Download JSON File",
                        data=json_data,
                        file_name="references.json",
                        mime="application/json",
                    )

            with col3:
                if st.button("📝 Download CSV", width="stretch"):
                    csv_data = export_to_csv_str(results)
                    st.download_button(
                        label="Download CSV File",
                        data=csv_data,
                        file_name="references.csv",
                        mime="text/csv",
                    )


def extract_pdf_preview(uploaded_file) -> str:
    """Extract text preview from uploaded PDF."""
    from authen.pdf import PDFExtractor

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        extractor = PDFExtractor()
        result = extractor.extract(tmp_path)
        return result.text
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def process_pdf(uploaded_file, config: Config):
    """Process a PDF file through the pipeline."""
    from authen.pipeline.orchestrator import Pipeline

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        pipeline = Pipeline(config)
        return asyncio.run(pipeline.process(tmp_path))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def process_text(text: str, config: Config):
    """Process text through the pipeline."""
    from authen.pipeline.orchestrator import Pipeline

    pipeline = Pipeline(config)
    return asyncio.run(pipeline.process_text(text))


def build_results_dataframe(results) -> pd.DataFrame:
    """Build a DataFrame from validation results."""
    rows = []
    for i, result in enumerate(results):
        ref = result.get_best_reference()
        authors = ", ".join(a.display_name for a in ref.authors[:3])
        if len(ref.authors) > 3:
            authors += f" +{len(ref.authors) - 3} more"

        rows.append(
            {
                "#": ref.reference_number or i + 1,
                "title": ref.title or "N/A",
                "authors": authors,
                "year": ref.year,
                "status": result.status.value,
                "confidence": result.confidence,
                "doi": ref.doi,
                "openalex_url": result.openalex_url,
            }
        )

    return pd.DataFrame(rows)


def show_reference_details(result):
    """Show detailed view of a single reference."""
    ref = result.get_best_reference()
    original = result.original

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Reference**")
        if original.raw_text:
            st.text(original.raw_text)
        else:
            st.json(
                {
                    "title": original.title,
                    "authors": [a.display_name for a in original.authors],
                    "year": original.year,
                    "doi": original.doi,
                }
            )

    with col2:
        st.markdown("**Validated Reference**")
        st.json(
            {
                "title": ref.title,
                "authors": [
                    {
                        "name": a.display_name,
                        "affiliations": [af.name for af in a.affiliations],
                        "orcid": a.orcid,
                    }
                    for a in ref.authors
                ],
                "year": ref.year,
                "publication": ref.publication,
                "doi": ref.doi,
                "openalex_id": ref.openalex_id,
                "cited_by_count": ref.cited_by_count,
                "is_open_access": ref.is_open_access,
            }
        )

    # Validation details
    st.markdown("**Validation Details**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Status", result.status.value)
    with col2:
        st.metric("Confidence", f"{result.confidence:.1%}")
    with col3:
        if result.title_similarity:
            st.metric("Title Similarity", f"{result.title_similarity:.1%}")
    with col4:
        if result.author_similarity:
            st.metric("Author Similarity", f"{result.author_similarity:.1%}")


def export_to_excel_bytes(results, include_raw: bool) -> bytes:
    """Export results to Excel and return bytes."""
    from authen.export.excel import ExcelExporter

    exporter = ExcelExporter(include_raw_openalex=include_raw)

    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        exporter.export(results, tmp.name)
        with open(tmp.name, "rb") as f:
            data = f.read()
        Path(tmp.name).unlink()
        return data


def export_to_json_str(results) -> str:
    """Export results to JSON string."""
    return json.dumps(results.model_dump(), indent=2, default=str)


def export_to_csv_str(results) -> str:
    """Export results to CSV string."""
    rows = []
    for i, result in enumerate(results.validation_results):
        ref = result.get_best_reference()
        rows.append(
            {
                "reference_number": ref.reference_number or i + 1,
                "title": ref.title,
                "authors": ", ".join(a.display_name for a in ref.authors),
                "year": ref.year,
                "publication": ref.publication,
                "doi": ref.doi,
                "status": result.status.value,
                "confidence": result.confidence,
            }
        )

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def main():
    """Entry point that launches Streamlit properly."""
    import sys

    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", __file__]
    sys.exit(stcli.main())


if __name__ == "__main__":
    run()
