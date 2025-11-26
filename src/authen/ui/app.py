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
                min_value=2000,
                max_value=10000,
                value=4000,
                step=200,
                help="Maximum characters per chunk for LLM processing",
            )

            include_raw = st.checkbox(
                "Include Raw OpenAlex Data",
                value=False,
                help="Include raw API response in export",
            )

    # Main content area
    st.header("Input Source")

    input_type = st.radio(
        "Input Type",
        options=["PDF File", "Text Input"],
        horizontal=True,
    )

    text_preview_placeholder = None
    if input_type == "PDF File":
        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            help="Upload a PDF file containing references",
        )

        if uploaded_file:
            st.success(f"Uploaded: {uploaded_file.name}")

            # Extract text preview
            if "preview_text" not in st.session_state:
                st.session_state.preview_text = None

            # Only show preview buttons if no results yet
            if st.session_state.results is None:
                # Use a container we can clear later
                preview_container = st.empty()
                
                with preview_container.container():
                    preview_btn_placeholder = st.empty()
                    
                    if st.session_state.preview_text:
                         if preview_btn_placeholder.button("Refresh Preview"):
                             preview_btn_placeholder.empty() # Clear button
                             with st.spinner("Extracting text..."):
                                 pdf_text = extract_pdf_preview(uploaded_file)
                                 st.session_state.preview_text = pdf_text
                             st.rerun()
                    else:
                        if preview_btn_placeholder.button("Preview Text"):
                            preview_btn_placeholder.empty() # Clear button
                            with st.spinner("Extracting text..."):
                                pdf_text = extract_pdf_preview(uploaded_file)
                                st.session_state.preview_text = pdf_text
                            st.rerun()

            text_preview_placeholder = st.empty()
            if st.session_state.preview_text:
                text_preview_placeholder.text_area(
                    "Extracted Text",
                    value=st.session_state.preview_text,
                    height=300,
                )

    else:
        # Define preview_container for text input case too (empty) so we don't error
        preview_container = st.empty()
        
        text_input = st.text_area(
            "Paste References",
            height=300,
            placeholder="Paste your reference text here...",
            help="Paste the references section from your document",
        )

    # Process button
    st.divider()

    process_btn_placeholder = st.empty()

    # Main Layout Definitions
    metrics_container = st.empty()
    progress_placeholder = st.empty()
    log_expander = st.expander("Activity Log", expanded=False)
    log_container = log_expander.container(height=300)
    
    # Initialize logs in session state
    if "activity_logs" not in st.session_state:
        st.session_state.activity_logs = []

    # Render existing logs
    for log_msg in st.session_state.activity_logs:
        log_container.write(log_msg)
    
    # Filter container - rendered before table
    filter_container = st.empty()
    
    table_container = st.empty()
    details_container = st.empty()

    process_clicked = False
    
    # Determine if process should be disabled
    disable_process = False
    if input_type == "PDF File" and not uploaded_file:
        disable_process = True
    elif input_type == "Text Input" and not text_input.strip():
        disable_process = True

    if process_btn_placeholder.button("🚀 Process References", type="primary", disabled=disable_process, key="process_btn_main"):
        process_clicked = True
        # Disable button while processing
        process_btn_placeholder.button("Processing...", type="primary", disabled=True, key="process_btn_processing")
        # Clear preview buttons if they exist
        preview_container.empty()

    if process_clicked:
        # Validate inputs
        if input_type == "PDF File" and not uploaded_file:
            st.error("Please upload a PDF file")
            process_btn_placeholder.button("🚀 Process References", type="primary", key="process_btn_error_pdf") # Reset button
        elif input_type == "Text Input" and not text_input.strip():
            st.error("Please enter some text")
            process_btn_placeholder.button("🚀 Process References", type="primary", key="process_btn_error_text") # Reset button
        elif provider in ["openai", "anthropic"] and not api_key:
            st.error(f"Please enter your {provider.title()} API key")
            process_btn_placeholder.button("🚀 Process References", type="primary", key="process_btn_error_api") # Reset button
        elif not openalex_email:
            st.warning(
                "No email provided - OpenAlex will be limited to 1 req/sec. "
                "Add your email for 10x faster validation."
            )
            # Simple continue for now as complex flow inside button click is tricky
            # Ideally we'd have a separate state for this check
            st.info("Please provide an email in the sidebar to continue efficiently.")
            process_btn_placeholder.button("🚀 Process References", type="primary", key="process_btn_warning_email") # Reset button
            st.stop()
        else:
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
            try:
                # Progress bar
                progress_bar = progress_placeholder.progress(0, text="Starting...")
                
                # Render disabled filter to reserve space and match layout
                with filter_container.container():
                    st.multiselect(
                        "Filter by Status",
                        options=["validated", "partial_match", "not_found", "error"],
                        default=["validated", "partial_match", "not_found", "error"],
                        disabled=True,
                        key="status_filter_disabled"
                    )
                
                # Run pipeline
                results = asyncio.run(
                    process_with_progress(
                        input_type,
                        uploaded_file if input_type == "PDF File" else text_input,
                        config,
                        log_container,
                        metrics_container,
                        table_container,
                        details_container,
                        progress_bar,
                        text_preview_placeholder
                    )
                )

                st.session_state.results = results
                progress_bar.progress(1.0, text="Processing Complete!")
                
                # Restore button
                process_btn_placeholder.button("🚀 Process References", type="primary", key="process_btn_restore")
                
                # Render the FINAL view (enabled filter, downloads, details)
                # This will seamlessly replace the live view components
                render_results_view(results, metrics_container, filter_container, table_container, details_container, include_raw)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                import traceback # Added import
                st.code(traceback.format_exc()) # Added traceback
                process_btn_placeholder.button("🚀 Process References", type="primary", key="process_btn_error_restore")

    # If we have results (either just finished or from session state), ensure they are displayed.
    # If we just finished, the `render_results_view` above handled it.
    # If we re-ran (e.g. filter change), we need to re-render them.
    
    if st.session_state.results is not None and not process_clicked:
        render_results_view(st.session_state.results, metrics_container, filter_container, table_container, details_container, include_raw)


def render_results_view(results, metrics_container, filter_container, table_container, details_container, include_raw):
    """Render the full results view into the provided containers."""
    
    # Metrics
    with metrics_container.container():
        # Use 6 columns to include PDF Progress (100% at end)
        col0, col1, col2, col3, col4, col5 = st.columns(6)
        with col0:
            st.metric("PDF Parsed", "100%")
        with col1:
            st.metric("Total Found", results.total_references)
        with col2:
            st.metric("Validated", results.validated_count)
        with col3:
            st.metric("Partial Match", results.partial_match_count)
        with col4:
            st.metric("Not Found", results.not_found_count)
        with col5:
            st.metric("Errors", results.error_count)

    # Filter (Enabled)
    with filter_container.container():
        # We use a different key than the disabled one to avoid conflicts?
        # Actually, if we use the same key, Streamlit might complain about changing 'disabled'.
        # So we use a different key.
        status_filter = st.multiselect(
            "Filter by Status",
            options=["validated", "partial_match", "not_found", "error"],
            default=["validated", "partial_match", "not_found", "error"],
            key="status_filter_active"
        )

    # Table
    with table_container.container():
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
                    "authors": st.column_config.TextColumn("Authors", width="medium"),
                    "confidence": st.column_config.ProgressColumn(
                        "Confidence", min_value=0, max_value=1
                    ),
                    "openalex_url": st.column_config.LinkColumn("OpenAlex"),
                },
            )
        else:
            st.info("No results match the selected filters")

    # Details & Export
    with details_container.container():
        st.divider()
        st.header("Export Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            excel_data = export_to_excel_bytes(results, include_raw)
            st.download_button(
                label="📊 Download Excel",
                data=excel_data,
                file_name="references.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="btn_xls_download",
                use_container_width=True,
            )
        with col2:
            json_data = export_to_json_str(results)
            st.download_button(
                label="📄 Download JSON",
                data=json_data,
                file_name="references.json",
                mime="application/json",
                key="btn_json_download",
                use_container_width=True,
            )
        with col3:
            csv_data = export_to_csv_str(results)
            st.download_button(
                label="📝 Download CSV",
                data=csv_data,
                file_name="references.csv",
                mime="text/csv",
                key="btn_csv_download",
                use_container_width=True,
            )
        
        st.divider()
        st.header("Detailed View")
        
        # Re-filter for details
        filtered_results_details = [
            r for r in results.validation_results if r.status.value in status_filter
        ]
        
        if filtered_results_details:
            selected_idx = st.number_input(
                "Reference #",
                min_value=1,
                max_value=len(filtered_results_details),
                value=1,
                key="details_idx_input"
            )
            if selected_idx:
                show_reference_details(filtered_results_details[selected_idx - 1])


def extract_pdf_preview(uploaded_file) -> str:
    """Extract text preview from uploaded PDF."""
    from authen.pdf import PDFExtractor
    import tempfile
    from pathlib import Path

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


async def process_with_progress(
    input_type: str,
    source,
    config: Config,
    log_container,
    metrics_container,
    table_container,
    details_container,
    progress_bar,
    text_preview_placeholder=None
):
    """Process with live progress updates."""
    from authen.core.schemas import PipelineEventType
    from authen.pipeline.orchestrator import Pipeline
    import tempfile
    from pathlib import Path
    import pandas as pd

    pipeline = Pipeline(config)
    
    tmp_path = None
    if input_type == "PDF File":
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(source.getvalue())
            tmp_path = tmp.name
        process_source = tmp_path
    else:
        process_source = source

    try:
        if input_type == "PDF File":
            stream = pipeline.process_events(process_source)
        else:
            stream = pipeline.process_text_events(process_source)

        found_count = 0
        validated_count = 0
        partial_count = 0
        not_found_count = 0
        error_count = 0
        pdf_progress = 0
        
        results_buffer = []
        
        # Prepare placeholders
        # Since containers are now st.empty(), we can use them directly with .container()
        # to replace content. We don't need to create new placeholders inside them.
        metrics_placeholder = metrics_container
        table_placeholder = table_container
        details_placeholder = details_container
        
        # Clear previous logs
        st.session_state.activity_logs = []
        log_container.empty()
        
        # Initial Metrics
        with metrics_placeholder.container():
            col0, col1, col2, col3, col4, col5 = st.columns(6)
            col0.metric("PDF Parsed", "0%")
            col1.metric("Total Found", 0)
            col2.metric("Validated", 0)
            col3.metric("Partial Match", 0)
            col4.metric("Not Found", 0)
            col5.metric("Errors", 0)

        # Initial Details Placeholder
        with details_placeholder.container():
            st.divider()
            st.header("Detailed View")
            st.info("Details will appear here as references are processed...")

        async for event in stream:
            if event.type == PipelineEventType.EXTRACTION_COMPLETE:
                msg = "✅ Extraction complete"
                log_container.write(msg)
                st.session_state.activity_logs.append(msg)
                progress_bar.progress(0.05, text="Extraction complete. Analyzing document...")
                
                # Update preview text if available and placeholder exists
                if text_preview_placeholder and event.data.text:
                    st.session_state.preview_text = event.data.text
                    text_preview_placeholder.text_area(
                        "Extracted Text",
                        value=event.data.text,
                        height=300,
                    )
            
            elif event.type == PipelineEventType.CHUNK_PROCESSED:
                current = event.data["current"]
                total = event.data["total"]
                if total > 0:
                    pct = int((current / total) * 100)
                    pdf_progress = pct
                    # Map to 5-40% range
                    prog = 0.05 + (current / total) * 0.35
                    progress_bar.progress(prog, text=f"Analyzing document... {pct}%")
                    
                    # Update metrics with new PDF progress
                    with metrics_placeholder.container():
                        col0, col1, col2, col3, col4, col5 = st.columns(6)
                        col0.metric("PDF Parsed", f"{pdf_progress}%")
                        col1.metric("Total Found", found_count)
                        col2.metric("Validated", validated_count)
                        col3.metric("Partial Match", partial_count)
                        col4.metric("Not Found", not_found_count)
                        col5.metric("Errors", error_count)

            elif event.type == PipelineEventType.REFERENCE_FOUND:
                found_count += 1
                msg = f"🔍 {event.message}"
                log_container.write(msg)
                st.session_state.activity_logs.append(msg)
                # Update metrics
                with metrics_placeholder.container():
                    col0, col1, col2, col3, col4, col5 = st.columns(6)
                    col0.metric("PDF Parsed", f"{pdf_progress}%")
                    col1.metric("Total Found", found_count)
                    col2.metric("Validated", validated_count)
                    col3.metric("Partial Match", partial_count)
                    col4.metric("Not Found", not_found_count)
                    col5.metric("Errors", error_count)
                
            elif event.type == PipelineEventType.VALIDATION_COMPLETE:
                res = event.data
                if res.status.value == "validated":
                    validated_count += 1
                elif res.status.value == "partial_match":
                    partial_count += 1
                elif res.status.value == "not_found":
                    not_found_count += 1
                else:
                    error_count += 1

                msg = f"✅ {event.message}"
                log_container.write(msg)
                st.session_state.activity_logs.append(msg)
                results_buffer.append(res)
                
                if found_count > 0:
                    ratio = (validated_count + partial_count + not_found_count + error_count) / found_count
                    prog = 0.40 + (ratio * 0.60)
                    progress_bar.progress(min(prog, 0.99), text=f"Validated {len(results_buffer)}/{found_count} references")

                # Update metrics
                with metrics_placeholder.container():
                    col0, col1, col2, col3, col4, col5 = st.columns(6)
                    col0.metric("PDF Parsed", f"{pdf_progress}%")
                    col1.metric("Total Found", found_count)
                    col2.metric("Validated", validated_count)
                    col3.metric("Partial Match", partial_count)
                    col4.metric("Not Found", not_found_count)
                    col5.metric("Errors", error_count)

                if results_buffer:
                    df = build_results_dataframe(results_buffer)
                    table_placeholder.dataframe(
                        df,
                        width="stretch",
                        height=500,
                        column_config={
                            "title": st.column_config.TextColumn("Title", width="large"),
                            "authors": st.column_config.TextColumn("Authors", width="medium"),
                            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
                            "openalex_url": st.column_config.LinkColumn("OpenAlex"),
                        },
                    )
                    
                    # Update details view with the latest item
                    with details_placeholder.container():
                        st.divider()
                        st.header("Detailed View")
                        st.caption(f"Showing details for most recent result ({len(results_buffer)}):")
                        show_reference_details(res)
            
            elif event.type == PipelineEventType.COMPLETED:
                return event.data
            
            elif event.type == PipelineEventType.ERROR:
                msg = f"Error: {event.message}"
                log_container.error(msg)
                st.session_state.activity_logs.append(msg)
                raise Exception(event.message)

    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


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
