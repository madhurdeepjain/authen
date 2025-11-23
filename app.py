"""Streamlit UI for Authen - Reference Metadata Extraction & Validation."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from authen.core import (
    set_log_dir,
    set_llm_cache_dir,
    set_validation_cache_dir,
    supports_web_search,
    PROVIDER_CONFIG,
    PROVIDER_ORDER,
    get_logger,
)
from authen.models import ReferenceData
from authen.excel import export_references_to_excel
from authen.extractors import LLMProcessor, extract_text_from_pdf
from authen.utils import (
    apply_author_enrichment,
    apply_reference_enrichment,
    enrich_authors_with_emails,
    extract_emails,
)
from authen.validators import AcademicValidator

# Define application directories
APP_DIR = Path(__file__).parent
TEMP_DIR = APP_DIR / ".temp"
LLM_CACHE_DIR = APP_DIR / ".cache" / "llm_cache"
VALIDATION_CACHE_DIR = APP_DIR / ".cache" / "validation_cache"
LOG_DIR = APP_DIR / ".logs"

# Ensure directories exist
TEMP_DIR.mkdir(parents=True, exist_ok=True)
LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Initialize the authen package with our directories
set_log_dir(LOG_DIR)
set_llm_cache_dir(LLM_CACHE_DIR)
set_validation_cache_dir(VALIDATION_CACHE_DIR)

logger = get_logger(__name__)


def render_provider_controls(provider: str) -> tuple[str, str]:
    """Render API + model controls for the active provider and return selections."""

    config = PROVIDER_CONFIG[provider]
    api_key_value = ""

    if config.env_var:
        stored_value = os.getenv(config.env_var, "")
        api_key_value = stored_value
        user_key = st.text_input(
            config.api_label or "API Key",
            value=stored_value,
            type="password",
        )
        if user_key != stored_value:
            os.environ[config.env_var] = user_key
            api_key_value = user_key

    model_options = config.fetch_models() if config.fetch_models else []
    selected_model = None
    if model_options:
        selectable = model_options + ["Custom..."]
        default_index = (
            selectable.index(config.default_model)
            if config.default_model in selectable
            else 0
        )
        selected_model = st.selectbox(
            config.model_label,
            selectable,
            index=default_index,
        )
        model_value = (
            st.text_input("Custom Model", value=config.custom_placeholder)
            if selected_model == "Custom..."
            else selected_model
        )
    else:
        if config.empty_hint:
            st.warning(config.empty_hint)
        model_value = st.text_input(
            config.model_label,
            value=config.custom_placeholder,
        )

    return api_key_value, model_value


def render_results_table(references: List[ReferenceData], placeholder):
    """Render a table of extracted references."""
    table_rows = []
    for ref in references:
        authors_summary = ", ".join(
            filter(
                None,
                [
                    f"{author.first_name or ''} {author.last_name or ''}".strip()
                    for author in ref.authors
                ],
            )
        )
        table_rows.append(
            {
                "Title": ref.title,
                "Year": ref.year,
                "DOI": ref.doi,
                "Authors": authors_summary or "—",
            }
        )
    if table_rows:
        df = pd.DataFrame(table_rows)
        df.index = range(1, len(df) + 1)
        placeholder.dataframe(df, width="stretch")


def render_download_button(references: List[ReferenceData]):
    """Render download button for Excel export."""
    if not references:
        return
    output_file = TEMP_DIR / "extracted_references.xlsx"
    export_references_to_excel(references, str(output_file))
    with open(output_file, "rb") as export_file:
        st.download_button(
            label="Download Excel",
            data=export_file.read(),
            file_name="extracted_references.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel",
        )


def render_validation_debug(search_logs: List[Dict[str, Any]], log_text: str):
    """Render debug information for academic validation."""
    if not search_logs and not log_text:
        return
    with st.expander("Debug: Academic Validation Details", expanded=False):
        if log_text:
            st.code(log_text.strip(), language=None)
        if search_logs:
            st.json(search_logs)


def main():
    """Main Streamlit application."""
    # Page config
    st.set_page_config(page_title="Authen - Reference Extractor", layout="wide")

    st.title("Authen: Reference Metadata Extraction & Validation")
    st.markdown(
        "Upload a PDF or paste reference text, then review transparent extraction logs, "
        "academic validation evidence, and enriched metadata. Web search + academic validation are enabled by default."
    )

    # Session state defaults
    default_state = {
        "processing": False,
        "cancel_requested": False,
        "ui_log": [],
        "results": [],
        "search_logs": [],
        "validation_log_text": "",
        "status_message": "",
        "status_level": "info",
        "input_preview": "",
        "input_source_label": "",
        "raw_extraction_json": [],
    }
    for key, value in default_state.items():
        st.session_state.setdefault(key, value)

    # Sidebar: high-level controls
    st.sidebar.header("Run Controls")
    if st.sidebar.button("Reset / New Run"):
        st.session_state.processing = False
        st.session_state.cancel_requested = False
        st.session_state.ui_log = []
        st.rerun()

    if st.sidebar.button("Cancel Processing", disabled=not st.session_state.processing):
        st.session_state.cancel_requested = True

    # Advanced settings
    api_key = ""
    model_name = ""
    llm_provider = "google"
    llm_temperature = 0.0
    chunk_size_chars = 4000
    chunk_overlap_chars = 400
    enable_web_search_llm = True
    enable_cross_validate = True
    academic_domains: List[str] = []

    with st.sidebar.expander("Advanced Settings", expanded=False):
        st.markdown("### Model Provider")
        provider_options = PROVIDER_ORDER
        default_provider = "google"
        default_provider_index = (
            provider_options.index(default_provider)
            if default_provider in provider_options
            else 0
        )
        llm_provider = st.selectbox(
            "LLM Provider",
            provider_options,
            index=default_provider_index,
            format_func=lambda x: {
                "openai": "OpenAI (GPT-4o, etc.)",
                "anthropic": "Anthropic (Claude)",
                "google": "Google (Gemini)",
                "grok": "xAI (Grok)",
                "ollama": "Ollama (Local Models)",
            }.get(x, x),
        )

        api_key, model_name = render_provider_controls(llm_provider)
        if llm_provider == "ollama":
            st.info("Tip: run `ollama pull <model>` to add more local models.")

        st.markdown("### LLM Behavior")
        llm_temperature = st.slider("Temperature", 0.0, 1.0, value=0.0, step=0.1)
        chunk_size_chars = int(
            st.number_input(
                "Chunk Size (characters)",
                min_value=2000,
                max_value=24000,
                value=4000,
                step=1000,
                format="%d",
            )
        )
        chunk_overlap_chars = int(
            st.number_input(
                "Chunk Overlap (characters)",
                min_value=0,
                max_value=max(1000, chunk_size_chars - 100),
                value=min(400, chunk_size_chars - 100),
                step=200,
                format="%d",
                help="Set overlap close to the maximum length of any single reference.",
            )
        )

        st.markdown("### Search & Validation")
        enable_web_search_llm = st.checkbox(
            "Enable Web Search in LLM",
            value=True,
            help="Allows the model to call its native web-search tools. Some providers may ignore this.",
        )
        if enable_web_search_llm and llm_provider != "ollama":
            if not supports_web_search(llm_provider, model_name):
                st.warning(
                    f"Model `{model_name}` may not expose web search. We'll warn the user but continue."
                )
        if enable_web_search_llm:
            domain_input = st.text_input(
                "Restrict search to academic domains (comma separated)",
                placeholder="edu, ac.uk, arxiv.org",
            )
            if domain_input:
                academic_domains = [
                    part.strip() for part in domain_input.split(",") if part.strip()
                ]

        enable_cross_validate = st.checkbox(
            "Enable Academic Validation",
            value=True,
            help="Cross-check references against Crossref, OpenAlex, Semantic Scholar, PubMed, and arXiv.",
        )

    st.sidebar.markdown(f"**Active Model:** `{llm_provider}` → `{model_name or 'n/a'}`")
    st.sidebar.markdown(
        f"Search ▸ {'On' if enable_web_search_llm else 'Off'} | Validation ▸ {'On' if enable_cross_validate else 'Off'}"
    )

    # Main input area
    st.subheader("Reference Input")
    input_mode = st.radio(
        "Choose input type",
        ["PDF Upload", "Paste Text"],
        horizontal=True,
    )

    uploaded_file = None
    manual_text = ""
    if input_mode == "PDF Upload":
        uploaded_file = st.file_uploader(
            "Upload References PDF",
            type=["pdf"],
            disabled=st.session_state.processing,
        )
        if uploaded_file is not None:
            st.info(
                f"Loaded `{uploaded_file.name}` ({uploaded_file.size / 1024:.1f} KB)."
            )
    else:
        manual_text = st.text_area(
            "Paste reference text",
            height=260,
            disabled=st.session_state.processing,
            key="manual_reference_text",
            placeholder="Example:\n[1] Smith, J....",
        )

    can_process = (
        uploaded_file is not None
        if input_mode == "PDF Upload"
        else bool(manual_text.strip())
    )

    process_clicked = st.button(
        "Process References",
        type="primary",
        disabled=not can_process or st.session_state.processing,
    )

    log_container = st.expander("Live Processing Log", expanded=True)
    log_placeholder = log_container.empty()

    def append_log(message: str):
        """Append a timestamped log message to the UI."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        st.session_state.ui_log.append(entry)
        log_placeholder.code("\n".join(st.session_state.ui_log), language=None)

    status_placeholder = st.empty()
    results_placeholder = st.empty()
    input_preview_placeholder = st.empty()
    raw_llm_placeholder = st.empty()
    progress_placeholder = st.empty()
    validation_debug_placeholder = st.empty()
    download_placeholder = st.empty()

    def render_status_message():
        """Render the persisted status message if available."""
        message = st.session_state.get("status_message", "")
        level = st.session_state.get("status_level", "info")
        if not message:
            status_placeholder.empty()
            return

        renderer = {
            "success": status_placeholder.success,
            "warning": status_placeholder.warning,
            "error": status_placeholder.error,
        }.get(level, status_placeholder.info)
        renderer(message)

    def update_status_message(message: str, level: str = "info"):
        """Persist a status message and refresh the placeholder."""
        st.session_state.status_message = message
        st.session_state.status_level = level
        render_status_message()

    render_status_message()

    def render_cached_artifacts():
        """Render stored input preview and raw LLM output within stable placeholders."""
        input_preview_placeholder.empty()
        raw_llm_placeholder.empty()

        if st.session_state.input_preview:
            with input_preview_placeholder.container():
                with st.expander(
                    f"View Full Input ({st.session_state.input_source_label or 'Input'})",
                    expanded=False,
                ):
                    st.text_area(
                        "Input Text",
                        st.session_state.input_preview,
                        height=300,
                        key="cached_input_preview",
                    )

        if st.session_state.raw_extraction_json:
            with raw_llm_placeholder.container():
                with st.expander("Debug: Raw LLM Extracted Data", expanded=False):
                    st.json(st.session_state.raw_extraction_json)

    if process_clicked:
        st.session_state.processing = True
        st.session_state.cancel_requested = False
        st.session_state.ui_log = []
        st.session_state.results = []
        st.session_state.search_logs = []
        st.session_state.validation_log_text = ""
        download_placeholder.empty()
        validation_debug_placeholder.empty()
        append_log("Initiating new extraction run.")

        if llm_provider != "ollama" and not api_key:
            st.error(
                f"Please provide an API key for `{llm_provider}` before processing."
            )
            st.session_state.processing = False
            return

        async def run_processing():
            try:
                # Extract text from input
                if input_mode == "PDF Upload" and uploaded_file:
                    with st.spinner("Extracting text from PDF..."):
                        TEMP_DIR.mkdir(parents=True, exist_ok=True)
                        temp_path = TEMP_DIR / "temp_references.pdf"
                        with open(temp_path, "wb") as temp_file:
                            temp_file.write(uploaded_file.getbuffer())
                        reference_text = extract_text_from_pdf(str(temp_path))
                        source_label = uploaded_file.name
                else:
                    reference_text = manual_text.strip()
                    source_label = "Manual Text Input"

                if not reference_text:
                    st.error("No text content detected. Please verify the input.")
                    append_log("Run aborted: empty input text.")
                    return

                append_log(
                    f"Captured {len(reference_text)} characters from {source_label}."
                )
                st.session_state.input_preview = reference_text
                st.session_state.input_source_label = source_label

                # Show extracted text immediately
                render_cached_artifacts()

                effective_overlap = min(
                    chunk_overlap_chars, max(0, chunk_size_chars - 100)
                )
                if effective_overlap != chunk_overlap_chars:
                    append_log(
                        f"Adjusted chunk overlap to {effective_overlap} to keep it below chunk size."
                    )

                # Initialize LLM processor
                processor = LLMProcessor(
                    provider=llm_provider,
                    model_name=model_name,
                    temperature=llm_temperature,
                    enable_web_search=enable_web_search_llm,
                    academic_domains=academic_domains or None,
                    event_logger=append_log,
                )

                # Extract references using LLM
                with st.spinner("Chunking document and prompting the LLM..."):
                    extracted_refs = await processor.extract_references(
                        reference_text,
                        chunk_size=chunk_size_chars,
                        overlap=effective_overlap,
                    )

                append_log(
                    f"LLM returned {len(extracted_refs)} references before filtering."
                )
                if not extracted_refs:
                    st.warning(
                        "No references detected. Try adjusting chunk size or input quality."
                    )
                    return

                st.session_state.raw_extraction_json = [
                    ref.model_dump() for ref in extracted_refs
                ]

                # Initialize validator if enabled
                validator = AcademicValidator() if enable_cross_validate else None

                # Process and enrich each reference
                progress_bar = progress_placeholder.progress(0)
                results: List[ReferenceData] = []
                search_logs: List[Dict[str, Any]] = []
                total_refs = len(extracted_refs)
                processed_count = 0
                cancelled = False

                async def validate_ref(ref, idx):
                    authors_payload = [author.model_dump() for author in ref.authors]
                    validation_result = await validator.validate_reference_details(
                        title=ref.title,
                        authors=authors_payload,
                        doi=ref.doi,
                    )
                    return validation_result, ref, idx

                if enable_cross_validate and validator:
                    import asyncio

                    validation_tasks = [
                        validate_ref(ref, idx) for idx, ref in enumerate(extracted_refs)
                    ]
                    validation_results = await asyncio.gather(
                        *validation_tasks, return_exceptions=True
                    )

                    with st.spinner(
                        "Enriching references and applying academic validation..."
                    ):
                        for res in validation_results:
                            if isinstance(res, Exception):
                                logger.error(f"Validation failed: {res}")
                                continue
                            validation_result, ref, idx = res
                            processed_count += 1
                            update_status_message(
                                f"Processing reference {processed_count}/{total_refs}: {ref.title or 'Unknown Title'}",
                                level="info",
                            )

                            # Update validation logs
                            if validation_result.logs:
                                validation_log_text = (
                                    st.session_state.validation_log_text
                                    + (
                                        "\n".join(
                                            [
                                                f"[{processed_count}/{total_refs}] {entry}"
                                                for entry in validation_result.logs
                                            ]
                                        )
                                        + "\n"
                                    )
                                )
                                st.session_state.validation_log_text = (
                                    validation_log_text
                                )
                            ref.search_context = "\n".join(validation_result.logs)

                            # Apply enrichment
                            apply_reference_enrichment(
                                ref, validation_result.reference_metadata
                            )
                            apply_author_enrichment(
                                ref, validation_result.author_metadata
                            )

                            # Extract and assign emails
                            found_emails = (
                                extract_emails(ref.search_context)
                                if ref.search_context
                                else []
                            )
                            if found_emails:
                                enrich_authors_with_emails(ref, found_emails)

                            search_logs.append(
                                {
                                    "index": processed_count,
                                    "title": ref.title,
                                    "reference_metadata": validation_result.reference_metadata,
                                    "author_metadata": validation_result.author_metadata,
                                    "logs": validation_result.logs,
                                    "emails": found_emails,
                                }
                            )
                            with validation_debug_placeholder.container():
                                render_validation_debug(
                                    search_logs, st.session_state.validation_log_text
                                )

                            results.append(ref)
                            progress_bar.progress(
                                min(processed_count / total_refs, 1.0)
                            )
                            render_results_table(results, results_placeholder)
                else:
                    # No validation, just add refs
                    results = extracted_refs

                st.session_state.results = results
                st.session_state.search_logs = search_logs

                if cancelled:
                    update_status_message(
                        "Processing cancelled by user.", level="warning"
                    )
                    st.warning("Processing cancelled. Partial results are shown above.")
                else:
                    update_status_message(
                        "Extraction and validation complete.", level="success"
                    )

                with validation_debug_placeholder.container():
                    render_validation_debug(
                        search_logs, st.session_state.validation_log_text
                    )

                if results:
                    append_log(f"Exported {len(results)} references to Excel.")
                    with download_placeholder.container():
                        render_download_button(results)
                progress_placeholder.empty()
            finally:
                st.session_state.processing = False
                st.session_state.cancel_requested = False
                progress_placeholder.empty()

        import asyncio

        asyncio.run(run_processing())

    else:
        if st.session_state.results:
            if not st.session_state.status_message:
                update_status_message("Ready. Showing the most recent extraction.")
            else:
                render_status_message()
            render_cached_artifacts()
            render_results_table(st.session_state.results, results_placeholder)
            with validation_debug_placeholder.container():
                render_validation_debug(
                    st.session_state.search_logs, st.session_state.validation_log_text
                )
            with download_placeholder.container():
                render_download_button(st.session_state.results)


if __name__ == "__main__":
    main()
