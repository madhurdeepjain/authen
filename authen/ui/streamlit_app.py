"""Streamlit UI for Authen - Reference Metadata Extraction & Validation."""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from authen.core import (
    set_log_dir,
    set_llm_cache_dir,
    set_validation_cache_dir,
    supports_web_search,
    PROVIDER_CONFIG,
    PROVIDER_ORDER,
)
from authen.references.models import ReferenceData
from authen.export.excel import export_references_to_excel
from authen.pdf.reader import extract_text_from_pdf
from authen.pipeline.orchestrator import (
    AuthenPipeline,
    PipelineConfig,
    PipelineHooks,
)

# Define application directories
APP_DIR = Path(__file__).resolve().parents[2]
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


def render_results_table(
    references: List[ReferenceData],
    placeholder,
    validation_states: Optional[List[str]] = None,
):
    """Render a table of extracted references with optional validation states."""
    table_rows = []
    for idx, ref in enumerate(references):
        authors_summary = ", ".join(
            filter(
                None,
                [
                    f"{author.first_name or ''} {author.last_name or ''}".strip()
                    for author in ref.authors
                ],
            )
        )
        status_display = "—"
        if validation_states and idx < len(validation_states):
            state = validation_states[idx]
            status_display = {
                "Validated": "✅ Validated",
                "Validating": "🔄 Validating",
                "Pending": "⏳ Pending",
                "Failed": "⚠️ Failed",
                "Skipped": "—",
            }.get(state, state or "—")
        table_rows.append(
            {
                "Title": ref.title,
                "Year": ref.year,
                "DOI": ref.doi,
                "Authors": authors_summary or "—",
                "Validation": status_display,
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


def render_validation_debug(search_logs: List[Dict[str, Any]], placeholder):
    """Render debug information for academic validation inside a placeholder."""
    placeholder.empty()
    with placeholder.container():
        if not search_logs:
            st.info("Validation debug will appear here once available.")
            return
        if search_logs:
            st.json(search_logs)


def run_app():
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
        "validation_states": [],
        "chunk_progress": {"total": 0, "completed": 0},
        "validation_progress": {"total": 0, "completed": 0},
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

    cancel_clicked = st.sidebar.button(
        "Cancel Processing",
        help="Signal the pipeline to stop at the next safe checkpoint.",
    )
    if cancel_clicked:
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

    input_preview_placeholder = st.empty()
    raw_llm_placeholder = st.empty()

    dashboard_cols = st.columns([1.2, 1.8], gap="large")
    with dashboard_cols[0]:
        metrics_placeholder = st.empty()
        chunk_progress_placeholder = st.empty()
        validation_progress_placeholder = st.empty()
        status_placeholder = st.empty()
    with dashboard_cols[1]:
        results_placeholder = st.empty()
        download_placeholder = st.empty()

    logs_expander = st.expander("Logs & Debug", expanded=False)
    with logs_expander:
        (
            llm_tab,
            raw_tab,
            validation_tab,
            debug_tab,
        ) = st.tabs(
            [
                "LLM Logs",
                "LLM Debug",
                "Validation Logs",
                "Validation Debug",
            ]
        )
        with llm_tab:
            log_placeholder = st.empty()
        with validation_tab:
            validation_log_placeholder = st.empty()
        with debug_tab:
            validation_debug_placeholder = st.empty()
        with raw_tab:
            raw_llm_placeholder = st.empty()

    def append_log(message: str):
        """Append a timestamped log message to the UI."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        st.session_state.ui_log.append(entry)
        log_placeholder.code("\n".join(st.session_state.ui_log), language=None)

    if st.session_state.ui_log:
        log_placeholder.code("\n".join(st.session_state.ui_log), language=None)

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

    def render_input_preview():
        """Render stored input preview within stable placeholders."""
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

    def render_raw_llm_output():
        """Render raw LLM output inside the debug tab."""

        raw_llm_placeholder.empty()
        data = st.session_state.get("raw_extraction_json", [])
        if data:
            raw_llm_placeholder.json(data)
        else:
            raw_llm_placeholder.info("Raw LLM output will appear here once available.")

    def render_metrics_board():
        """Render key progress metrics."""
        metrics_placeholder.empty()
        with metrics_placeholder.container():
            chunk_state = st.session_state.get("chunk_progress", {})
            chunk_total = chunk_state.get("total", 0)
            chunk_completed = chunk_state.get("completed", 0)
            total_refs = len(st.session_state.get("results", []))
            validated_refs = sum(
                1
                for state in st.session_state.get("validation_states", [])
                if state == "Validated"
            )
            cols = st.columns(3)
            cols[0].metric(
                "Chunks",
                f"{chunk_completed}/{chunk_total or '—'}",
                delta=None,
            )
            cols[1].metric("References Extracted", total_refs)
            cols[2].metric("References Validated", validated_refs)

    def render_chunk_progress_bar():
        """Render chunk progress bar."""
        chunk_state = st.session_state.get("chunk_progress", {})
        chunk_total = chunk_state.get("total", 0)
        chunk_completed = chunk_state.get("completed", 0)
        chunk_progress_placeholder.empty()
        if chunk_total > 0:
            chunk_progress_placeholder.progress(
                min(chunk_completed / chunk_total, 1.0),
                text=f"LLM Chunk Progress: {chunk_completed}/{chunk_total}",
            )
        else:
            chunk_progress_placeholder.info("LLM chunk progress will appear here.")

    def render_validation_progress_bar():
        """Render validation progress bar."""
        validation_state = st.session_state.get("validation_progress", {})
        total = validation_state.get("total", 0)
        completed = validation_state.get("completed", 0)
        validation_progress_placeholder.empty()
        if total > 0:
            validation_progress_placeholder.progress(
                min(completed / total, 1.0),
                text=f"Validation Progress: {completed}/{total}",
            )
        else:
            validation_progress_placeholder.info(
                "Validation progress will appear once validation starts."
            )

    def refresh_progress_widgets():
        """Refresh metrics and progress bars together."""
        render_metrics_board()
        render_chunk_progress_bar()
        render_validation_progress_bar()

    def render_validation_logs():
        """Render validation log text into its placeholder."""
        validation_log_placeholder.empty()
        log_text = st.session_state.get("validation_log_text", "").strip()
        if log_text:
            validation_log_placeholder.code(log_text, language=None)
        else:
            validation_log_placeholder.info("Validation logs will appear here.")

    def handle_validation_log(message: str):
        """Append validation log messages and re-render the log viewer."""
        existing = st.session_state.get("validation_log_text", "").strip()
        combined = f"{existing}\n{message}".strip() if existing else message
        st.session_state.validation_log_text = combined
        render_validation_logs()

    refresh_progress_widgets()
    render_validation_logs()
    render_validation_debug(
        st.session_state.get("search_logs", []),
        validation_debug_placeholder,
    )
    render_raw_llm_output()

    if process_clicked:
        st.session_state.processing = True
        st.session_state.cancel_requested = False
        st.session_state.ui_log = []
        log_placeholder.empty()
        st.session_state.results = []
        st.session_state.search_logs = []
        st.session_state.validation_log_text = ""
        st.session_state.validation_states = []
        st.session_state.chunk_progress = {"total": 0, "completed": 0}
        st.session_state.validation_progress = {"total": 0, "completed": 0}
        st.session_state.raw_extraction_json = []
        download_placeholder.empty()
        validation_debug_placeholder.empty()
        render_validation_logs()
        refresh_progress_widgets()
        append_log("Initiating new extraction run.")

        if llm_provider != "ollama" and not api_key:
            st.error(
                f"Please provide an API key for `{llm_provider}` before processing."
            )
            st.session_state.processing = False
            return

        def handle_pipeline_status(message: str):
            append_log(message)
            update_status_message(message, level="info")

        def handle_pipeline_progress(payload: Dict[str, int]):
            if "chunks_total" in payload:
                st.session_state.chunk_progress["total"] = payload["chunks_total"]
            if "chunks_completed" in payload:
                st.session_state.chunk_progress["completed"] = payload[
                    "chunks_completed"
                ]
            if "validation_total" in payload:
                st.session_state.validation_progress["total"] = payload[
                    "validation_total"
                ]
            if "validation_completed" in payload:
                st.session_state.validation_progress["completed"] = payload[
                    "validation_completed"
                ]
            refresh_progress_widgets()

        def handle_reference(reference: ReferenceData, index: int):
            if len(st.session_state.results) <= index:
                st.session_state.results.append(reference)
                st.session_state.raw_extraction_json.append(reference.model_dump())
                default_state = "Pending" if enable_cross_validate else "Skipped"
                st.session_state.validation_states.append(default_state)
            else:
                st.session_state.results[index] = reference
                st.session_state.raw_extraction_json[index] = reference.model_dump()
            render_results_table(
                st.session_state.results,
                results_placeholder,
                st.session_state.validation_states,
            )
            render_raw_llm_output()
            refresh_progress_widgets()

        def handle_validation_state(index: int, state: str):
            while len(st.session_state.validation_states) <= index:
                default_state = "Pending" if enable_cross_validate else "Skipped"
                st.session_state.validation_states.append(default_state)
            st.session_state.validation_states[index] = state
            render_results_table(
                st.session_state.results,
                results_placeholder,
                st.session_state.validation_states,
            )
            refresh_progress_widgets()

        def handle_validation_snapshot(snapshot: Dict[str, Any]):
            st.session_state.search_logs.append(snapshot)
            render_validation_debug(
                st.session_state.search_logs,
                validation_debug_placeholder,
            )

        effective_overlap = min(chunk_overlap_chars, max(0, chunk_size_chars - 100))
        if effective_overlap != chunk_overlap_chars:
            append_log(
                f"Adjusted chunk overlap to {effective_overlap} to keep it below chunk size."
            )

        pipeline_config = PipelineConfig(
            provider=llm_provider,
            model_name=model_name,
            temperature=llm_temperature,
            enable_web_search=enable_web_search_llm,
            enable_validation=enable_cross_validate,
            academic_domains=academic_domains or None,
            chunk_size=chunk_size_chars,
            chunk_overlap=effective_overlap,
        )

        pipeline_hooks = PipelineHooks(
            on_status=handle_pipeline_status,
            on_progress=handle_pipeline_progress,
            on_validation_log=handle_validation_log,
            on_reference=handle_reference,
            on_validation_state=handle_validation_state,
            on_validation_snapshot=handle_validation_snapshot,
            should_cancel=lambda: st.session_state.cancel_requested,
        )

        async def run_processing():
            try:
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
                render_input_preview()

                pipeline = AuthenPipeline(config=pipeline_config, hooks=pipeline_hooks)
                with st.spinner("Running extraction and validation pipeline..."):
                    references = await pipeline.run_from_text_async(
                        reference_text,
                        source_label=source_label,
                    )

                if not references:
                    st.warning(
                        "No references detected. Try adjusting chunk size or input quality."
                    )
                    return

                st.session_state.results = references
                render_results_table(
                    st.session_state.results,
                    results_placeholder,
                    st.session_state.validation_states,
                )
                render_raw_llm_output()
                render_validation_debug(
                    st.session_state.search_logs,
                    validation_debug_placeholder,
                )

                if st.session_state.cancel_requested:
                    update_status_message(
                        "Processing cancelled by user.", level="warning"
                    )
                    st.warning("Processing cancelled. Partial results are shown above.")
                else:
                    update_status_message(
                        "Extraction and validation complete.", level="success"
                    )

                if st.session_state.results:
                    append_log(
                        f"Exported {len(st.session_state.results)} references to Excel."
                    )
                    with download_placeholder.container():
                        render_download_button(st.session_state.results)
            finally:
                st.session_state.processing = False
                st.session_state.cancel_requested = False

        asyncio.run(run_processing())

    else:
        if st.session_state.results:
            if not st.session_state.status_message:
                update_status_message("Ready. Showing the most recent extraction.")
            else:
                render_status_message()
            render_input_preview()
            render_raw_llm_output()
            render_results_table(
                st.session_state.results,
                results_placeholder,
                st.session_state.validation_states,
            )
            render_validation_debug(
                st.session_state.search_logs,
                validation_debug_placeholder,
            )
            with download_placeholder.container():
                render_download_button(st.session_state.results)


if __name__ == "__main__":
    run_app()
