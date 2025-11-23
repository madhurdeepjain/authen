"""Extraction modules for Authen: PDF text extraction and LLM processing."""

import os
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_xai import ChatXAI

import pdfplumber

from .core import get_logger, supports_web_search, get_llm_cache_dir
from .models import ReferenceData, ReferenceList
from .utils import CacheManager, chunk_text, clean_json_output

load_dotenv()
logger = get_logger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    text_content = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_content += text + "\n"
        return text_content
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise e


class LLMProcessor:
    """Handles LLM-based extraction of references from text."""

    def __init__(
        self,
        provider="google",
        model_name="gemini-3-pro",
        temperature=0,
        enable_web_search=False,
        academic_domains: Optional[List[str]] = None,
        event_logger: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.enable_web_search = enable_web_search
        self.academic_domains = academic_domains or []
        self.event_logger = event_logger
        self.progress_callback = progress_callback

        # Use provided cache directory or global default
        if cache_dir is None:
            cache_dir = get_llm_cache_dir()
        self.cache = CacheManager(cache_dir)
        self.llm = self._get_llm()
        self.result_parser = PydanticOutputParser(pydantic_object=ReferenceList)

    def _log_event(self, message: str):
        """Send progress updates to the optional event logger."""
        if self.event_logger:
            try:
                self.event_logger(message)
            except Exception:
                logger.debug("UI logger callback failed", exc_info=True)
        else:
            logger.debug(message)

    def _emit_progress(self, event: str, payload: Optional[Dict[str, Any]] = None):
        """Send structured progress events to the optional callback."""
        if not self.progress_callback:
            return
        body = {"event": event}
        if payload:
            body.update(payload)
        try:
            self.progress_callback(body)
        except Exception:
            logger.debug("Progress callback failed", exc_info=True)

    def _get_llm(self):
        """Initialize the LLM based on provider configuration."""
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            llm = ChatOpenAI(
                model=self.model_name, temperature=self.temperature, api_key=api_key
            )
            if self.enable_web_search:
                if not supports_web_search(self.provider, self.model_name):
                    logger.warning(
                        f"Model {self.model_name} may not support web search. "
                        "Web search may not work as expected."
                    )
                return llm.bind_tools([{"type": "web_search_preview"}])
            return llm

        elif self.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            llm = ChatAnthropic(
                model=self.model_name, temperature=self.temperature, api_key=api_key
            )
            if self.enable_web_search:
                return llm.bind_tools(
                    [
                        {
                            "type": "web_search_20250305",
                            "name": "web_search",
                            "max_uses": 3,
                        }
                    ]
                )
            return llm

        elif self.provider == "google":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                google_api_key=api_key,
            )
            if self.enable_web_search:
                return llm.bind_tools([{"google_search": {}}])
            return llm

        elif self.provider == "grok":
            api_key = os.getenv("XAI_API_KEY")
            if not api_key:
                raise ValueError("XAI_API_KEY not found in environment variables")
            kwargs = {}
            if self.enable_web_search:
                kwargs["search_parameters"] = {"mode": "auto"}
            return ChatXAI(
                model=self.model_name,
                temperature=self.temperature,
                xai_api_key=api_key,
                **kwargs,
            )

        elif self.provider == "ollama":
            if self.enable_web_search:
                logger.warning(
                    "Ollama models do not support web search. Ignoring enable_web_search."
                )
            return ChatOllama(model=self.model_name, temperature=self.temperature)

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def extract_references(
        self, text: str, chunk_size: int = 15000, overlap: int = 2000
    ) -> AsyncIterator[ReferenceData]:
        """
        Split text into chunks and extract references using the LLM.
        Yields references as they are extracted from each chunk.
        Handles large text by chunking with overlap.
        """
        if not text or not text.strip():
            return

        self._log_event(
            f"Starting extraction | provider={self.provider} model={self.model_name} "
            f"chars={len(text)} chunk_size={chunk_size} overlap={overlap}"
        )

        chunks = (
            [text]
            if len(text) <= chunk_size
            else chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        )

        if len(chunks) == 1:
            self._log_event("Processing single chunk (no splitting needed).")

        seen_keys: set[str] = set()
        total_chunks = len(chunks)
        self._emit_progress("chunks_initialized", {"total": total_chunks})

        # Parallelize chunk processing with semaphore
        import asyncio

        semaphore = asyncio.Semaphore(3)  # Limit concurrent LLM calls

        async def process_chunk(index: int, chunk: str):
            async with semaphore:
                chunk_label = (
                    "full" if total_chunks == 1 else f"{index + 1}/{total_chunks}"
                )
                self._log_event(
                    f"[Chunk {chunk_label}] size={len(chunk)} characters. Beginning parse."
                )
                self._emit_progress(
                    "chunk_started", {"index": index + 1, "total": total_chunks}
                )
                referenced = await self._extract_references_chunk_async(
                    chunk, chunk_label=chunk_label
                )
                self._emit_progress(
                    "chunk_completed",
                    {
                        "index": index + 1,
                        "total": total_chunks,
                        "references": len(referenced),
                    },
                )
                return referenced

        tasks = [
            asyncio.create_task(process_chunk(i, chunk))
            for i, chunk in enumerate(chunks)
        ]
        for completed_task in asyncio.as_completed(tasks):
            res = await completed_task
            if isinstance(res, Exception):
                logger.error(f"Chunk processing failed: {res}")
                continue
            for ref in res:
                key = self._reference_key(ref)
                if key and key not in seen_keys:
                    seen_keys.add(key)
                    yield ref

        self._log_event(
            f"Finished extraction. Total references after dedupe: {len(seen_keys)}"
        )
        self._emit_progress("extraction_finished", {"total": total_chunks})

    async def _extract_references_chunk_async(
        self, text: str, chunk_label: str = "full"
    ) -> List[ReferenceData]:
        """Extract references from a single chunk of text asynchronously."""
        # Check cache first
        cache_key = self.cache.get_cache_key(
            f"{self.provider}:{self.model_name}:{text}"
        )
        cached = self.cache.get_cached_response(cache_key)
        if cached is not None:
            refs = [ReferenceData(**ref) for ref in cached]
            self._log_event(
                f"[Chunk {chunk_label}] Cache hit. Reusing {len(refs)} references."
            )
            return refs

        # Build academic domains restriction if provided
        domain_restriction = ""
        if self.academic_domains:
            domain_list = " OR ".join([f"site:{d}" for d in self.academic_domains])
            domain_restriction = f"\n\nWhen using web search, prioritize results from these academic domains: {domain_list}"

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert academic librarian. Identify and extract all individual references from the following text. "
                    "For each reference, extract the following structured data fields:\n"
                    "- raw_text: The original reference text as it appears in the document\n"
                    "- title: The full title of the academic paper\n"
                    "- authors: A list of authors, each with:\n"
                    "  * first_name: First name of the author\n"
                    "  * last_name: Last name of the author\n"
                    "  * title: Academic/Professional title (Prof., Dr., etc.)\n"
                    "  * country: Country of the author\n"
                    "  * affiliation: Detailed affiliation object with:\n"
                    "    - name: Institution/university name\n"
                    "    - department: Department or division\n"
                    "    - country: Country where institution is located\n"
                    "    - city: City where institution is located\n"
                    "  * emails: List of email addresses\n"
                    "  * address: Physical address if available\n"
                    "- year: Year of publication\n"
                    "- publication: Journal or conference name\n"
                    "- publisher: Publisher or organization responsible for the work\n"
                    "- volume: Volume number\n"
                    "- issue: Issue number\n"
                    "- pages: Page range\n"
                    "- doi: Digital Object Identifier\n"
                    "- isbn: ISBN if applicable\n"
                    "- url: URL to the paper if available\n"
                    "\n"
                    "If search tools are available, use them to:\n"
                    "1. Verify the existence of the paper.\n"
                    "2. Find and fill in missing author affiliations including institution name, department, country, and city.\n"
                    "3. Find and fill in author contact details (email, address).\n"
                    "4. Ensure author names are complete and correct.\n"
                    "5. Extract author country information.\n"
                    f"{domain_restriction}"
                    "\n"
                    "If a reference seems cut off at the beginning or end of the text, DO NOT include it. "
                    "Output ONLY the JSON object, no other text. "
                    "Ensure the output is a valid JSON instance, NOT the JSON schema. "
                    "Use the following format:\n{format_instructions}",
                ),
                ("user", "{text}"),
            ]
        )

        chain = (
            prompt
            | self.llm
            | RunnableLambda(self._handle_llm_response)
            | self.result_parser
        )

        try:
            self._log_event(f"[Chunk {chunk_label}] Invoking LLM for structured parse.")
            result = await chain.ainvoke(
                {
                    "text": text,
                    "format_instructions": self.result_parser.get_format_instructions(),
                }
            )
            references = result.references if hasattr(result, "references") else []

            # Cache the result
            data = [ref.model_dump() for ref in references]
            self.cache.save_cached_response(cache_key, data)
            self._log_event(
                f"[Chunk {chunk_label}] Parsed {len(references)} references via LLM."
            )

            return references
        except Exception as e:
            logger.error(f"Error extracting references from chunk: {e}", exc_info=True)
            self._log_event(
                f"[Chunk {chunk_label}] Failed to parse references. See logs."
            )
            return []

    def _handle_llm_response(self, ai_message):
        """
        Handle LLM response, extracting tool calls and results when web search is enabled.
        """
        messages = ai_message if isinstance(ai_message, list) else [ai_message]
        for msg in messages:
            if not isinstance(msg, AIMessage):
                continue
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                continue
            logger.info(f"Tool calls detected: {len(tool_calls)}")
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name", "unknown")
                    tool_args = tool_call.get("args", {})
                else:
                    tool_name = getattr(tool_call, "name", "unknown")
                    tool_args = getattr(tool_call, "args", {})
                logger.debug(f"Tool call: {tool_name} - {tool_args}")

        return clean_json_output(ai_message)

    @staticmethod
    def _reference_key(reference: ReferenceData) -> str:
        """Return a normalized key used for deduplicating references."""
        source = reference.title or reference.raw_text or ""
        return source.strip().lower()
