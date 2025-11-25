"""
OpenAlex API client with rate limiting, retry logic, and caching.

Implements best practices from OpenAlex LLM API guide:
- Use email for polite pool (10 req/sec)
- Batch DOI lookups (up to 50)
- Exponential backoff on errors
- Select only needed fields for performance
- Cache results to reduce API calls

Supports streaming mode where validation starts as soon as
references become available from the parser.
"""

import asyncio
from collections.abc import AsyncIterator
from urllib.parse import quote

import httpx
import structlog
from aiolimiter import AsyncLimiter
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from authen.core.cache import CacheManager, get_cache
from authen.core.schemas import (
    Affiliation,
    Author,
    ReferenceData,
    ValidationResult,
    ValidationStatus,
)

logger = structlog.get_logger()

# Base URL for OpenAlex API
OPENALEX_BASE_URL = "https://api.openalex.org"

# Fields to select for works (minimize response size)
WORKS_SELECT_FIELDS = [
    "id",
    "doi",
    "title",
    "publication_year",
    "type",
    "authorships",
    "primary_location",
    "cited_by_count",
    "open_access",
    "biblio",
]


class OpenAlexClient:
    """
    Async client for OpenAlex API with rate limiting and caching.

    Best practices implemented:
    - Email for polite pool (10 req/sec instead of 1)
    - Batch ID lookups using pipe separator
    - Exponential backoff on errors
    - Field selection for faster responses
    - Response caching for efficiency
    """

    def __init__(
        self,
        email: str = "user@example.com",
        rate_limit: int = 10,
        max_retries: int = 5,
        timeout: int = 30,
        cache: CacheManager | None = None,
        enable_cache: bool = True,
    ):
        """
        Initialize the OpenAlex client.

        Args:
            email: Email for polite pool access
            rate_limit: Requests per second (max 10 with email)
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
            cache: Optional cache manager
            enable_cache: Whether to enable caching
        """
        self.email = email
        self.max_retries = max_retries
        self.timeout = timeout
        self._cache = cache
        self._enable_cache = enable_cache

        # Rate limiter: X requests per second
        self.rate_limiter = AsyncLimiter(rate_limit, 1.0)

        # Reusable HTTP client
        self._client: httpx.AsyncClient | None = None

    @property
    def cache(self) -> CacheManager | None:
        """Get the cache manager, initializing if needed."""
        if not self._enable_cache:
            return None
        if self._cache is None:
            self._cache = get_cache(enabled=True)
        return self._cache

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={"User-Agent": f"authen/0.1.0 (mailto:{self.email})"},
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _build_url(
        self,
        endpoint: str,
        params: dict | None = None,
    ) -> str:
        """Build URL with parameters and email."""
        params = params or {}
        params["mailto"] = self.email

        query_parts = [f"{k}={quote(str(v))}" for k, v in params.items()]
        query_string = "&".join(query_parts)

        return f"{OPENALEX_BASE_URL}/{endpoint}?{query_string}"

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, asyncio.TimeoutError)),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        stop=stop_after_attempt(5),
    )
    async def _request(
        self,
        endpoint: str,
        params: dict | None = None,
    ) -> dict | None:
        """
        Make a rate-limited request to OpenAlex API.

        Returns:
            Response JSON or None if not found
        """
        async with self.rate_limiter:
            client = await self._get_client()
            url = self._build_url(endpoint, params)

            logger.debug("openalex_request", url=url)

            response = await client.get(url)

            if response.status_code == 404:
                return None

            if response.status_code == 403:
                logger.warning("rate_limited", url=url)
                raise httpx.HTTPError("Rate limited")

            response.raise_for_status()
            return response.json()

    async def get_work_by_doi(self, doi: str) -> dict | None:
        """
        Get a work by DOI.

        Args:
            doi: The DOI (with or without https://doi.org/ prefix)

        Returns:
            Work object or None if not found
        """
        # Normalize DOI
        norm_doi = doi.replace("https://doi.org/", "").lower()

        # Check cache first
        if self.cache:
            cached = self.cache.get_openalex_result(norm_doi, "doi")
            if cached:
                logger.debug("openalex_cache_hit_doi", doi=norm_doi)
                return cached

        if not doi.startswith("https://doi.org/"):
            doi = f"https://doi.org/{doi}"

        # Direct lookup by DOI
        try:
            result = await self._request(
                f"works/{quote(doi, safe='')}",
                {"select": ",".join(WORKS_SELECT_FIELDS)},
            )
            # Cache the result
            if result and self.cache:
                self.cache.set_openalex_result(norm_doi, "doi", result)
            return result
        except Exception as e:
            logger.warning("doi_lookup_failed", doi=doi, error=str(e))
            return None

    async def batch_get_works_by_doi(self, dois: list[str]) -> dict[str, dict | None]:
        """
        Batch lookup multiple DOIs (up to 50 per request).

        Args:
            dois: List of DOIs to look up

        Returns:
            Dict mapping DOI to work object (or None if not found)
        """
        results = {}
        dois_to_fetch = []

        # Check cache for each DOI first
        for doi in dois:
            norm_doi = doi.replace("https://doi.org/", "").lower()
            if self.cache:
                cached = self.cache.get_openalex_result(norm_doi, "doi")
                if cached:
                    results[norm_doi] = cached
                    logger.debug("openalex_batch_cache_hit", doi=norm_doi)
                    continue
            dois_to_fetch.append(doi)

        # Process uncached DOIs in batches of 50 (OpenAlex limit)
        for i in range(0, len(dois_to_fetch), 50):
            batch = dois_to_fetch[i : i + 50]

            # Normalize DOIs
            normalized = []
            for doi in batch:
                if not doi.startswith("https://doi.org/"):
                    doi = f"https://doi.org/{doi}"
                normalized.append(doi)

            # Build pipe-separated filter
            doi_filter = "|".join(normalized)

            try:
                response = await self._request(
                    "works",
                    {
                        "filter": f"doi:{doi_filter}",
                        "per-page": "50",
                        "select": ",".join(WORKS_SELECT_FIELDS),
                    },
                )

                if response and "results" in response:
                    for work in response["results"]:
                        work_doi = work.get("doi", "")
                        if work_doi:
                            # Normalize the returned DOI for matching
                            norm_doi = work_doi.replace("https://doi.org/", "").lower()
                            results[norm_doi] = work
                            # Cache the result
                            if self.cache:
                                self.cache.set_openalex_result(norm_doi, "doi", work)
            except Exception as e:
                logger.error("batch_doi_lookup_failed", error=str(e))

        # Fill in None for DOIs not found
        for doi in dois:
            norm_doi = doi.replace("https://doi.org/", "").lower()
            if norm_doi not in results:
                results[norm_doi] = None

        return results

    async def search_works_by_title(
        self,
        title: str,
    ) -> list[dict]:
        """
        Search for works by title.

        Args:
            title: Title to search for

        Returns:
            List of matching works
        """
        # Generate cache key from normalized title
        title_key = " ".join(title.lower().split())[:100]  # Limit key length

        # Check cache first
        if self.cache:
            cached = self.cache.get_openalex_result(title_key, "title_search")
            if cached:
                logger.debug("openalex_cache_hit_title", title=title[:50])
                return cached.get("results", [])

        params = {
            "filter": f"title.search:{quote(title)}",
            "per-page": "15",
            "select": ",".join(WORKS_SELECT_FIELDS),
        }

        try:
            response = await self._request("works", params)
            if response and "results" in response:
                results = response["results"]
                # Cache the results
                if self.cache:
                    self.cache.set_openalex_result(
                        title_key, "title_search", {"results": results}
                    )
                return results
        except Exception as e:
            logger.warning("title_search_failed", title=title, error=str(e))

        return []

    async def search_works_by_author(
        self,
        author_name: str,
        per_page: int = 25,
    ) -> list[dict]:
        """
        Search for works by author name.

        Args:
            author_name: Author name to search for
            per_page: Results per page

        Returns:
            List of matching works
        """
        params = {
            "filter": f"authorships.author.display_name.search:{quote(author_name)}",
            "per-page": str(per_page),
            "select": ",".join(WORKS_SELECT_FIELDS),
        }

        try:
            response = await self._request("works", params)
            if response and "results" in response:
                return response["results"]
        except Exception as e:
            logger.warning("author_search_failed", author=author_name, error=str(e))

        return []

    async def search_works(
        self,
        query: str,
        per_page: int = 10,
    ) -> list[dict]:
        """
        Full-text search for works.

        Args:
            query: Search query
            per_page: Results per page

        Returns:
            List of matching works
        """
        params = {
            "search": query,
            "per-page": str(per_page),
            "select": ",".join(WORKS_SELECT_FIELDS),
        }

        try:
            response = await self._request("works", params)
            if response and "results" in response:
                return response["results"]
        except Exception as e:
            logger.warning("search_failed", query=query, error=str(e))

        return []


class OpenAlexValidator:
    """
    Validate references against OpenAlex and enrich with data.

    Validation strategy (prioritizing title and authors):
    1. If DOI exists, lookup by DOI (fast, accurate)
    2. If DOI lookup fails or no DOI, search by title
       - Rank candidates primarily by author similarity
       - Use year/DOI as secondary tiebreakers
    3. If title search fails, search by author name
       - Rank candidates primarily by title similarity
       - Use year/DOI as secondary tiebreakers
    4. Enrich reference with OpenAlex data
    """

    def __init__(
        self,
        email: str = "user@example.com",
        rate_limit: int = 10,
        title_threshold: float = 0.85,
        author_threshold: float = 0.7,
        max_retries: int = 5,
        timeout: int = 30,
        cache: CacheManager | None = None,
        enable_cache: bool = True,
    ):
        """
        Initialize the validator.

        Args:
            email: Email for OpenAlex polite pool
            rate_limit: Requests per second
            title_threshold: Minimum title similarity (0-1)
            author_threshold: Minimum author similarity (0-1)
            max_retries: Maximum retry attempts
            timeout: Request timeout
            cache: Optional cache manager
            enable_cache: Whether to enable caching
        """
        self.client = OpenAlexClient(
            email=email,
            rate_limit=rate_limit,
            max_retries=max_retries,
            timeout=timeout,
            cache=cache,
            enable_cache=enable_cache,
        )
        self.title_threshold = title_threshold
        self.author_threshold = author_threshold

    async def close(self) -> None:
        """Close the client."""
        await self.client.close()

    async def validate_streaming(
        self,
        references: AsyncIterator[ReferenceData],
        max_concurrent: int = 5,
    ) -> AsyncIterator[ValidationResult]:
        """
        Validate references in streaming mode.

        Processes references as they arrive, validating in parallel
        while respecting rate limits. DOI references are batched
        when possible for efficiency.

        Args:
            references: Async iterator of references (e.g., from parser)
            max_concurrent: Maximum concurrent validation tasks

        Yields:
            ValidationResult as each reference is validated
        """
        logger.info("starting_streaming_validation")

        # Semaphore for concurrency control (rate limit is handled by aiolimiter)
        semaphore = asyncio.Semaphore(max_concurrent)
        output_queue: asyncio.Queue[ValidationResult | None] = asyncio.Queue()
        pending_tasks: set[asyncio.Task] = set()

        # DOI batching state
        doi_batch: list[ReferenceData] = []
        doi_batch_lock = asyncio.Lock()
        DOI_BATCH_SIZE = 25  # Batch DOIs for efficiency (max 50)
        DOI_BATCH_TIMEOUT = 0.5  # Flush batch after this many seconds

        async def validate_single(ref: ReferenceData) -> None:
            """Validate a single non-DOI reference."""
            async with semaphore:
                try:
                    result = await self._validate_by_search(ref)
                    await output_queue.put(result)
                except Exception as e:
                    logger.error("streaming_validation_error", error=str(e))
                    await output_queue.put(
                        ValidationResult(
                            original=ref,
                            status=ValidationStatus.ERROR,
                            error_message=str(e),
                        )
                    )

        async def flush_doi_batch() -> None:
            """Flush the current DOI batch."""
            async with doi_batch_lock:
                if not doi_batch:
                    return
                batch = doi_batch.copy()
                doi_batch.clear()

            if batch:
                async with semaphore:
                    try:
                        results = await self._validate_by_doi(batch)
                        for result in results:
                            await output_queue.put(result)
                    except Exception as e:
                        logger.error("doi_batch_validation_error", error=str(e))
                        for ref in batch:
                            await output_queue.put(
                                ValidationResult(
                                    original=ref,
                                    status=ValidationStatus.ERROR,
                                    error_message=str(e),
                                )
                            )

        async def batch_flush_timer() -> None:
            """Periodically flush DOI batch to avoid waiting too long."""
            while True:
                await asyncio.sleep(DOI_BATCH_TIMEOUT)
                await flush_doi_batch()

        # Start batch flush timer
        flush_timer_task = asyncio.create_task(batch_flush_timer())

        async def process_references() -> None:
            """Process all incoming references."""
            try:
                async for ref in references:
                    if ref.doi:
                        # Add to DOI batch
                        batch_to_process = None
                        async with doi_batch_lock:
                            doi_batch.append(ref)
                            if len(doi_batch) >= DOI_BATCH_SIZE:
                                batch_to_process = doi_batch.copy()
                                doi_batch.clear()

                        if batch_to_process:
                            task = asyncio.create_task(
                                self._validate_doi_batch_and_queue(
                                    batch_to_process, semaphore, output_queue
                                )
                            )
                            pending_tasks.add(task)
                            task.add_done_callback(pending_tasks.discard)
                    else:
                        # Validate immediately
                        task = asyncio.create_task(validate_single(ref))
                        pending_tasks.add(task)
                        task.add_done_callback(pending_tasks.discard)

                # Flush remaining DOI batch
                await flush_doi_batch()

                # Wait for all pending tasks
                if pending_tasks:
                    await asyncio.gather(*pending_tasks, return_exceptions=True)

            finally:
                flush_timer_task.cancel()
                try:
                    await flush_timer_task
                except asyncio.CancelledError:
                    pass
                await output_queue.put(None)  # Signal completion

        # Start processing in background
        process_task = asyncio.create_task(process_references())

        # Yield results as they arrive
        validated_count = 0
        try:
            while True:
                result = await output_queue.get()
                if result is None:
                    break
                validated_count += 1
                yield result
        finally:
            await process_task

        logger.info("streaming_validation_complete", total=validated_count)

    async def _validate_doi_batch_and_queue(
        self,
        batch: list[ReferenceData],
        semaphore: asyncio.Semaphore,
        output_queue: asyncio.Queue,
    ) -> None:
        """Validate a batch of DOI references and put results in queue."""
        async with semaphore:
            try:
                results = await self._validate_by_doi(batch)
                for result in results:
                    await output_queue.put(result)
            except Exception as e:
                logger.error("doi_batch_validation_error", error=str(e))
                for ref in batch:
                    await output_queue.put(
                        ValidationResult(
                            original=ref,
                            status=ValidationStatus.ERROR,
                            error_message=str(e),
                        )
                    )

    async def validate(
        self,
        references: list[ReferenceData],
    ) -> list[ValidationResult]:
        """
        Validate a list of references against OpenAlex.

        Args:
            references: References to validate

        Returns:
            List of validation results
        """
        logger.info("starting_validation", count=len(references))

        # Separate references with and without DOIs
        with_doi = [(i, ref) for i, ref in enumerate(references) if ref.doi]
        without_doi = [(i, ref) for i, ref in enumerate(references) if not ref.doi]

        results = [None] * len(references)

        # Batch validate DOI references
        if with_doi:
            doi_results = await self._validate_by_doi([ref for _, ref in with_doi])
            for (i, _), result in zip(with_doi, doi_results):
                results[i] = result

        # Validate non-DOI references
        for i, ref in without_doi:
            result = await self._validate_by_search(ref)
            results[i] = result

        # Ensure all results are filled
        final_results = []
        for i, result in enumerate(results):
            if result is None:
                final_results.append(
                    ValidationResult(
                        original=references[i],
                        status=ValidationStatus.ERROR,
                        error_message="Validation not completed",
                    )
                )
            else:
                final_results.append(result)

        validated = sum(
            1 for r in final_results if r.status == ValidationStatus.VALIDATED
        )
        partial = sum(
            1 for r in final_results if r.status == ValidationStatus.PARTIAL_MATCH
        )
        not_found = sum(
            1 for r in final_results if r.status == ValidationStatus.NOT_FOUND
        )
        errors = sum(1 for r in final_results if r.status == ValidationStatus.ERROR)

        logger.info(
            "validation_complete",
            total=len(references),
            validated=validated,
            partial=partial,
            not_found=not_found,
            errors=errors,
        )

        return final_results

    async def _validate_by_doi(
        self, references: list[ReferenceData]
    ) -> list[ValidationResult]:
        """Validate references with DOIs using batch lookup."""
        dois = [ref.doi for ref in references if ref.doi]

        # Batch lookup
        doi_works = await self.client.batch_get_works_by_doi(dois)

        results = []
        for ref in references:
            if not ref.doi:
                results.append(
                    ValidationResult(
                        original=ref,
                        status=ValidationStatus.ERROR,
                        error_message="No DOI provided",
                    )
                )
                continue

            norm_doi = ref.doi.replace("https://doi.org/", "")
            work = doi_works.get(norm_doi)

            if work:
                result = self._create_validation_result(ref, work, "doi")
            else:
                # DOI not found, try search-based validation
                result = await self._validate_by_search(ref)

            results.append(result)

        return results

    async def _validate_by_search(self, ref: ReferenceData) -> ValidationResult:
        """
        Validate a reference using search strategies.

        Strategy:
        1. Search by title first, rank by author similarity
        2. If title search fails, search by author, rank by title similarity
        """
        # Try title search first
        if ref.title:
            result = await self._search_by_title(ref)
            if result.status != ValidationStatus.NOT_FOUND:
                return result

        # Fallback: search by author
        if ref.authors:
            result = await self._search_by_author(ref)
            if result.status != ValidationStatus.NOT_FOUND:
                return result

        # Nothing found
        return ValidationResult(
            original=ref,
            status=ValidationStatus.NOT_FOUND,
            match_method="search",
            error_message="No matches found by title or author search",
        )

    async def _search_by_title(self, ref: ReferenceData) -> ValidationResult:
        """Search by title, rank candidates by author similarity."""
        if not ref.title:
            return ValidationResult(
                original=ref,
                status=ValidationStatus.NOT_FOUND,
                error_message="No title available",
            )

        try:
            search_title = ref.get_search_title()
            works = await self.client.search_works_by_title(search_title)

            if not works:
                # Try broader full-text search
                works = await self.client.search_works(search_title)

            if not works:
                return ValidationResult(
                    original=ref,
                    status=ValidationStatus.NOT_FOUND,
                    match_method="title_search",
                )

            # Find best match: primarily by author, with secondary factors
            best_match = self._find_best_candidate(ref, works, primary_weight="author")

            if best_match:
                return best_match

            return ValidationResult(
                original=ref,
                status=ValidationStatus.NOT_FOUND,
                match_method="title_search",
            )

        except Exception as e:
            logger.error("title_search_error", title=ref.title, error=str(e))
            return ValidationResult(
                original=ref,
                status=ValidationStatus.ERROR,
                error_message=str(e),
            )

    async def _search_by_author(self, ref: ReferenceData) -> ValidationResult:
        """Search by author, rank candidates by title similarity."""
        if not ref.authors:
            return ValidationResult(
                original=ref,
                status=ValidationStatus.NOT_FOUND,
                error_message="No authors available",
            )

        try:
            # Use first author for search (most reliable)
            first_author = ref.authors[0].display_name
            works = await self.client.search_works_by_author(first_author)

            if not works:
                return ValidationResult(
                    original=ref,
                    status=ValidationStatus.NOT_FOUND,
                    match_method="author_search",
                )

            # Find best match: primarily by title, with secondary factors
            best_match = self._find_best_candidate(ref, works, primary_weight="title")

            if best_match:
                best_match.match_method = "author_search"
                return best_match

            return ValidationResult(
                original=ref,
                status=ValidationStatus.NOT_FOUND,
                match_method="author_search",
            )

        except Exception as e:
            logger.error("author_search_error", error=str(e))
            return ValidationResult(
                original=ref,
                status=ValidationStatus.ERROR,
                error_message=str(e),
            )

    def _find_best_candidate(
        self,
        ref: ReferenceData,
        works: list[dict],
        primary_weight: str = "author",
    ) -> ValidationResult | None:
        """
        Find the best matching work from candidates.

        Args:
            ref: Original reference
            works: List of candidate works
            primary_weight: "author" or "title" - which to prioritize

        Returns:
            ValidationResult if good match found, None otherwise
        """
        candidates = []

        for work in works:
            title_sim = self._calculate_title_similarity(
                ref.title or "", work.get("title", "")
            )
            author_sim = self._calculate_author_similarity(
                ref.authors, work.get("authorships", [])
            )

            # Secondary factors (small bonuses for matching)
            year_bonus = 0.0
            doi_bonus = 0.0

            # Year match bonus
            work_year = str(work.get("publication_year", ""))
            if ref.year and work_year and ref.year == work_year:
                year_bonus = 0.05

            # DOI match bonus (if ref has DOI and it matches)
            work_doi = (work.get("doi") or "").replace("https://doi.org/", "")
            ref_doi = (ref.doi or "").replace("https://doi.org/", "")
            if ref_doi and work_doi and ref_doi.lower() == work_doi.lower():
                doi_bonus = 0.1

            # Calculate score based on primary weight
            if primary_weight == "author":
                # When searching by title, authors are the differentiator
                score = (author_sim * 0.6) + (title_sim * 0.3) + year_bonus + doi_bonus
            else:
                # When searching by author, title is the differentiator
                score = (title_sim * 0.6) + (author_sim * 0.3) + year_bonus + doi_bonus

            candidates.append(
                {
                    "work": work,
                    "score": score,
                    "title_sim": title_sim,
                    "author_sim": author_sim,
                }
            )

        if not candidates:
            return None

        # Sort by score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]

        # Check if best candidate meets thresholds
        if best["title_sim"] >= self.title_threshold:
            result = self._create_validation_result(ref, best["work"], "title_search")
            result.title_similarity = best["title_sim"]
            result.author_similarity = best["author_sim"]
            result.confidence = best["score"]
            return result

        return None

    def _create_validation_result(
        self,
        original: ReferenceData,
        work: dict,
        match_method: str,
    ) -> ValidationResult:
        """Create a validation result from an OpenAlex work."""
        # Extract enriched reference data
        validated_ref = self._extract_reference_data(work, original)

        # Calculate similarity scores
        title_sim = self._calculate_title_similarity(
            original.title or "", work.get("title", "")
        )
        author_sim = self._calculate_author_similarity(
            original.authors, work.get("authorships", [])
        )

        # Determine status
        if title_sim >= self.title_threshold:
            if author_sim >= self.author_threshold:
                status = ValidationStatus.VALIDATED
            else:
                status = ValidationStatus.PARTIAL_MATCH
        else:
            status = ValidationStatus.PARTIAL_MATCH

        confidence = (title_sim * 0.7) + (author_sim * 0.3)

        openalex_id = work.get("id", "")
        openalex_url = openalex_id if openalex_id.startswith("http") else None

        return ValidationResult(
            original=original,
            validated=validated_ref,
            status=status,
            confidence=confidence,
            title_similarity=title_sim,
            author_similarity=author_sim,
            match_method=match_method,
            openalex_url=openalex_url,
            openalex_raw=work,
        )

    def _extract_reference_data(
        self, work: dict, original: ReferenceData
    ) -> ReferenceData:
        """Extract reference data from OpenAlex work object."""
        # Get DOI
        doi = work.get("doi", "")
        if doi:
            doi = doi.replace("https://doi.org/", "")

        # Get publication info
        primary_location = work.get("primary_location", {}) or {}
        source = primary_location.get("source", {}) or {}

        publication = source.get("display_name")
        publisher = source.get("publisher")

        # Get biblio info
        biblio = work.get("biblio", {}) or {}

        # Extract authors
        authors = []
        for authorship in work.get("authorships", []):
            author_info = authorship.get("author", {}) or {}

            # Parse name
            display_name = author_info.get("display_name", "")
            first_name = None
            last_name = None

            if display_name:
                parts = display_name.split()
                if len(parts) >= 2:
                    first_name = parts[0]
                    last_name = " ".join(parts[1:])
                elif parts:
                    last_name = parts[0]

            # Get affiliations
            affiliations = []
            for inst in authorship.get("institutions", []):
                aff = Affiliation(
                    name=inst.get("display_name"),
                    country=inst.get("country_code"),
                    ror_id=inst.get("ror"),
                    openalex_id=inst.get("id"),
                )
                affiliations.append(aff)

            author = Author(
                first_name=first_name,
                last_name=last_name,
                full_name=display_name,
                affiliations=affiliations,
                orcid=author_info.get("orcid"),
                openalex_id=author_info.get("id"),
            )
            authors.append(author)

        return ReferenceData(
            raw_text=original.raw_text,
            reference_number=original.reference_number,
            title=work.get("title"),
            authors=authors,
            year=str(work.get("publication_year", "")),
            publication=publication,
            publisher=publisher,
            volume=biblio.get("volume"),
            issue=biblio.get("issue"),
            pages=f"{biblio.get('first_page', '')}-{biblio.get('last_page', '')}"
            if biblio.get("first_page")
            else None,
            doi=doi,
            url=original.url,
            work_type=work.get("type"),
            openalex_id=work.get("id"),
            cited_by_count=work.get("cited_by_count"),
            is_open_access=work.get("open_access", {}).get("is_oa"),
        )

    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles."""
        if not title1 or not title2:
            return 0.0

        from rapidfuzz import fuzz

        # Normalize titles
        t1 = title1.lower().strip()
        t2 = title2.lower().strip()

        # Use token set ratio for robustness to word order
        return fuzz.token_set_ratio(t1, t2) / 100.0

    def _calculate_author_similarity(
        self,
        authors1: list[Author],
        authorships: list[dict],
    ) -> float:
        """Calculate similarity between author lists."""
        if not authors1 or not authorships:
            return 0.0 if (authors1 or authorships) else 1.0

        from rapidfuzz import fuzz

        # Get author names from original
        names1 = set()
        for author in authors1:
            name = author.display_name.lower()
            names1.add(name)

        # Get author names from OpenAlex
        names2 = set()
        for authorship in authorships:
            author_info = authorship.get("author", {}) or {}
            name = author_info.get("display_name", "").lower()
            if name:
                names2.add(name)

        if not names1 or not names2:
            return 0.0

        # Calculate best match for each author
        matches = 0
        for name1 in names1:
            best_match = max(fuzz.token_set_ratio(name1, name2) for name2 in names2)
            if best_match >= 70:  # 70% similarity threshold
                matches += 1

        return matches / max(len(names1), len(names2))


async def validate_references(
    references: list[ReferenceData],
    email: str = "user@example.com",
    rate_limit: int = 10,
) -> list[ValidationResult]:
    """
    Convenience function to validate references.

    Args:
        references: References to validate
        email: Email for OpenAlex polite pool
        rate_limit: Requests per second

    Returns:
        List of validation results
    """
    validator = OpenAlexValidator(email=email, rate_limit=rate_limit)
    try:
        return await validator.validate(references)
    finally:
        await validator.close()
