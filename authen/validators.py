"""Academic validation using multiple academic APIs."""

import asyncio
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from .core import get_logger, get_validation_cache_dir
from .models import AcademicValidationResult
from .utils import normalize_author_key, split_name

logger = get_logger(__name__)


class AcademicValidator:
    """
    Validates references using academic databases and APIs.
    Provides both validation logs and structured metadata that can enrich results.
    """

    def __init__(self):
        self.crossref_base = "https://api.crossref.org/works"
        self.pubmed_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.arxiv_base = "http://export.arxiv.org/api/query"
        self.semantic_scholar_base = "https://api.semanticscholar.org/graph/v1"
        self.openalex_base = "https://api.openalex.org"
        self.rate_limit_delay = 0.1  # Reduced delay
        self.max_concurrent = 5  # Max concurrent requests
        # Per-API semaphores to prevent simultaneous requests to same API
        self.crossref_sem = asyncio.Semaphore(1)
        self.openalex_sem = asyncio.Semaphore(1)
        self.semantic_sem = asyncio.Semaphore(1)
        self.pubmed_sem = asyncio.Semaphore(1)
        self.arxiv_sem = asyncio.Semaphore(1)
        self.cache_dir = get_validation_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(
        self, title: Optional[str], doi: Optional[str], authors: List[dict]
    ) -> str:
        """Generate a cache key for validation request."""
        normalized_authors = []
        for author in authors or []:
            normalized_authors.append(
                normalize_author_key(
                    author.get("first_name"),
                    author.get("last_name"),
                    fallback="",
                )
            )

        payload = {
            "title": (title or "").strip(),
            "doi": (doi or "").strip(),
            "authors": normalized_authors,
        }
        serialized = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _load_cached_validation(
        self, cache_key: str
    ) -> Optional[AcademicValidationResult]:
        """Load cached validation result if available."""
        cache_path = self.cache_dir / f"{cache_key}.json"
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as cache_file:
                data = json.load(cache_file)
            logger.info(f"Academic validation cache hit ({cache_key[:8]}...).")
            return AcademicValidationResult(
                logs=data.get("logs", []),
                reference_metadata=data.get("reference_metadata") or {},
                author_metadata=data.get("author_metadata") or {},
            )
        except Exception as exc:
            logger.warning(f"Failed to read validation cache {cache_path}: {exc}")
            return None

    def _save_cached_validation(
        self, cache_key: str, result: AcademicValidationResult
    ) -> None:
        """Save validation result to cache."""
        cache_path = self.cache_dir / f"{cache_key}.json"
        payload = {
            "logs": result.logs,
            "reference_metadata": result.reference_metadata,
            "author_metadata": result.author_metadata,
        }
        try:
            with open(cache_path, "w", encoding="utf-8") as cache_file:
                json.dump(payload, cache_file, indent=2, ensure_ascii=False)
            logger.info(f"Academic validation cached ({cache_key[:8]}...).")
        except Exception as exc:
            logger.warning(f"Failed to write validation cache {cache_path}: {exc}")

    async def _safe_request(
        self,
        session: aiohttp.ClientSession,
        url: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        expect_json: bool = True,
    ) -> Optional[Any]:
        """Make a safe async HTTP request with error handling."""
        try:
            await asyncio.sleep(self.rate_limit_delay)
            async with session.get(
                url,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                response.raise_for_status()
                if expect_json:
                    return await response.json()
                return await response.text()
        except aiohttp.ClientError as e:
            logger.warning(f"API request failed for {url}: {e}")
            return None
        except ValueError as e:
            logger.warning(f"Failed to decode response from {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in API request: {e}")
            return None

    def _format_crossref_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Map Crossref response to our metadata structure."""
        metadata: Dict[str, Any] = {
            "title": (item.get("title", [None])[0] if item.get("title") else None),
            "publication": (
                item.get("container-title", [None])[0]
                if item.get("container-title")
                else None
            ),
            "publisher": item.get("publisher"),
            "volume": item.get("volume"),
            "issue": item.get("issue"),
            "pages": item.get("page"),
            "doi": item.get("DOI"),
            "isbn": (item.get("ISBN", [None])[0] if item.get("ISBN") else None),
            "url": item.get("URL"),
        }

        issued = item.get("issued", {}).get("date-parts", [])
        if issued and issued[0]:
            metadata["year"] = str(issued[0][0])

        authors = []
        for author in item.get("author", []):
            aff_details = None
            if author.get("affiliation"):
                aff = author["affiliation"][0]
                aff_details = {
                    "name": aff.get("name"),
                    "department": aff.get("department"),
                    "country": aff.get("country"),
                    "city": aff.get("city"),
                }
            authors.append(
                {
                    "first_name": author.get("given"),
                    "last_name": author.get("family"),
                    "title": author.get("prefix"),
                    "country": (aff_details or {}).get("country"),
                    "affiliation": aff_details,
                    "emails": [],
                    "address": None,
                }
            )
        metadata["authors"] = authors
        return metadata

    def _format_openalex_work(self, work: Dict[str, Any]) -> Dict[str, Any]:
        """Map OpenAlex work response to our metadata structure."""
        host = work.get("host_venue") or {}
        primary_loc = work.get("primary_location") or {}
        oa_location = work.get("best_oa_location") or {}
        metadata: Dict[str, Any] = {
            "title": work.get("title") or work.get("display_name"),
            "publication": host.get("display_name"),
            "publisher": host.get("publisher"),
            "volume": work.get("biblio", {}).get("volume"),
            "issue": work.get("biblio", {}).get("issue"),
            "pages": work.get("biblio", {}).get("pages"),
            "doi": work.get("doi"),
            "isbn": None,
            "url": oa_location.get("url")
            or primary_loc.get("source", {}).get("homepage_url")
            or host.get("url"),
            "year": str(work.get("publication_year"))
            if work.get("publication_year")
            else None,
        }

        authors = []
        for authorship in (work.get("authorships") or [])[:15]:
            if not isinstance(authorship, dict):
                continue
            author_info = authorship.get("author", {}) or {}
            first_name, last_name = split_name(author_info.get("display_name"))
            institutions = authorship.get("institutions") or []
            aff = institutions[0] if institutions else {}
            authors.append(
                {
                    "first_name": first_name,
                    "last_name": last_name,
                    "title": None,
                    "country": aff.get("country_code") or aff.get("country"),
                    "affiliation": {
                        "name": aff.get("display_name"),
                        "department": None,
                        "country": aff.get("country") or aff.get("country_code"),
                        "city": aff.get("city"),
                    }
                    if aff
                    else None,
                    "emails": [],
                    "address": None,
                }
            )
        metadata["authors"] = authors
        return metadata

    def _format_openalex_author(self, author: Dict[str, Any]) -> Dict[str, Any]:
        """Map OpenAlex author response to our metadata structure."""
        if not isinstance(author, dict):
            return {}
        institution = author.get("last_known_institution") or {}
        return {
            "title": None,
            "country": institution.get("country") or institution.get("country_code"),
            "affiliation": {
                "name": institution.get("display_name"),
                "department": None,
                "country": institution.get("country")
                or institution.get("country_code"),
                "city": institution.get("city"),
            }
            if institution
            else None,
            "emails": [],
        }

    async def search_crossref(
        self, session: aiohttp.ClientSession, title: str, doi: Optional[str] = None
    ) -> Tuple[List[str], Optional[Dict[str, Any]]]:
        """Search Crossref API for paper metadata."""
        async with self.crossref_sem:
            logs: List[str] = []
            metadata = None
            try:
                if doi:
                    url = f"{self.crossref_base}/{doi}"
                    data = await self._safe_request(session, url)
                    item = data.get("message") if data else None
                else:
                    params = {"query.title": title, "rows": 3}
                    data = await self._safe_request(
                        session, self.crossref_base, params=params
                    )
                    items = data.get("message", {}).get("items", []) if data else []
                    item = items[0] if items else None
                if item:
                    metadata = self._format_crossref_item(item)
                    logs.append(
                        f"Crossref match: {metadata.get('title') or 'Unknown title'} (DOI: {metadata.get('doi')})"
                    )
                else:
                    logs.append("Crossref: No close matches found.")
            except Exception as e:
                logger.error(f"Error in Crossref search: {e}")
                logs.append("Crossref search failed.")
            return logs, metadata

    async def search_pubmed(
        self, session: aiohttp.ClientSession, title: str
    ) -> List[str]:
        """Search PubMed API for biomedical papers."""
        async with self.pubmed_sem:
            results = []
            try:
                search_params = {
                    "db": "pubmed",
                    "term": f'"{title}"[Title]',
                    "retmode": "json",
                    "retmax": 3,
                }
                search_url = f"{self.pubmed_base}/esearch.fcgi"
                search_data = await self._safe_request(
                    session, search_url, params=search_params
                )

                if (
                    search_data
                    and "esearchresult" in search_data
                    and "idlist" in search_data["esearchresult"]
                ):
                    pmids = search_data["esearchresult"]["idlist"]
                    if pmids:
                        results.append(
                            f"PubMed: Found {len(pmids)} potential match(es)."
                        )
            except Exception as e:
                logger.error(f"Error in PubMed search: {e}")
            return results

    async def search_arxiv(
        self, session: aiohttp.ClientSession, title: str
    ) -> List[str]:
        """Search arXiv API for preprints."""
        async with self.arxiv_sem:
            results = []
            try:
                params = {"search_query": f'ti:"{title}"', "start": 0, "max_results": 3}
                response_text = await self._safe_request(
                    session, self.arxiv_base, params=params, expect_json=False
                )
                if response_text and "<entry>" in response_text:
                    results.append("arXiv: Found potential matching preprints.")
            except Exception as e:
                logger.error(f"Error in arXiv search: {e}")
            return results

    async def search_semantic_scholar(
        self, session: aiohttp.ClientSession, title: str, doi: Optional[str] = None
    ) -> List[str]:
        """Search Semantic Scholar API."""
        async with self.semantic_sem:
            results = []
            try:
                if doi:
                    url = f"{self.semantic_scholar_base}/paper/DOI:{doi}"
                    data = await self._safe_request(session, url)
                    if data:
                        results.append(
                            f"Semantic Scholar: {data.get('title', 'Match found')}."
                        )
                else:
                    params = {"query": title, "limit": 2}
                    url = f"{self.semantic_scholar_base}/paper/search"
                    data = await self._safe_request(session, url, params=params)
                    if data and "data" in data:
                        for paper in data["data"]:
                            results.append(
                                f"Semantic Scholar candidate: {paper.get('title', 'Unknown')}"
                            )
            except Exception as e:
                logger.error(f"Error in Semantic Scholar search: {e}")
            return results

    async def search_openalex(
        self, session: aiohttp.ClientSession, title: str, doi: Optional[str] = None
    ) -> Tuple[List[str], Optional[Dict[str, Any]]]:
        """Search OpenAlex API."""
        async with self.openalex_sem:
            logs: List[str] = []
            metadata = None
            try:
                if doi:
                    url = f"{self.openalex_base}/works/doi:{doi}"
                    data = await self._safe_request(session, url)
                    work = data if data else None
                else:
                    url = f"{self.openalex_base}/works"
                    params = {"search": title, "per_page": 1}
                    data = await self._safe_request(session, url, params=params)
                    works = data.get("results", []) if data else []
                    work = works[0] if works else None
                if work:
                    metadata = self._format_openalex_work(work)
                    logs.append(
                        f"OpenAlex match: {metadata.get('title') or 'Unknown title'} (Year: {metadata.get('year')})"
                    )
                else:
                    logs.append("OpenAlex: No close matches found.")
            except Exception as e:
                logger.error(f"Error in OpenAlex search: {e}")
                logs.append("OpenAlex search failed.")
            return logs, metadata

    async def validate_author(
        self,
        session: aiohttp.ClientSession,
        author_name: str,
        affiliation: Optional[str] = None,
    ) -> Tuple[List[str], Optional[Dict[str, Any]]]:
        """Validate author information using OpenAlex."""
        async with self.openalex_sem:
            logs: List[str] = []
            metadata = None
            if not author_name:
                return logs, metadata

            logger.info(f"Validating author: {author_name}")

            try:
                params = {"search": author_name, "per_page": 1}
                url = f"{self.openalex_base}/authors"
                data = await self._safe_request(session, url, params=params)
                candidates = data.get("results", []) if data else []
                if candidates:
                    author = candidates[0]
                    metadata = self._format_openalex_author(author)
                    display_name = author.get("display_name", author_name)
                    logs.append(f"OpenAlex Author: {display_name}")
                    inst = metadata.get("affiliation", {}) if metadata else {}
                    if inst:
                        logs.append(
                            f"  Institution: {inst.get('name')} ({inst.get('country')})"
                        )
                else:
                    logs.append(
                        f"Author validation: No OpenAlex match for {author_name} (affiliation hint: {affiliation or 'n/a'})."
                    )
            except Exception as e:
                logger.error(f"Error validating author: {e}")
                logs.append(f"Author validation failed for {author_name}.")

            return logs, metadata

    async def validate_reference_details(
        self, title: Optional[str], authors: List[dict], doi: Optional[str] = None
    ) -> AcademicValidationResult:
        """
        Perform comprehensive validation using academic databases and return
        both logs and any structured metadata discovered.
        """
        result = AcademicValidationResult()
        if not title and not doi:
            result.logs.append(
                "Validation skipped: title or DOI required to query academic sources."
            )
            return result

        authors = authors or []
        cache_key = self._get_cache_key(title, doi, authors)
        cached = self._load_cached_validation(cache_key)
        if cached:
            return cached

        async with aiohttp.ClientSession() as session:
            # Parallelize main searches
            tasks = [
                self.search_crossref(session, title or "", doi),
                self.search_openalex(session, title or "", doi),
                self.search_semantic_scholar(session, title or "", doi),
            ]

            # Add PubMed and arXiv if title
            if title:
                if any(
                    keyword in title.lower()
                    for keyword in [
                        "medical",
                        "clinical",
                        "disease",
                        "patient",
                        "treatment",
                    ]
                ):
                    tasks.append(self.search_pubmed(session, title))
                tasks.append(self.search_arxiv(session, title))

            # Gather results
            search_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for res in search_results:
                if isinstance(res, Exception):
                    logger.error(f"Search task failed: {res}")
                    continue
                if isinstance(res, tuple):  # crossref and openalex return tuples
                    logs, meta = res
                    result.logs.extend(logs)
                    result.merge_reference_metadata(meta)
                elif isinstance(res, list):  # others return lists
                    result.logs.extend(res)

            # Parallelize author validations
            author_tasks = []
            for author in authors or []:
                first = author.get("first_name")
                last = author.get("last_name")
                name = " ".join([part for part in [first, last] if part]).strip()
                if not name:
                    continue
                aff_payload = author.get("affiliation")
                if isinstance(aff_payload, dict):
                    aff = aff_payload.get("name", "")
                else:
                    aff = aff_payload or ""
                author_tasks.append(self.validate_author(session, name, aff))

            if author_tasks:
                author_results = await asyncio.gather(
                    *author_tasks, return_exceptions=True
                )
                for res in author_results:
                    if isinstance(res, Exception):
                        logger.error(f"Author validation failed: {res}")
                        continue
                    logs, meta = res
                    result.logs.extend(logs)
                    normalized_name = normalize_author_key(first, last, fallback=name)
                    result.merge_author_metadata(normalized_name, meta or {})

        if not result.logs:
            result.logs.append("No academic validation signals were gathered.")

        self._save_cached_validation(cache_key, result)

        return result
