"""
Web search utilities.

This module provides web search related tools, with unified frontend methods
for search, fetch, answer, crawl, and structured output operations with
multiple backends.

Supported backends:
- Search: linkup, serper, serpapi, brave, tavily
- Answer: linkup, tavily
- Fetch: linkup, serper, tavily
- Crawl: tavily (not implemented yet)
- Structured output: linkup
"""

# NOTE: The first line of docstrings and their Return sections are shown to Open Interpreter in its system message, so make them very concise to avoid wasting tokens, and don't mention atypical things like error condition outputs that will confuse the AI.  Tell the AI the typical use case, and it will deal with errors when it gets to them.

from __future__ import annotations

import json
import os
from typing import Any, Optional

import requests
from babel import Locale
from babel.core import UnknownLocaleError

from .backends import BackendPlumbing, BraveBackend, LinkupBackend, SerpApiBackend, SerperBackend, TavilyBackend
from .results import (
    AnswerResult,
    ApiKeyError,
    FetchResult,
    SearchResult,
    StructuredOutputResult,
    WebToolboxError,
    _default_locale_from_environment,
    _normalize_locale_country_for_gl,
    _normalize_locale_language_for_hl,
    _normalize_tavily_single_page,
)

__all__ = [
    "AnswerResult",
    "ApiKeyError",
    "FetchResult",
    "SearchResult",
    "StructuredOutputResult",
    "Web",
    "WebToolboxError",
]


class Web(BackendPlumbing, BraveBackend, SerperBackend, SerpApiBackend, TavilyBackend, LinkupBackend):
    def __init__(self, toolbox):
        self.toolbox = toolbox
        _loc = _default_locale_from_environment()
        self._default_lang = (_loc.language or "en").lower()
        self._default_country = (_loc.territory or "US").upper()
        # Session-scoped cache: keyed by URL. Web page content doesn't change
        # mid-session, so re-fetching the same URL is always wasteful.
        self._fetch_cache: dict[str, FetchResult] = {}

    def search(
        self,
        query: str,
        backend: str | None = None,
        country_code: str | None = None,
        language_code: str | None = None,
        **kwargs,
    ) -> SearchResult:
        """
        Search the web for links and snippets.

        This method automatically selects the best available backend or uses
        the specified one. Backends are tried in order: serper, serpapi, tavily, brave, linkup.

        Args:
            query (str): The search query
            backend (str, optional): Force a specific backend ("brave", "tavily", "linkup", "serpapi", or "serper").
                                     If None, auto-selects based on availability.
            country_code (str, optional): 2-letter country code for localized results (e.g., "US", "GB", "FR").
                                          Defaults to system locale. Supported by: brave, serpapi, serper.
            language_code (str, optional): 2-letter language code for search interface (e.g., "en", "es", "fr").
                                           Defaults to system locale. Supported by: brave, serpapi, serper.
            **kwargs: Additional backend-specific parameters:

                BRAVE:
                    - count (int, 1-20): Number of results (default: 10, max: 20)
                    - safesearch (str): "off", "moderate", or "strict" (default: "moderate")
                    - freshness (str): "pd" (past day), "pw" (past week), "pm" (past month), "py" (past year)
                    - text_decorations (bool): Include text decorations in snippets (default: True)
                    - spellcheck (bool): Enable spellcheck (default: True)
                    NOTE: Use country_code and language_code parameters (not country/search_lang)

                TAVILY:
                    - max_results (int): Number of results to return (default: 10)
                    - search_depth (str): "basic" or "advanced" (default: "basic")
                    - include_domains (list): list of domains to include (e.g., ["example.com"])
                    - exclude_domains (list): list of domains to exclude
                    - include_raw_content (bool): Include full HTML content (default: False)
                    - include_images (bool): Include images in results (default: False)
                    - topic (str): "general" or "news" (default: "general")
                    - days (int): Number of days back to search (for topic="news")

                LINKUP:
                    - depth (str): "standard" or "deep" (default: "standard")
                    - output_type (str): "searchResults" (default), "sourcedAnswer", or "structured"
                    - structured_output_schema (dict): Schema for structured output mode

                SERPAPI - supports 80+ search engines:
                    - num (int): Number of results (default: 10)
                    - engine (str): Search engine to use (default: "google")
                        Common engines:
                          - "google" (default): Regular Google search
                          - "google_scholar": Academic papers, citations
                          - "bing": Microsoft Bing search
                          - "yahoo": Yahoo search
                          - "duckduckgo": DuckDuckGo search
                          - "baidu": Chinese search engine
                          - "yandex": Russian search engine
                          - "youtube": YouTube video search
                          - "google_news": Google News search
                          - "google_images": Google Images search
                          - "google_shopping": Google Shopping search
                          - "ebay": eBay product search
                          - "walmart": Walmart product search
                          - "home_depot": Home Depot search
                          - "apple_app_store": App Store search
                          - "google_play": Google Play Store search
                    - location (str): Location for localized results (e.g., "Austin, Texas")
                    - google_domain (str): Google domain (e.g., "google.com", "google.co.uk")
                    - safe (str): Safe search - "active" or "off"
                    - start (int): Pagination offset
                    - filter (str): Duplicate filter - "0" (off) or "1" (on)
                    - tbm (str): Search type - "nws" (news), "isch" (images), "vid" (videos), "shop" (shopping)
                    NOTE: Use country_code and language_code parameters (not gl/hl). Each engine has specific parameters. See https://serpapi.com/ for details.

                SERPER - supports multiple search types:
                    - num (int): Number of results (default: 10)
                    - type (str): Search type (default: "search")
                        Supported types:
                          - "search": Regular web search (default)
                          - "images": Image search
                          - "videos": Video search
                          - "places": Google Maps places search
                          - "maps": Google Maps search
                          - "news": News search
                          - "shopping": Shopping search
                          - "patents": Patent search
                        NOTE: "scholar" is NOT supported by Serper. Use serpapi with engine="google_scholar" instead.
                    - location (str): Location for localized results
                    - autocorrect (bool): Enable query autocorrection (default: True)
                    - page (int): Page number for pagination
                    - tbs (str): Time-based search (e.g., "qdr:d" for past day, "qdr:w" for past week)
                    NOTE: Use country_code and language_code parameters (not gl/hl)

        Returns:
            SearchResult: .results, .raw_response, .backend (use attribute access)

        Examples:
            # Basic search (auto-selects backend)
            results = toolbox.web.search("machine learning tutorials")
            for row in results.results:
                print(f"{row['title']}: {row['url']}")

            # Search Google Scholar for academic papers (using serpapi)
            results = toolbox.web.search(
                "quantum computing",
                backend="serpapi",
                engine="google_scholar"
            )

            # Search YouTube videos (using serpapi)
            results = toolbox.web.search(
                "python tutorial",
                backend="serpapi",
                engine="youtube"
            )

            # Search Google News (using serper)
            results = toolbox.web.search(
                "AI breakthroughs",
                backend="serper",
                type="news"
            )

            # Deep web search with specific domains (using tavily)
            results = toolbox.web.search(
                "climate change research",
                backend="tavily",
                search_depth="advanced",
                include_domains=["nature.com", "science.org"]
            )

            # Shopping search (using serpapi)
            results = toolbox.web.search(
                "laptop",
                backend="serpapi",
                engine="google_shopping"
            )
        """
        used_backend = None

        # Prepare normalized parameters for all backends
        backend_kwargs = kwargs.copy()
        backend_kwargs["country_code"] = country_code
        backend_kwargs["language_code"] = language_code

        # Define backend methods once
        backend_methods = {
            "brave": self._search_brave,
            "tavily": self._search_tavily,
            "linkup": self._search_linkup,
            "serpapi": self._search_serpapi,
            "serper": self._search_serper,
        }

        if backend:
            backend = backend.lower()

            if backend not in backend_methods:
                raise WebToolboxError(
                    f"Supported backends for search: {', '.join(backend_methods.keys())}. "
                    "Try without specifying a backend to auto-select."
                )

            result = backend_methods[backend](query, **backend_kwargs)
            result["backend"] = backend
            print("→ result.results[i] | page=result.fetch(i) → page.content | page.find(term) | page.links()")
            return SearchResult(result, web=self)

        # Auto-select backend
        # Priority order based on AI agent needs: serper (rich snippets, knowledge panels, structured data, best for AI) > serpapi (comprehensive Google results) > tavily (AI-optimized) > brave (alternative sources, fewer snippets) > linkup
        backends_to_try = ["serper", "serpapi", "tavily", "brave", "linkup"]
        failed_results = []

        for backend_name in backends_to_try:
            if not self._check_backend_available(backend_name):
                continue
            try:
                result = backend_methods[backend_name](query, **backend_kwargs)
                result["backend"] = backend_name
                print("→ result.results[i] | page=result.fetch(i) → page.content | page.find(term) | page.links()")
                return SearchResult(result, web=self)
            except (WebToolboxError, ApiKeyError) as e:
                failed_results.append((backend_name, e))

        search_backend_to_package = {
            "serper": "google-search-results (serper)",
            "serpapi": "google-search-results",
            "tavily": "tavily-python",
            "brave": "brave-search-sdk",
            "linkup": "linkup-sdk",
        }
        search_backend_to_key = {
            "serper": "SERPER_API_KEY",
            "serpapi": "SERPAPI_API_KEY",
            "tavily": "TAVILY_API_KEY",
            "brave": "BRAVE_API_KEY",
            "linkup": "LINKUP_API_KEY",
        }
        message = self._build_no_backends_error(
            backends_to_try, failed_results, search_backend_to_package, search_backend_to_key, kind="search"
        )
        raise WebToolboxError(message)

    def answer(self, question: str, backend: str | None = None, **kwargs) -> AnswerResult:
        """
        AI-synthesized answer from web sources. PREFERRED for direct questions about current events

        This method automatically selects the best available backend or uses
        the specified one. Backends are tried in order: linkup, tavily.

        Args:
            question (str): MUST be a natural-language question ending in "?", e.g. "What is the best vacuum cleaner in 2026?" — NOT search terms like "best vacuum cleaner 2026". An AI agent reads the web and answers the question for you.
            backend (str, optional): Force a specific backend ("tavily" or "linkup").
                                     If None, auto-selects based on availability.
            **kwargs: Additional backend-specific parameters:
                - For tavily: answer_mode ("basic" or "advanced"), search_depth, etc.
                - For linkup: depth ("standard" or "deep"), include_inline_citations, etc.

        Returns:
            AnswerResult: .answer, .sources, .backend (use attribute access)

        Example:
            result = toolbox.web.answer("What is the latitude of Lilongwe in decimal format?")
            print(result.answer)
            for source in result.sources:
                print(f"- {source['title']}: {source['url']}")
        """
        if "?" not in question:
            print("⚠️  web.answer() AI expects a question ending in '?', not search terms.\n")

        if backend:
            backend = backend.lower()
            if backend not in ("tavily", "linkup"):
                raise WebToolboxError(
                    "Supported backends for answer: 'tavily', 'linkup'. Try without specifying a backend to auto-select."
                )
            backend_methods = {"linkup": self._answer_linkup, "tavily": self._answer_tavily}
            result = backend_methods[backend](question, **kwargs)
            result["backend"] = backend
            print("→ result.answer | page=result.fetch(i) → page.content | page.find(term) | page.links()")
            return AnswerResult(result, web=self)

        backends_to_try = ["linkup", "tavily"]
        backend_methods = {"linkup": self._answer_linkup, "tavily": self._answer_tavily}
        failed_results = []

        for backend_name in backends_to_try:
            if not self._check_backend_available(backend_name):
                continue
            try:
                result = backend_methods[backend_name](question, **kwargs)
                result["backend"] = backend_name
                print("→ result.answer | page=result.fetch(i) → page.content | page.find(term) | page.links()")
                return AnswerResult(result, web=self)
            except (WebToolboxError, ApiKeyError) as e:
                failed_results.append((backend_name, e))

        message = self._build_no_backends_error(
            backends_to_try,
            failed_results,
            backend_to_package={"linkup": "linkup-sdk", "tavily": "tavily-python"},
            backend_to_key={"linkup": "LINKUP_API_KEY", "tavily": "TAVILY_API_KEY"},
            kind="answer",
        )
        raise WebToolboxError(message)

    def structured_output(
        self, query: str, schema: Any, backend: str | None = "linkup", **kwargs
    ) -> StructuredOutputResult:
        """
        Search and extract specific fields defined by schema (dict or Pydantic). PREFERRED for data extraction.

        This method is best for tasks requiring extracting specific fields (like author, year, title)
        directly from web resources into a schema-defined format.

        Args:
            query (str): The search query or data extraction prompt.
            schema (dict or Pydantic model): The JSON schema defining the desired output structure.
            backend (str, optional): Force a specific backend (default: "linkup").
            **kwargs: Additional backend-specific parameters:
                - For linkup: depth ("standard" or "deep"), etc.

        Returns:
            StructuredOutputResult: .structured_output, .sources, .backend (use attribute access)

        Example:
            # Using journal article schema
            schema = {
                "type": "object",
                "properties": {
                    "author_last_name": {"type": "string", "description": "Last name of the first author"},
                    "year": {"type": "integer", "description": "Year of publication"},
                    "title": {"type": "string", "description": "Full title of the article"}
                },
                "required": ["author_last_name", "year", "title"]
            }
            result = toolbox.web.structured_output("Attention is All You Need journal article", schema=schema)
            print(result.structured_output["author_last_name"])
        """
        import json

        # LinkUp SDK expects a Pydantic model CLASS or a JSON STRING or None.
        # It does NOT accept a dictionary directly.
        is_pydantic = False
        try:
            # Check if it's a Pydantic class (v1 or v2)
            if isinstance(schema, type):
                # Try to import any version of Pydantic to check inheritance
                try:
                    from pydantic import BaseModel as BM2

                    if issubclass(schema, BM2):
                        is_pydantic = True
                except ImportError:
                    pass

                if not is_pydantic:
                    try:
                        from pydantic.v1 import BaseModel as BM1

                        if issubclass(schema, BM1):
                            is_pydantic = True
                    except ImportError:
                        pass
            elif hasattr(schema, "__pydantic_model__"):  # some wrappers
                is_pydantic = True
        except Exception:
            # If any check fails, treat as non-pydantic
            pass

        if not is_pydantic and isinstance(schema, dict):
            # Convert dictionary to JSON string as expected by the LinkUp SDK
            schema = json.dumps(schema)
        # If it is a Pydantic class or already a string, pass it through to the backend.

        if backend:
            backend = backend.lower()
            if backend != "linkup":
                raise WebToolboxError("Only LinkUp currently supports structured output via backend='linkup'.")
            backend_methods = {"linkup": self._structured_output_linkup}
            result = backend_methods[backend](query, schema, **kwargs)
            result["backend"] = backend
            print("→ result.structured_output | page=result.fetch(i) → page.content | page.find(term) | page.links()")
            return StructuredOutputResult(result, web=self)

        # Default/Auto-select (currently only linkup)
        backends_to_try = ["linkup"]
        backend_methods = {"linkup": self._structured_output_linkup}
        failed_results = []

        for backend_name in backends_to_try:
            if not self._check_backend_available(backend_name):
                continue
            try:
                result = backend_methods[backend_name](query, schema, **kwargs)
                result["backend"] = backend_name
                print(
                    "→ result.structured_output | page=result.fetch(i) → page.content | page.find(term) | page.links()"
                )
                return StructuredOutputResult(result, web=self)
            except (WebToolboxError, ApiKeyError) as e:
                failed_results.append((backend_name, e))

        message = self._build_no_backends_error(
            backends_to_try,
            failed_results,
            backend_to_package={"linkup": "linkup-sdk"},
            backend_to_key={"linkup": "LINKUP_API_KEY"},
            kind="structured output",
        )
        raise WebToolboxError(message)

    def fetch(
        self, url: str, backend: str | None = None, render_js: bool = False, extract_depth: str | None = None, **kwargs
    ) -> FetchResult:
        """
        Fetch web page content from a URL as markdown.

        This method automatically selects the best available backend or uses
        the specified one. Backends are tried in order: serper, linkup, tavily.

        Args:
            url (str): The URL to fetch
            backend (str, optional): Force a specific backend ("serper", "linkup", or "tavily").
                                     If None, auto-selects based on availability.
            render_js (bool): Whether to render JavaScript (default: False). Supported by: linkup
            extract_depth (str, optional): Extraction depth - "basic" or "advanced". Supported by: tavily (defaults to API default if not specified)
            **kwargs: Additional backend-specific parameters:

                SERPER:
                    - Other Serper scrape parameters (markdown is always enabled)

                LINKUP:
                    - Other LinkUp fetch parameters (output is always markdown)

                TAVILY:
                    - urls (list): list of URLs to fetch (max 20). If provided, overrides url parameter.
                    - include_images (bool): Include images in extraction (default: False)
                    - Other Tavily extract parameters

        Returns:
            FetchResult: .url, .title, .content, .backend (single URL; multi-URL uses .results)

        Examples:
            # Basic fetch (auto-selects backend)
            result = toolbox.web.fetch("https://example.com")
            print(result.title)
            print(result.content[:500])

            # Fetch with JavaScript rendering
            result = toolbox.web.fetch(
                "https://example.com",
                render_js=True
            )

            # Fetch with advanced extraction depth
            result = toolbox.web.fetch(
                "https://example.com",
                extract_depth="advanced"
            )

            # Fetch multiple pages at once (using tavily)
            result = toolbox.web.fetch(
                "https://example.com",
                backend="tavily",
                urls=["https://example.com", "https://example.org"]
            )
        """
        # Define backend methods
        backend_methods = {"serper": self._fetch_serper, "linkup": self._fetch_linkup, "tavily": self._fetch_tavily}

        # Validate backend name before touching the cache, so an invalid backend name
        # always errors immediately rather than silently returning a stale cached result.
        if backend and backend.lower() not in backend_methods:
            raise WebToolboxError(
                f"Supported backends for fetch: {', '.join(backend_methods.keys())}. "
                "Try without specifying a backend to auto-select."
            )

        # Multi-URL calls (tavily urls=[...]) bypass the cache — too varied to key simply.
        # Explicit backend requests also bypass the cache: the caller is deliberately
        # choosing a different source and should not silently get a prior result.
        is_multi_url = "urls" in kwargs

        if not is_multi_url and not backend and url in self._fetch_cache:
            cached = self._fetch_cache[url]
            cached._cached = True
            print("→ result.content | result.find(term) | result.links()")
            return cached

        if backend:
            backend = backend.lower()

            if is_multi_url:
                result = backend_methods[backend](
                    kwargs["urls"], extract_depth=extract_depth, **{k: v for k, v in kwargs.items() if k != "urls"}
                )
            elif backend == "tavily":
                result = backend_methods[backend]([url], extract_depth=extract_depth, **kwargs)
                result = _normalize_tavily_single_page(result)
            elif backend == "linkup":
                result = backend_methods[backend](url, render_js=render_js, **kwargs)
            else:
                result = backend_methods[backend](url, **kwargs)

            result["backend"] = backend
            fetch_result = FetchResult(result)
            if not is_multi_url:
                self._fetch_cache[url] = fetch_result
            if is_multi_url:
                print("→ result.results[i]['content'] | result.find(term) | result.links()")
            else:
                print("→ result.content | result.find(term) | result.links()")
            return fetch_result

        backends_to_try = ["serper", "linkup", "tavily"]
        failed_results = []

        for backend_name in backends_to_try:
            if not self._check_backend_available(backend_name):
                continue
            try:
                if is_multi_url:
                    result = backend_methods[backend_name](
                        kwargs["urls"], extract_depth=extract_depth, **{k: v for k, v in kwargs.items() if k != "urls"}
                    )
                elif backend_name == "tavily":
                    result = backend_methods[backend_name]([url], extract_depth=extract_depth, **kwargs)
                    result = _normalize_tavily_single_page(result)
                elif backend_name == "linkup":
                    result = backend_methods[backend_name](url, render_js=render_js, **kwargs)
                else:
                    result = backend_methods[backend_name](url, **kwargs)
                result["backend"] = backend_name
                fetch_result = FetchResult(result)
                if not is_multi_url:
                    self._fetch_cache[url] = fetch_result
                if is_multi_url:
                    print("→ result.results[i]['content'] | result.find(term) | result.links()")
                else:
                    print("→ result.content | result.find(term) | result.links()")
                return fetch_result
            except (WebToolboxError, ApiKeyError) as e:
                failed_results.append((backend_name, e))

        fetch_backend_to_package = {"serper": "requests (built-in)", "linkup": "linkup-sdk", "tavily": "tavily-python"}
        fetch_backend_to_key = {"serper": "SERPER_API_KEY", "linkup": "LINKUP_API_KEY", "tavily": "TAVILY_API_KEY"}
        message = self._build_no_backends_error(
            backends_to_try, failed_results, fetch_backend_to_package, fetch_backend_to_key, kind="fetch"
        )
        raise WebToolboxError(message)
