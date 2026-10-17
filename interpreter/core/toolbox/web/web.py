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
import re
from typing import Any, Optional

import requests
from babel import Locale
from babel.core import UnknownLocaleError

from .backends import (
    BackendPlumbing,
    BraveBackend,
    DirectMixin,
    LinkupBackend,
    SerpApiBackend,
    SerperBackend,
    TavilyBackend,
)
from .backends.linkup import _normalize_structured_schema
from .page import PageMixin
from .results import (
    AnswerResult,
    ApiKeyError,
    FetchResult,
    PageSearchResult,
    ResultItem,
    SearchResult,
    StructuredOutputResult,
    WebToolboxError,
    _default_locale_from_environment,
    _normalize_locale_country_for_gl,
    _normalize_locale_language_for_hl,
)

__all__ = [
    "AnswerResult",
    "ApiKeyError",
    "FetchResult",
    "PageSearchResult",
    "ResultItem",
    "SearchResult",
    "StructuredOutputResult",
    "Web",
    "WebToolboxError",
]


_ISO_3166_1_ALPHA2 = frozenset(
    """
    AD AE AF AG AI AL AM AO AQ AR AS AT AU AW AX AZ BA BB BD BE BF BG BH BI BJ BL
    BM BN BO BQ BR BS BT BV BW BY BZ CA CC CD CF CG CH CI CK CL CM CN CO CR CU
    CV CW CX CY CZ DE DJ DK DM DO DZ EC EE EG EH ER ES ET FI FJ FK FM FO FR GA
    GB GD GE GF GG GH GI GL GM GN GP GQ GR GS GT GU GW GY HK HM HN HR HT HU ID
    IE IL IM IN IO IQ IR IS IT JE JM JO JP KE KG KH KI KM KN KP KR KW KY KZ LA
    LB LC LI LK LR LS LT LU LV LY MA MC MD ME MF MG MH MK ML MM MN MO MP MQ MR
    MS MT MU MV MW MX MY MZ NA NC NE NF NG NI NL NO NP NR NU NZ OM PA PE PF PG
    PH PK PL PM PN PR PS PT PW PY QA RE RO RS RU RW SA SB SC SD SE SG SH SI SJ
    SK SL SM SN SO SR SS ST SV SX SY SZ TC TD TF TG TH TJ TK TL TM TN TO TR TT
    TV TW TZ UA UG UM US UY UZ VA VC VE VG VI VN VU WF WS YE YT ZA ZM ZW
    """.split()
)
_ISO_639_1 = frozenset(
    """
    aa ab ae af ak am an ar as av ay az ba be bg bh bi bm bn bo br bs ca ce ch co
    cr cs cu cv cy da de dv dz ee el en eo es et eu fa ff fi fj fo fr fy ga gd gl
    gn gu gv ha he hi ho hr ht hu hy hz ia id ie ig ii ik io is it iu ja jv ka kg
    ki kj kk kl km kn ko kr ks ku kv kw ky la lb lg li ln lo lt lu lv mg mh mi mk
    ml mn mr ms mt my na nb nd ne ng nl nn no nr nv ny oc oj om or os pa pi pl ps
    pt qu rm rn ro ru rw sa sc sd se sg si sk sl sm sn so sq sr ss st su sv sw ta
    te tg th ti tk tl tn to tr ts tt tw ty ug uk ur uz ve vi vo wa wo xh yi yo za
    zh zu
    """.split()
)


def _reject_bad_country_code(country):
    """Raise if a country code is not a real ISO 3166-1 alpha-2 code.

    Called before any network request, so an unusable code is reported as a bad
    argument rather than as a backend failure the agent cannot act on.
    """
    if not country:
        return
    code = " ".join(str(country).replace("\u00a0", " ").split()).strip()
    if not code:
        return
    # Accept a full locale like "en_GB" and validate only the territory part.
    parts = re.split(r"[_-]", code)
    candidate = parts[-1].upper() if len(parts[-1]) == 2 else code.upper()
    if candidate in _ISO_3166_1_ALPHA2:
        return
    if len(candidate) == 2:
        # Any two-character candidate gets a near-miss suggestion, letters or
        # not: a digit slipped into a code ("U5" for "UA") is the commonest
        # typo and the closest match is the most useful thing to say about it.
        suggestion = _closest_country_code(candidate)
        detail = (
            f" (did you mean {suggestion!r}?)" if suggestion else ""
        )
        if candidate.isalpha():
            extra = " Do not pass a name or a language code here."
        else:
            extra = " Country codes are two letters, not digits."
        raise WebToolboxError(
            f"country_code {country!r} is not a valid ISO 3166-1 alpha-2 country "
            f"code. Use a two-letter code such as 'US', 'GB' or 'FR'{detail}."
            + extra
        )
    raise WebToolboxError(
        f"country_code {country!r} is not a valid ISO 3166-1 alpha-2 country code. "
        "Use a two-letter code such as 'US', 'GB' or 'FR', or omit it to use the "
        "system locale."
    )


def _closest_country_code(code):
    """A real country differing from `code` by one character, if one exists."""
    for candidate in sorted(_ISO_3166_1_ALPHA2):
        if sum(a != b for a, b in zip(code.upper(), candidate)) == 1:
            return candidate
    return None


def _reject_bad_language_code(language):
    """Raise if a language code is not a real ISO 639-1 code."""
    if not language:
        return
    code = " ".join(str(language).replace("\u00a0", " ").split()).strip()
    if not code:
        return
    parts = re.split(r"[_-]", code)
    candidate = parts[0].lower() if len(parts[0]) == 2 else code.lower()
    if candidate in _ISO_639_1:
        return
    if len(candidate) == 2 and candidate.isalpha():
        raise WebToolboxError(
            f"language_code {language!r} is not a valid ISO 639-1 language code. "
            "Use a two-letter code such as 'en', 'es' or 'fr', or omit it to use "
            "the system locale."
        )
    raise WebToolboxError(
        f"language_code {language!r} is not a valid ISO 639-1 language code. Use a "
        "two-letter code such as 'en', 'es' or 'fr'."
    )


def _reject_empty_query(query, method="search"):
    """Raise a clear error for a blank query, before spending a request.

    An empty query was passed through to the backend, which answered 400 and the
    failure was reported as an API-key or connectivity problem -- sending the
    agent to fix credentials when the fault was its own argument.
    """
    if query is None:
        raise WebToolboxError(f"{method}: query is required")
    if not str(query).strip():
        raise WebToolboxError(
            f"{method}: query is empty. Provide a non-empty search query."
        )


class Web(
    BackendPlumbing,
    PageMixin,
    BraveBackend,
    DirectMixin,
    SerperBackend,
    SerpApiBackend,
    TavilyBackend,
    LinkupBackend,
):
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
            SearchResult: .results (items: .title or ['title']; content→snippet), .raw_response, .backend

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
        _reject_empty_query(query, "search")
        _reject_bad_country_code(country_code)
        _reject_bad_language_code(language_code)
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
            print("→ result.results[i] | detail=result.search_page(i, query) | page=result.fetch(i) → page.find(term)")
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
                print(
                    "→ result.results[i] | detail=result.search_page(i, query) | page=result.fetch(i) → page.find(term)"
                )
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
            AnswerResult: .answer, .sources (items: .title or ['title']; content→snippet), .backend

        Example:
            result = toolbox.web.answer("What is the latitude of Lilongwe in decimal format?")
            print(result.answer)
            for source in result.sources:
                print(f"- {source['title']}: {source['url']}")
        """
        _reject_empty_query(question, "answer")
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
            print("→ result.answer | detail=result.search_page(i, query) | page=result.fetch(i) → page.find(term)")
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
                print("→ result.answer | detail=result.search_page(i, query) | page=result.fetch(i) → page.find(term)")
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
        Search and extract specific fields (simple field map, JSON schema, or Pydantic). PREFERRED for data extraction.

        This method is best for tasks requiring extracting specific fields (like author, year, title)
        directly from web resources into a schema-defined format.

        Args:
            query (str): The search query or data extraction prompt.
            schema: What to extract — a simple field map ({"name": "string", "founded": "integer"};
                all fields required; types: string, integer, number, boolean, array, object, null),
                a full JSON schema dict, a JSON string, or a Pydantic model class.
                A dict with a "properties" mapping is treated as a full schema.
            backend (str, optional): Force a specific backend (default: "linkup").
            **kwargs: Additional backend-specific parameters:
                - For linkup: depth ("standard" or "deep"), etc.

        Returns:
            StructuredOutputResult: .structured_output, .sources (items: .title or ['title']), .backend

        Examples:
            # Simple field map (converted to a schema with all fields required)
            result = toolbox.web.structured_output(
                "Apple Inc",
                schema={"name": "string", "founded": "integer", "headquarters": "string"},
            )
            print(result.structured_output["name"])

            # Full JSON schema, e.g. for a journal article
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
        _reject_empty_query(query, "structured_output")
        # Field maps become full JSON schemas here, so a bad one fails with a
        # local message instead of a backend 400. Full schemas, Pydantic
        # classes and JSON strings pass through; the backend encodes them.
        schema = _normalize_structured_schema(schema)

        if backend:
            backend = backend.lower()
            if backend != "linkup":
                raise WebToolboxError("Only LinkUp currently supports structured output via backend='linkup'.")
            backend_methods = {"linkup": self._structured_output_linkup}
            result = backend_methods[backend](query, schema, **kwargs)
            result["backend"] = backend
            print(
                "→ result.structured_output | detail=result.search_page(i, query) | page=result.fetch(i) → page.find(term)"
            )
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
                    "→ result.structured_output | detail=result.search_page(i, query) | page=result.fetch(i) → page.find(term)"
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
