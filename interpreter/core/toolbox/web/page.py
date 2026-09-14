"""Reading one page, whole or in part.

`fetch` brings back the entire page as markdown. `search_page` returns only
the passages matching a query, which is usually what was wanted and a small
fraction of the tokens: a backend that can extract by query does it at the
source, and otherwise the page is fetched once and matched locally.
"""

from __future__ import annotations

from .results import (
    ApiKeyError,
    FetchResult,
    PageSearchResult,
    ResultItem,
    WebToolboxError,
    _normalize_fetch_url,
    _normalize_tavily_single_page,
)


class PageMixin:
    def fetch(
        self, url: str, backend: str | None = None, render_js: bool = False, extract_depth: str | None = None, **kwargs
    ) -> FetchResult:
        """
        Fetch a whole web page as markdown. Prefer search_page() for one detail; it costs far fewer tokens.

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
            FetchResult: .url, .title, .content, .backend (single URL; multi-URL .results items: .title or ['title'])

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
        # Normalize first, so a malformed URL is named as such before any
        # backend sees it, and so "example.com" shares a cache entry with
        # "https://example.com" instead of fetching the page twice.
        url = _normalize_fetch_url(url)
        if "urls" in kwargs:
            kwargs["urls"] = [_normalize_fetch_url(u) for u in kwargs["urls"]]

        # Define backend methods
        backend_methods = {
            "serper": self._fetch_serper,
            "linkup": self._fetch_linkup,
            "tavily": self._fetch_tavily,
            # Keyless, and therefore always available. Last, because the keyed
            # backends strip boilerplate and render JavaScript; this one is the
            # floor that keeps fetch usable on a fresh install.
            "direct": self._fetch_direct,
        }

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
                if backend != "tavily":
                    raise WebToolboxError(
                        f"Backend '{backend}' does not support multi-URL fetch (urls=[...]). "
                        "Use backend='tavily' for multi-URL fetch."
                    )
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

        # "direct" needs no key, so auto-selection always ends somewhere that works.
        backends_to_try = ["serper", "linkup", "tavily", "direct"]
        failed_results = []

        for backend_name in backends_to_try:
            if not self._check_backend_available(backend_name):
                continue
            if is_multi_url and backend_name != "tavily":
                # Only tavily fetches several URLs in one call; the rest would
                # be handed a list where they expect one URL.
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

    def _search_page_via_fetch(self, fetch_backend, url, query, max_results=5, context_chars=500):
        """
        Emulate within-page search for backends with no native support (serper,
        linkup, direct): fetch the full page, then match locally. Scores are
        None (unranked, document order).
        """
        page = self.fetch(url, backend=fetch_backend)
        snippets = page.find(query, context=context_chars, max_results=max_results)
        return {
            "url": url,
            "query": query,
            "matches": [ResultItem({"heading": None, "snippet": s, "score": None}) for s in snippets],
            "raw_response": {"emulated_via_fetch": page.get("backend", fetch_backend)},
        }

    def search_page(
        self, url: str, query: str, backend: str | None = None, max_results: int = 5, context_chars: int = 500, **kwargs
    ) -> PageSearchResult:
        """
        Search within a single page for passages matching a query. PREFERRED over fetch() when you only need a detail.

        This method automatically selects the best available backend or uses
        the specified one: tavily extracts by query at the source (semantic,
        so it also matches paraphrases); every other backend emulates the
        search by fetching the page once and matching locally, unranked, in
        document order. `.backend` names the backend that produced the text.

        Args:
            url (str): The URL of the page to search in
            query (str): Space-separated search terms (case-insensitive), e.g. "pricing plans"
            backend (str, optional): Force a specific backend ("tavily", "serper", "linkup", or "direct").
                                     All but "tavily" emulate the search via a full-page fetch.
                                     If None, auto-selects.
            max_results (int): Maximum passages to return (default: 5).
                               For tavily this maps to chunks_per_source unless overridden in kwargs.
            context_chars (int): Per-passage character budget (default: 500). Used as the local
                                 context window for emulated backends.
            **kwargs: Additional backend-specific parameters:
                TAVILY:
                    - chunks_per_source (int): Max chunks per source (defaults to max_results)
                    - extract_depth (str): "basic" or "advanced"

        Returns:
            PageSearchResult: .url, .query, .matches (items: .snippet or ['snippet']), .backend (use attribute access)

        Examples:
            # Find a detail without reading the whole page (auto-selects backend)
            result = toolbox.web.search_page("https://example.com/docs", "rate limits")
            for match in result.matches:
                print(match["snippet"])

            # Full page, once a passage looks promising
            page = result.fetch()
            print(page.content[:500])
        """
        url = _normalize_fetch_url(url)
        native_methods = {"tavily": self._search_page_tavily}
        emulated_backends = ("serper", "linkup", "direct")
        hint = "→ result.matches[i]['snippet'] | page=result.fetch() → page.content"

        if backend:
            backend = backend.lower()
            if backend in native_methods:
                result = native_methods[backend](url, query, max_results=max_results, **kwargs)
            elif backend in emulated_backends:
                result = self._search_page_via_fetch(
                    backend, url, query, max_results=max_results, context_chars=context_chars
                )
            else:
                raise WebToolboxError(
                    f"Supported backends for search_page: {', '.join(list(native_methods) + list(emulated_backends))}. "
                    "Try without specifying a backend to auto-select."
                )
            result["backend"] = backend
            print(hint)
            return PageSearchResult(result, web=self)

        # Auto-select: a backend that extracts by query first, since it sends
        # back only the relevant passages.
        backends_to_try = ["tavily"]
        failed_results = []

        for backend_name in backends_to_try:
            if not self._check_backend_available(backend_name):
                continue
            try:
                result = native_methods[backend_name](url, query, max_results=max_results, **kwargs)
                result["backend"] = backend_name
                print(hint)
                return PageSearchResult(result, web=self)
            except (WebToolboxError, ApiKeyError) as e:
                failed_results.append((backend_name, e))

        # Otherwise emulate through fetch()'s own auto-selection, which ends at
        # the keyless `direct` backend — so this path always has somewhere to
        # go. The label names the fetch backend actually used (feedable back
        # into backend=) and raw_response records that it was emulated.
        try:
            result = self._search_page_via_fetch(None, url, query, max_results=max_results, context_chars=context_chars)
            result["backend"] = result.get("raw_response", {}).get("emulated_via_fetch") or "fetch"
            print(hint)
            return PageSearchResult(result, web=self)
        except (WebToolboxError, ApiKeyError) as e:
            failed_results.append(("fetch", e))

        # Report the backends that were actually tried and what each said; the
        # fetch error carries its own per-backend detail.
        reasons = ". ".join(f"{name}: {' '.join(str(exc).splitlines())}" for name, exc in failed_results)
        raise WebToolboxError(f"No page search backends are working. {reasons}")
