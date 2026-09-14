"""Reading one page.

`fetch` brings back a whole page as markdown. It is the sibling of the search
frontends in web.py and lives here because that module is at its line budget;
keeping the page-reading frontend together also gives the within-page search
that follows it somewhere obvious to live.
"""

from __future__ import annotations

from .results import (
    ApiKeyError,
    FetchResult,
    WebToolboxError,
    _normalize_fetch_url,
    _normalize_tavily_single_page,
)


class PageMixin:
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
