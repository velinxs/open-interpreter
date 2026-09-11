"""Result objects and errors the web toolbox hands back.

Every result is a dict (so it prints and serializes plainly) with attribute
access and a few conveniences on top, such as fetching one of the results by
index or searching within fetched page text.
"""

from __future__ import annotations

import json
import os
from typing import Any

from babel import Locale
from babel.core import UnknownLocaleError


def _default_locale_from_environment() -> Locale:
    """System locale via Babel (LANG/LC_*); search APIs need ISO language + territory codes."""
    try:
        return Locale.default("LC_CTYPE")
    except (UnknownLocaleError, TypeError, ValueError, OSError):
        return Locale.parse("en_US")


def _normalize_locale_language_for_hl(lang: str) -> str:
    """ISO 639-1 for hl= / search_lang."""
    if not lang:
        return "en"
    s = lang.strip()
    if len(s) == 2 and s.isalpha():
        return s.lower()
    try:
        loc = Locale.parse(s.replace("_", "-"), sep="-")
        return (loc.language or "en").lower()
    except (UnknownLocaleError, ValueError):
        return "en"


def _normalize_locale_country_for_gl(country: str) -> str:
    """ISO 3166-1 alpha-2 (uppercase) for gl= / country."""
    if not country:
        return "US"
    s = " ".join(country.replace("\u00a0", " ").split()).strip()
    if len(s) == 2 and s.isalpha():
        return s.upper()
    try:
        loc = Locale.parse(s.replace("_", "-"), sep="-")
        if loc.territory:
            return loc.territory.upper()
    except (UnknownLocaleError, ValueError):
        pass
    return "US"


class ApiKeyError(Exception):
    """Exception raised when an API key is missing. Contains error dict."""

    def __init__(self, error_dict):
        self.error_dict = error_dict
        super().__init__(error_dict.get("error", "API key missing"))

    def _render_traceback_(self):
        # Suppress the full Jupyter/IPython traceback — just show the message.
        msg = self.error_dict.get("message", str(self))
        return [f"ApiKeyError: {msg}"]


class WebToolboxError(Exception):
    """Raised when a web toolbox operation fails (missing package, no backends, API error, etc.)."""

    def _render_traceback_(self):
        # Suppress the full Jupyter/IPython traceback — just show the message.
        return [f"WebToolboxError: {self}"]


class SearchResult(dict):
    """dict subclass for web search results. Has a compact repr to avoid flooding the context window."""

    def __init__(self, data, web=None):
        super().__init__(data)
        self._web = web

    def __getattr__(self, name):
        """Allow attribute-style access for dict keys (result.results, result.backend, ...)."""
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"'SearchResult' object has no attribute '{name}'. "
                "Use attribute access (e.g. result.results). See result.keys()."
            ) from exc

    def fetch(self, index):
        """Fetch the full page for search result at the given index. Returns a FetchResult."""
        results = self.get("results", [])
        url = results[index]["url"]
        return self._web.fetch(url)

    def __repr__(self):
        backend = self.get("backend", "?")
        results = self.get("results", [])
        n = len(results)
        lines = [f"SearchResult({n} results) [backend={backend}]"]
        lines.append("  Keys: results[list of {title,url,snippet}], raw_response[dict], backend[str]")
        lines.append("  → result.results[i] | page=result.fetch(i) → page.content | page.find(term) | page.links()")
        for i, r in enumerate(results[:5]):
            title = r.get("title", "")[:70]
            url = r.get("url", "")
            domain = url.split("/")[2] if url.count("/") >= 2 else url
            snippet = r.get("snippet", "")[:120]
            lines.append(f'  {i}. "{title}" \u2014 {domain}')
            if snippet:
                lines.append(f"     {snippet}")
        if n > 5:
            lines.append(f"  ... {n - 5} more")
        return "\n".join(lines)


class FetchResult(dict):
    """dict subclass for web fetch results. Has a compact repr to avoid flooding the context window."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached = False

    def __getattr__(self, name):
        """
        Allow attribute-style access for common fields (result.content, result.title, ...).
        This keeps FetchResult ergonomic in notebooks/REPL while remaining dict-compatible.
        """
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"'FetchResult' object has no attribute '{name}'. "
                "Use attribute access (e.g. result.content). See result.keys()."
            ) from exc

    def _get_content(self):
        """Return content string. For multi-page results, concatenates all pages."""
        if "content" in self:
            return self["content"]
        return "\n\n".join(r.get("content", "") for r in self.get("results", []))

    def find(self, term, context=100, max_results=None):
        """
        Find all occurrences of term in content (case-insensitive).
        Returns a list of snippet strings, each with up to `context` chars of surrounding text.
        Pass max_results to cap the number of matches returned.
        """
        import re

        content = self._get_content()
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        snippets = []
        for m in pattern.finditer(content):
            start = max(0, m.start() - context)
            end = min(len(content), m.end() + context)
            snippets.append(content[start:end].replace("\n", " ").strip())
            if max_results is not None and len(snippets) >= max_results:
                break
        return snippets

    def links(self):
        """
        Extract hyperlinks from content.
        Returns a list of (anchor_text, url) tuples parsed from markdown [text](url) syntax.
        """
        import re

        content = self._get_content()
        return re.findall(r"\[([^\]]*)\]\((https?://[^)]+)\)", content)

    def __repr__(self):
        backend = self.get("backend", "?")
        cached_tag = " [cached]" if getattr(self, "_cached", False) else ""
        if "results" in self:
            # Multi-URL result from explicit urls=[...] kwarg (tavily only)
            results = self.get("results", [])
            n = len(results)
            lines = [f"FetchResult({n} pages) [backend={backend}]{cached_tag}"]
            lines.append("  Keys: results[list of {url,title,content}], raw_response[dict], backend[str]")
            lines.append("  → result.results[i]['content'] | result.find(term) | result.links()")
            for r in results[:3]:
                title = r.get("title", "")[:50]
                url = r.get("url", "")
                domain = url.split("/")[2] if url.count("/") >= 2 else url
                content_len = len(r.get("content", ""))
                lines.append(f'  \u2022 "{title}" \u2014 {domain} ({content_len:,} chars)')
            if n > 3:
                lines.append(f"  ... {n - 3} more")
        else:
            # Single-page result (all backends for single-URL fetch)
            title = self.get("title", "")
            content = self.get("content", "")
            content_len = len(content)
            preview = content[:150].replace("\n", " ") if content else ""
            extra_keys = ", ".join(f"{k}[dict]" for k in self.keys() if k not in ("title", "url", "content", "backend"))
            lines = [f"FetchResult [backend={backend}]{cached_tag}"]
            lines.append(
                f"  Keys: url[str], title[str], content[str={content_len:,} chars]"
                + (f", {extra_keys}" if extra_keys else "")
                + ", backend[str]"
            )
            lines.append("  → result.content | result.find(term) | result.links()")
            if title:
                lines.append(f'  "{title}"')
            else:
                lines.append("  [no title]")
            if preview:
                lines.append(f"  {preview}...")
        return "\n".join(lines)


class AnswerResult(dict):
    """dict subclass for web answer results. Has a compact repr to avoid flooding the context window."""

    def __init__(self, data, web=None):
        super().__init__(data)
        self._web = web

    def __getattr__(self, name):
        """Allow attribute-style access for dict keys (result.answer, result.sources, ...)."""
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"'AnswerResult' object has no attribute '{name}'. "
                "Use attribute access (e.g. result.sources). See result.keys()."
            ) from exc

    def fetch(self, index):
        """Fetch the full page for source at the given index. Returns a FetchResult."""
        sources = self.get("sources", [])
        url = sources[index]["url"]
        return self._web.fetch(url)

    def __repr__(self):
        backend = self.get("backend", "?")
        answer = self.get("answer", "")
        sources = self.get("sources", [])
        n_sources = len(sources)
        lines = [f"AnswerResult({n_sources} sources) [backend={backend}]"]
        lines.append("  Keys: answer[str], sources[list of {title,url,snippet}], backend[str]")
        lines.append("  → result.answer | page=result.fetch(i) → page.content | page.find(term) | page.links()")
        if answer:
            for line in answer.split("\n"):
                lines.append(f"  {line}")
        return "\n".join(lines)


class StructuredOutputResult(dict):
    """dict subclass for web search results with structured output (JSON)."""

    def __init__(self, data, web=None):
        super().__init__(data)
        self._web = web

    def __getattr__(self, name):
        """Allow attribute-style access for dict keys (result.structured_output, result.sources, ...)."""
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"'StructuredOutputResult' object has no attribute '{name}'. "
                "Use attribute access (e.g. result.structured_output). See result.keys()."
            ) from exc

    def fetch(self, index):
        """Fetch the full page for source at the given index. Returns a FetchResult."""
        sources = self.get("sources", [])
        if not sources:
            raise WebToolboxError("No sources available in this result to fetch.")
        url = sources[index]["url"]
        return self._web.fetch(url)

    def __repr__(self):
        backend = self.get("backend", "?")
        data = self.get("structured_output", {})
        # Show some of the fields to be helpful but not flood repr
        keys = list(data.keys()) if isinstance(data, dict) else []
        lines = [f"StructuredOutputResult [backend={backend}]"]
        lines.append("  Fields: .structured_output, .sources, .backend")
        sk = ", ".join(keys[:10]) + ("..." if len(keys) > 10 else "")
        lines.append(f"  Keys inside .structured_output: {sk or '(empty)'}")
        lines.append(
            "  → result.structured_output | page=result.fetch(i) → page.content | page.find(term) | page.links()"
        )
        # Pretty print a bit of JSON as preview — use 2-space indent, max 6 lines
        try:
            preview = json.dumps(data, indent=2)
            for line in preview.splitlines()[:6]:
                lines.append(f"  {line}")
            if len(preview.splitlines()) > 6:
                lines.append("  ...")
        except (TypeError, ValueError):
            lines.append(f"  {str(data)[:200]}...")
        return "\n".join(lines)


def _normalize_tavily_single_page(result):
    """
    Tavily's extract API always returns a list, even for a single URL.
    Unwrap it to the flat {url, title, content, raw_response} structure
    that serper and linkup return, so FetchResult is consistent across backends.
    Only used for single-URL fetches; multi-URL calls keep the list structure.
    """
    pages = result.get("results", [])
    if not pages:
        raise WebToolboxError(
            "Tavily returned no results for this URL. The page may be inaccessible or blocked. Try a different backend."
        )
    flat = pages[0].copy()
    flat["raw_response"] = result.get("raw_response", {})
    return flat
