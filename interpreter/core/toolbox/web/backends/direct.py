"""Fetching a page with no API key, using the libraries already installed.

Every other backend needs a key from a commercial service, so on a fresh
install `toolbox.web.fetch("https://example.com")` — the most obvious thing to
try — failed with a wall of signup links. Reading a URL the user just named
does not need a search vendor; it needs an HTTP request and an HTML-to-markdown
pass, and `requests` and `html2text` are already dependencies.

This is the last backend tried. The keyed ones do more (JavaScript rendering,
boilerplate stripping, multi-page extraction) and are worth preferring when a
key exists. This one is the floor, so the toolbox is never simply unusable.
"""

import re

from ..results import WebToolboxError

# Long enough for a slow site, short enough that a stalled connection does not
# sit there. Every keyed backend's SDK client sets no timeout at all, which is
# how a fetch could hang until the idle timeout killed the whole block.
DEFAULT_TIMEOUT = 30

# A default python-requests User-Agent is refused by a lot of sites outright.
_USER_AGENT = "Mozilla/5.0 (compatible; OpenInterpreter/1.0; +https://openinterpreter.com)"

_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_STRIP_TAGS = re.compile(r"<(script|style|noscript)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)


def _to_markdown(html):
    """HTML as markdown, falling back to the raw text if html2text is missing."""
    try:
        import html2text
    except ImportError:
        return html

    converter = html2text.HTML2Text()
    converter.ignore_images = True
    converter.body_width = 0  # no hard wrapping; it breaks code and long URLs
    return converter.handle(html)


class DirectMixin:
    def _fetch_direct(self, url, timeout=DEFAULT_TIMEOUT, **kwargs):
        """Fetch one URL over plain HTTP and return it as markdown."""
        try:
            import requests
        except ImportError as error:  # requests is a hard dependency; be explicit anyway
            raise WebToolboxError("Fetching without an API key needs `requests` installed.") from error

        if not isinstance(url, str) or not url.strip():
            raise WebToolboxError("fetch needs a URL.")
        if not url.startswith(("http://", "https://")):
            raise WebToolboxError(f"fetch needs an http:// or https:// URL, got: {url!r}")

        try:
            response = requests.get(
                url,
                headers={"User-Agent": _USER_AGENT},
                timeout=timeout,
                allow_redirects=True,
            )
            response.raise_for_status()
        except Exception as error:
            raise WebToolboxError(f"Could not fetch {url}: {type(error).__name__}: {error}") from error

        content_type = (response.headers.get("Content-Type") or "").lower()
        if "html" in content_type or not content_type:
            html = response.text
            title_match = _TITLE.search(html)
            title = title_match.group(1).strip() if title_match else url
            content = _to_markdown(_STRIP_TAGS.sub("", html))
        else:
            # Plain text, JSON, CSV and friends are already what the caller wants.
            title = url
            content = response.text

        return {
            "url": response.url,
            "title": " ".join(title.split()),
            "content": content,
        }
