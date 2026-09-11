"""What every backend needs from its host: keys, locale, errors, normalization.

The provider mixins call these, so they are declared here rather than assumed
to exist on whatever class the mixins are combined into.
"""

import os

from babel import Locale

from ..results import (
    ApiKeyError,
    SearchResult,
    WebToolboxError,
    _default_locale_from_environment,
    _normalize_locale_country_for_gl,
    _normalize_locale_language_for_hl,
)


class BackendPlumbing:
    def _get_locale_defaults(self, country_code=None, language_code=None, country_case="lower"):
        """
        Get locale defaults for country_code and language_code.

        Args:
            country_code: Optional country code (if None, uses default)
            language_code: Optional language code (if None, uses default)
            country_case: "lower" or "upper" for country code case (default: "lower")

        Returns:
            tuple: (country_code, language_code) with defaults applied
        """
        if country_code is None:
            country_code = self._default_country
        else:
            country_code = _normalize_locale_country_for_gl(country_code)
        if language_code is None:
            language_code = self._default_lang
        else:
            language_code = _normalize_locale_language_for_hl(language_code)
        if country_case == "lower":
            country_code = country_code.lower()
        return country_code, language_code

    def _check_api_key(self, key_name):
        """
        Check if an API key is set and return it.

        Args:
            key_name: Environment variable name (e.g., "BRAVE_API_KEY")

        Returns:
            str: API key if found

        Raises:
            ApiKeyError: If API key is missing (error_dict is in exception)
        """
        # Mapping of API key names to (backend_name, key_url)
        api_key_info = {
            "BRAVE_API_KEY": ("Brave Search API", "https://brave.com/search/api/"),
            "SERPER_API_KEY": ("Serper API", "https://serper.dev/"),
            "SERPAPI_API_KEY": ("SerpApi", "https://serpapi.com/"),
            "TAVILY_API_KEY": ("Tavily", "https://tavily.com/"),
            "LINKUP_API_KEY": ("LinkUp", "https://linkup.so/"),
        }

        api_key = os.getenv(key_name)
        if not api_key:
            backend_name, key_url = api_key_info.get(key_name, ("this backend", ""))
            url_text = f" Get your API key at {key_url}" if key_url else ""
            error_dict = {
                "error": f"{key_name} environment variable not set",
                "message": f"To use {backend_name}, set the {key_name} environment variable.{url_text}",
                "alternative": "Try using a different backend",
            }
            raise ApiKeyError(error_dict)
        return api_key

    def _handle_import_error(self, package_name, install_cmd):
        """Raise WebToolboxError when a required package is not installed."""
        raise WebToolboxError(f"Install {package_name}: {install_cmd}")

    def _handle_api_request_error(self, backend_name, error):
        """Raise WebToolboxError for API request failures."""
        raise WebToolboxError(
            f"{backend_name} API request failed: {error}. Check your API key and internet connection."
        )

    def _normalize_result_item(self, result, engine=None):
        """
        Normalize a single result item from any backend.

        Args:
            result: Result dict or object from backend
            engine: Optional engine name for engine-specific handling

        Returns:
            dict: Normalized result with "title", "url", "snippet"
        """
        # Handle dict results
        if isinstance(result, dict):
            title = result.get("title", "") or result.get("name", "") or result.get("product_title", "")
            url = result.get("link", "") or result.get("url", "") or result.get("href", "")
            snippet = result.get("snippet", "") or result.get("description", "") or result.get("content", "")

            # Engine-specific handling
            if engine == "youtube" and "link" in result:
                url = result.get("link", "")
                snippet = (
                    result.get("description", "") or f"Video by {result.get('channel', {}).get('name', 'Unknown')}"
                )
            elif engine == "google_shopping" and "price" in result:
                price = result.get("price", "")
                if price:
                    snippet = f"{snippet} - {price}".strip()

            # Truncate snippet if it's from content field (Tavily/LinkUp style)
            if "content" in result and len(snippet) > 200:
                snippet = snippet[:200]

            return {"title": title, "url": url, "snippet": snippet}

        # Handle object results (LinkUp style)
        elif hasattr(result, "name"):
            return {
                "title": getattr(result, "name", ""),
                "url": getattr(result, "url", ""),
                "snippet": (getattr(result, "content", "") or getattr(result, "snippet", ""))[:200]
                if getattr(result, "content", None) or getattr(result, "snippet", None)
                else "",
            }

        # Unknown format
        raise ValueError(f"Result item is neither dict nor object: {type(result).__name__}")

    def _create_normalized_response(self, raw_response):
        """Create a normalized response structure."""
        return {"results": [], "raw_response": raw_response}

    def _check_backend_available(self, backend: str) -> bool:
        """Check if a backend is available (has API key)."""
        backend_keys = {
            "tavily": "TAVILY_API_KEY",
            "linkup": "LINKUP_API_KEY",
            "serper": "SERPER_API_KEY",
            "brave": "BRAVE_API_KEY",
            "serpapi": "SERPAPI_API_KEY",
        }
        key_name = backend_keys.get(backend.lower())
        if not key_name:
            return False
        return bool(os.getenv(key_name))

    def _build_no_backends_error(self, backends_to_try, failed_results, backend_to_package, backend_to_key, kind):
        """
        Build error message when no backends succeeded. failed_results is a list of
        (backend_name, exception). Returns a string message for WebToolboxError.
        """
        failed_by_backend = {b: exc for b, exc in failed_results}
        reasons = []
        for b in backends_to_try:
            if not self._check_backend_available(b):
                key_name = backend_to_key.get(b, b.upper() + "_API_KEY")
                reasons.append((b, f"API key not set (set {key_name})"))
            elif b in failed_by_backend:
                exc = failed_by_backend[b]
                err_str = str(exc)
                if isinstance(exc, ApiKeyError):
                    msg = exc.error_dict.get("message", err_str)
                    reasons.append((b, "API key not set (" + msg + ")"))
                elif "not installed" in err_str or "Install " in err_str:
                    pkg = backend_to_package.get(b, b)
                    reasons.append((b, f"package not installed (pip install {pkg})"))
                else:
                    # Surface the actual backend error message so callers can see
                    # why each backend failed instead of a generic placeholder.
                    # Flatten multi-line errors to keep the aggregate message readable.
                    clean_err = " ".join(err_str.splitlines()).strip()
                    if isinstance(exc, WebToolboxError):
                        # WebToolboxError messages are already user-oriented; use as-is.
                        reasons.append((b, clean_err or "request failed"))
                    else:
                        # For other exception types, include them as a failure reason.
                        reasons.append((b, f"request failed: {clean_err}" if clean_err else "request failed"))
            else:
                reasons.append((b, "unavailable"))
        kind_label = f"{kind} " if kind else ""
        return f"No {kind_label}backends are working. " + ". ".join(f"{b}: {msg}" for b, msg in reasons)
