"""Brave Search API."""

import json
import os

import requests

from ..results import ApiKeyError, WebToolboxError


class BraveBackend:
    def _search_brave(self, query, count=10, country_code=None, language_code=None, safesearch="moderate", **kwargs):
        """
        Search using Brave Search API backend.

        Args:
            query (str): The search query
            count (int): Number of results to return (default: 10, max: 20)
            country_code (str): 2-letter country code for localized results (e.g., "US", "GB", default: system locale)
            language_code (str): 2-letter language code for search (e.g., "en", "es", default: system locale)
            safesearch (str): Safe search level: "off", "moderate", or "strict" (default: "moderate")
            **kwargs: Additional Brave-specific parameters (freshness, text_decorations, spellcheck, etc.)

        Returns:
            Normalized dict with "results" key
        """
        country_code, language_code = self._get_locale_defaults(country_code, language_code, country_case="upper")

        try:
            api_key = self._check_api_key("BRAVE_API_KEY")
        except ApiKeyError as e:
            raise WebToolboxError(e.error_dict["message"]) from e

        url = "https://api.search.brave.com/res/v1/web/search"

        headers = {"Accept": "application/json", "Accept-Encoding": "gzip", "X-Subscription-Token": api_key}

        # kwargs last would let stray keys (e.g. country=...) override normalized locale — merge first.
        params = {
            **kwargs,
            "q": query,
            "count": min(count, 20),  # Brave has a max of 20
            "country": country_code,
            "search_lang": language_code,
            "safesearch": safesearch,
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            self._handle_api_request_error("Brave Search", e)

        # Normalize Brave response format
        # Brave returns: {"web": {"results": [...]}, "news": {...}, ...}
        normalized = self._create_normalized_response(data)

        # Extract web results
        web_results = data.get("web", {}).get("results", [])
        for result in web_results:
            normalized["results"].append(self._normalize_result_item(result))

        return normalized
