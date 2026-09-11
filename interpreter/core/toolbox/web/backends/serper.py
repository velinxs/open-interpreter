"""Serper.dev (Google SERP proxy)."""

import json
import os

import requests

from ..results import ApiKeyError, WebToolboxError


class SerperBackend:
    def _search_serper(
        self, query, num=10, type="search", country_code=None, language_code=None, autocorrect=True, **kwargs
    ):
        """
        Search using Serper API (Google search) backend.

        Supports multiple search types: search, images, videos, news, shopping, places, maps, patents.

        Args:
            query (str): The search query
            num (int): Number of results to return (default: 10)
            type (str): Search type (default: "search")
                       Supported: "search", "images", "videos", "news", "shopping", "places", "maps", "patents"
                       NOTE: "scholar" is NOT supported by Serper - use serpapi with engine="google_scholar" instead
            country_code (str): 2-letter country code for localized results (e.g., "us", "gb", default: system locale)
            language_code (str): 2-letter language code for interface (e.g., "en", "es", default: system locale)
            autocorrect (bool): Whether to autocorrect the query (default: True)
            **kwargs: Additional Serper-specific parameters (location, page, tbs, etc.)

        Returns:
            Normalized dict with "results" key
        """
        # Validate search type
        supported_types = ["search", "images", "videos", "news", "shopping", "places", "maps", "patents"]
        if type not in supported_types:
            raise WebToolboxError(
                f"Serper backend supports: {', '.join(supported_types)}. "
                f"For Google Scholar, use backend='serpapi' with engine='google_scholar'"
            )

        country_code, language_code = self._get_locale_defaults(country_code, language_code)

        try:
            api_key = self._check_api_key("SERPER_API_KEY")
        except ApiKeyError as e:
            raise WebToolboxError(e.error_dict["message"]) from e

        # Map type to correct endpoint
        url = f"https://google.serper.dev/{type}"

        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

        payload = {
            **kwargs,
            "q": query,
            "num": num,
            "gl": country_code,
            "hl": language_code,
            "autocorrect": autocorrect,
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            self._handle_api_request_error("Serper", e)

        # Normalize Serper response format
        # Response format varies by search type
        normalized = self._create_normalized_response(data)

        # Extract results based on search type - each type has its own response structure
        serper_results_keys = {
            "search": "organic",
            "news": "news",
            "images": "images",
            "videos": "videos",
            "shopping": "shopping",
            "places": "places",
            "maps": "places",  # Maps returns places
            "patents": "organic",  # Patents returns organic-style results
        }
        results_key = serper_results_keys.get(type, "organic")

        results = data.get(results_key, [])
        for result in results:
            normalized["results"].append(self._normalize_result_item(result))

        return normalized

    def _fetch_serper(self, url, **kwargs):
        """
        Fetch web page content using Serper API (scrape.serper.dev).

        Args:
            url (str): The URL to fetch
            **kwargs: Additional Serper-specific parameters

        Returns:
            Normalized dict with "content" (markdown), "title", "url" keys
        """
        try:
            api_key = self._check_api_key("SERPER_API_KEY")
        except ApiKeyError as e:
            raise WebToolboxError(e.error_dict["message"]) from e

        # Serper scrape endpoint
        scrape_url = "https://scrape.serper.dev"

        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

        # Ensure markdown output (default for Serper)
        payload = {"url": url, "markdown": True, **kwargs}

        try:
            response = requests.post(scrape_url, headers=headers, data=json.dumps(payload), timeout=60)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            self._handle_api_request_error("Serper", e)

        # Normalize Serper response format - always return markdown
        # Serper returns: {"markdown": str, "text": str, "metadata": {"title": str}, ...}
        # Title can be in metadata.title or top-level title
        title = data.get("title", "") or data.get("metadata", {}).get("title", "")

        normalized = {
            "url": url,
            "title": title,
            "content": data.get("markdown", "") or data.get("text", ""),  # Prefer markdown, fallback to text
            "raw_response": data,
        }

        return normalized
