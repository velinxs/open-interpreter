"""Linkup search, answer, structured output and fetch."""

import json
import os

import requests

from ..results import ApiKeyError, WebToolboxError


class LinkupBackend:
    def _search_linkup(self, query, depth="standard", country_code=None, language_code=None, **kwargs):
        """
        Search using LinkUp backend (searchResults mode).

        Args:
            query (str): The search query
            depth (str): "standard" or "deep" (default: "standard")
            country_code (str): 2-letter country code (not directly used by LinkUp, but kept for consistency)
            language_code (str): 2-letter language code (not directly used by LinkUp, but kept for consistency)
            **kwargs: Additional LinkUp search parameters

        Returns:
            Normalized dict with "results" key
        """
        # LinkUp doesn't use country/language codes directly, but we accept them for consistency
        try:
            from linkup import LinkupClient
        except ImportError:
            self._handle_import_error("linkup-sdk", "pip install linkup-sdk")

        try:
            api_key = self._check_api_key("LINKUP_API_KEY")
        except ApiKeyError as e:
            raise WebToolboxError(e.error_dict["message"]) from e

        try:
            client = LinkupClient(api_key=api_key)

            # Build search parameters
            search_params = {
                "query": query,
                "depth": depth,
                "output_type": "searchResults",  # Just search results, no AI answer
                **kwargs,
            }

            response = client.search(**search_params)
        except Exception as e:
            self._handle_api_request_error("LinkUp", e)

        # Normalize LinkUp response format
        # LinkUp returns a LinkupSearchResults object with .results attribute
        normalized = self._create_normalized_response(response)

        # Extract results - check if it's a list of objects or dicts
        results = getattr(response, "results", [])
        if not isinstance(results, list):
            raise ValueError(
                f"LinkUp response 'results' is not a list: {type(results).__name__}. "
                f"Response type: {type(response).__name__}"
            )

        for result in results:
            normalized["results"].append(self._normalize_result_item(result))

        return normalized

    def _answer_linkup(self, question, depth="standard", **kwargs):
        """
        Get AI-generated answer using LinkUp backend.

        Args:
            question: The question to answer
            depth: "standard" or "deep" (default: "standard")
            **kwargs: Additional LinkUp search parameters

        Returns:
            Normalized dict with "answer" and "sources" keys
        """
        try:
            from linkup import LinkupClient
        except ImportError:
            self._handle_import_error("linkup-sdk", "pip install linkup-sdk")

        try:
            api_key = self._check_api_key("LINKUP_API_KEY")
        except ApiKeyError as e:
            raise WebToolboxError(e.error_dict["message"]) from e

        try:
            client = LinkupClient(api_key=api_key)

            # Build search parameters
            search_params = {"query": question, "depth": depth, "output_type": "sourcedAnswer", **kwargs}

            response = client.search(**search_params)
        except Exception as e:
            # API request failed (network error, rate limit, etc.) - return error dict for fallback
            self._handle_api_request_error("LinkUp", e)

        # LinkUp returns a LinkupSourcedAnswer object, not a dict
        # Access attributes directly: response.answer, response.sources
        normalized = {"answer": getattr(response, "answer", ""), "sources": []}

        # Extract sources - check if it's a list of objects or dicts
        sources = getattr(response, "sources", [])
        if not isinstance(sources, list):
            raise ValueError(
                f"LinkUp response 'sources' is not a list: {type(sources).__name__}. "
                f"Response type: {type(response).__name__}, "
                f"Response attributes: {dir(response)}"
            )

        for source in sources:
            # Use the same normalization as search results
            normalized["sources"].append(self._normalize_result_item(source))

        return normalized

    def _structured_output_linkup(self, query, structured_output_schema, depth="standard", **kwargs):
        """
        Get JSON structured output using LinkUp backend.

        Args:
            query: The search query
            structured_output_schema: dict representing JSON schema
            depth: "standard" or "deep" (default: "standard")
            **kwargs: Additional LinkUp search parameters

        Returns:
            Normalized dict with "structured_output" and optionally "sources"
        """
        try:
            from linkup import LinkupClient
        except ImportError:
            self._handle_import_error("linkup-sdk", "pip install linkup-sdk")

        try:
            api_key = self._check_api_key("LINKUP_API_KEY")
        except ApiKeyError as e:
            raise WebToolboxError(e.error_dict["message"]) from e

        try:
            client = LinkupClient(api_key=api_key)

            # Build search parameters
            search_params = {
                "query": query,
                "depth": depth,
                "output_type": "structured",
                "structured_output_schema": structured_output_schema,
                **kwargs,
            }

            response = client.search(**search_params)
        except Exception as e:
            self._handle_api_request_error("LinkUp", e)

        # LinkUp returns a LinkupStructuredOutput object with .structured_output attribute
        # and .sources attribute.
        structured_data = getattr(response, "structured_output", response)

        # If response was a dict and had specific keys (e.g. if SDK version changes)
        if isinstance(structured_data, dict) and "structured_output" in structured_data:
            structured_data = structured_data["structured_output"]

        normalized = {"structured_output": structured_data, "sources": []}

        # Check if sources are available
        sources = getattr(response, "sources", [])
        if sources and isinstance(sources, list):
            for source in sources:
                normalized["sources"].append(self._normalize_result_item(source))

        return normalized

    def _fetch_linkup(self, url, render_js=False, **kwargs):
        """
        Fetch web page content using LinkUp backend.

        Args:
            url (str): The URL to fetch
            render_js (bool): Whether to render JavaScript (default: False)
            **kwargs: Additional LinkUp fetch parameters

        Returns:
            Normalized dict with "content" (markdown), "title", "url" keys
        """
        try:
            from linkup import LinkupClient
        except ImportError:
            self._handle_import_error("linkup-sdk", "pip install linkup-sdk")

        try:
            api_key = self._check_api_key("LINKUP_API_KEY")
        except ApiKeyError as e:
            raise WebToolboxError(e.error_dict["message"]) from e

        try:
            client = LinkupClient(api_key=api_key)

            # Build fetch parameters
            # LinkUp fetch() returns markdown by default, no output_format parameter needed
            fetch_params = {"url": url, "render_js": render_js, **kwargs}

            response = client.fetch(**fetch_params)
        except Exception as e:
            self._handle_api_request_error("LinkUp", e)

        # LinkUp returns a LinkupFetchResult object or dict
        # LinkUp fetch endpoint does NOT provide title as a separate field
        # Always extract markdown content (prefer markdown, fallback to html if needed)
        if hasattr(response, "markdown"):
            content = getattr(response, "markdown", "") or getattr(response, "html", "")
        elif isinstance(response, dict):
            content = response.get("markdown", "") or response.get("html", "")
        else:
            content = str(response)

        # Title is not provided by LinkUp backend
        title = ""

        normalized = {
            "url": url,
            "title": title,
            "content": content,  # Always markdown (normalized)
            "raw_response": response,
        }

        return normalized
