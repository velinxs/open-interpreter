"""Tavily search, answer and extract."""

import json
import os

import requests

from ..results import ApiKeyError, WebToolboxError


class TavilyBackend:
    def _search_tavily(self, query, max_results=10, country_code=None, language_code=None, **kwargs):
        """
        Search using Tavily backend (just search results, no AI answer).

        Args:
            query (str): The search query
            max_results (int): Maximum number of results to return (default: 10)
            country_code (str): 2-letter country code (not directly used by Tavily, but kept for consistency)
            language_code (str): 2-letter language code (not directly used by Tavily, but kept for consistency)
            **kwargs: Additional Tavily search parameters (search_depth, include_domains, exclude_domains, etc.)

        Returns:
            Normalized dict with "results" key
        """
        # Tavily doesn't use country/language codes directly, but we accept them for consistency
        try:
            from tavily import TavilyClient
        except ImportError:
            self._handle_import_error("tavily-python", "pip install tavily-python")

        try:
            api_key = self._check_api_key("TAVILY_API_KEY")
        except ApiKeyError as e:
            raise WebToolboxError(e.error_dict["message"]) from e

        try:
            client = TavilyClient(api_key=api_key)

            # Build search parameters - explicitly exclude AI answer
            search_params = {
                "query": query,
                "max_results": max_results,
                "include_answer": False,  # Just search results, no AI answer
                **kwargs,
            }

            response = client.search(**search_params)
        except Exception as e:
            self._handle_api_request_error("Tavily", e)

        # Normalize Tavily response format
        # Tavily returns: {"results": [{"title": str, "url": str, "content": str, ...}]}
        normalized = self._create_normalized_response(response)

        # Extract results
        results = response.get("results", [])
        for result in results:
            normalized["results"].append(self._normalize_result_item(result))

        return normalized

    def _answer_tavily(self, question, answer_mode="basic", **kwargs):
        """
        Get AI-generated answer using Tavily backend.

        Args:
            question: The question to answer
            answer_mode: "basic" or "advanced" (default: "basic")
            **kwargs: Additional Tavily search parameters

        Returns:
            Normalized dict with "answer" and "sources" keys
        """
        try:
            from tavily import TavilyClient
        except ImportError:
            self._handle_import_error("tavily-python", "pip install tavily-python")

        try:
            api_key = self._check_api_key("TAVILY_API_KEY")
        except ApiKeyError as e:
            raise WebToolboxError(e.error_dict["message"]) from e

        try:
            client = TavilyClient(api_key=api_key)

            # Build search parameters
            search_params = {
                "query": question,
                "include_answer": answer_mode,  # "basic" or "advanced"
                **kwargs,
            }

            response = client.search(**search_params)
        except Exception as e:
            # API request failed (network error, rate limit, etc.) - return error dict for fallback
            self._handle_api_request_error("Tavily", e)

        # Response format validation - raise exception for programming errors
        # Tavily returns: {"answer": str, "results": [{"title": str, "url": str, "content": str, ...}, ...]}
        if not isinstance(response, dict):
            raise ValueError(
                f"Tavily returned unexpected response type: {type(response).__name__}. "
                f"Expected dict, got {type(response).__name__}. "
                f"Response: {str(response)[:500]}"
            )

        normalized = {"answer": response.get("answer", ""), "sources": []}

        # Extract sources from results
        results = response.get("results", [])
        if not isinstance(results, list):
            raise ValueError(
                f"Tavily response missing or invalid 'results' field. "
                f"Expected list, got {type(results).__name__}. "
                f"Response keys: {list(response.keys()) if isinstance(response, dict) else 'N/A'}"
            )

        for result in results:
            if not isinstance(result, dict):
                raise ValueError(
                    f"Tavily result item is not a dict: {type(result).__name__}. Result: {str(result)[:200]}"
                )
            source = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("content", "")[:200] if result.get("content") else "",
            }
            normalized["sources"].append(source)

        return normalized

    def _fetch_tavily(self, urls, extract_depth=None, **kwargs):
        """
        Fetch web page content using Tavily extract endpoint.

        Args:
            urls (str or list): Single URL string or list of URLs (max 20)
            extract_depth (str): "basic" or "advanced" (default: "basic")
            **kwargs: Additional Tavily extract parameters

        Returns:
            Normalized dict with "results" list (each with "content" (markdown), "title", "url")
        """
        try:
            from tavily import TavilyClient
        except ImportError:
            self._handle_import_error("tavily-python", "pip install tavily-python")

        try:
            api_key = self._check_api_key("TAVILY_API_KEY")
        except ApiKeyError as e:
            raise WebToolboxError(e.error_dict["message"]) from e

        # Normalize urls to list
        if isinstance(urls, str):
            urls = [urls]
        elif not isinstance(urls, list):
            raise WebToolboxError(
                "urls must be a string (single URL) or list of URLs (max 20). Pass a single URL string or a list of URLs."
            )

        if len(urls) > 20:
            raise WebToolboxError(
                "Tavily extract endpoint supports maximum 20 URLs per request. Split URLs into multiple requests of 20 or fewer."
            )

        try:
            client = TavilyClient(api_key=api_key)

            # Build extract parameters
            # According to Tavily docs, format defaults to "markdown" which is what we want
            extract_params = {"urls": urls, "format": "markdown"}
            if extract_depth is not None:
                extract_params["extract_depth"] = extract_depth
            extract_params.update(kwargs)

            response = client.extract(**extract_params)
        except Exception as e:
            self._handle_api_request_error("Tavily", e)

        # Tavily returns: {"results": [{"url": str, "title": str, "content": str, ...}, ...]}
        if not isinstance(response, dict):
            raise ValueError(
                f"Tavily returned unexpected response type: {type(response).__name__}. "
                f"Expected dict, got {type(response).__name__}. "
                f"Response: {str(response)[:500]}"
            )

        normalized = {"results": [], "raw_response": response}

        results = response.get("results", [])
        if not isinstance(results, list):
            raise ValueError(
                f"Tavily response missing or invalid 'results' field. "
                f"Expected list, got {type(results).__name__}. "
                f"Response keys: {list(response.keys()) if isinstance(response, dict) else 'N/A'}"
            )

        # Check for failed results
        failed_results = response.get("failed_results", [])
        if failed_results and not results:
            failed_urls = [fr.get("url", "unknown") for fr in failed_results if isinstance(fr, dict)]
            errors = [fr.get("error", "unknown error") for fr in failed_results if isinstance(fr, dict)]
            raise WebToolboxError(
                f"Tavily extract failed for all URLs. Failed: {', '.join(failed_urls[:3])}. "
                f"Errors: {', '.join(errors[:3])}. Try a different backend or check if the URLs are accessible."
            )

        for result in results:
            if not isinstance(result, dict):
                raise ValueError(
                    f"Tavily result item is not a dict: {type(result).__name__}. Result: {str(result)[:200]}"
                )
            normalized["results"].append(
                {
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "content": result.get("raw_content", "")
                    or result.get("content", ""),  # Tavily returns "raw_content"
                }
            )

        # If some URLs failed but we have some results, include failed_results in raw_response
        # (already included, but we could add a warning if needed)

        return normalized
