"""SerpApi (multi-engine SERP proxy)."""

import json
import os

import requests

from ..results import ApiKeyError, WebToolboxError


class SerpApiBackend:
    def _search_serpapi(self, query, num=10, engine="google", country_code=None, language_code=None, **kwargs):
        """
        Search using SerpApi backend (supports multiple search engines).

        Uses appropriate SerpApi classes for each engine:
        - Google-based engines (google, google_scholar, google_news, google_shopping, google_images, youtube):
          Uses GoogleSearch with engine parameter
        - Other engines: Uses specific classes (BingSearch, YahooSearch, etc.)

        Args:
            query (str): The search query
            num (int): Number of results to return (default: 10)
            engine (str): Search engine to use (default: "google")
                         Options: "google", "google_scholar", "google_news", "google_shopping",
                                  "google_images", "youtube", "bing", "yahoo", "duckduckgo",
                                  "baidu", "yandex", "ebay", "walmart", "home_depot",
                                  "apple_app_store", "naver", etc.
            country_code (str): 2-letter country code for localized results (e.g., "us", "gb", default: system locale)
            language_code (str): 2-letter language code for interface (e.g., "en", "es", default: system locale)
            **kwargs: Additional SerpApi parameters (location, google_domain, safe, start, filter, tbm, etc.)

        Returns:
            Normalized dict with "results" key
        """
        country_code, language_code = self._get_locale_defaults(country_code, language_code)

        # Import SerpApi - try to get specific classes, fallback to GoogleSearch
        try:
            from serpapi import GoogleSearch
        except ImportError:
            self._handle_import_error("google-search-results", "pip install google-search-results")

        try:
            api_key = self._check_api_key("SERPAPI_API_KEY")
        except ApiKeyError as e:
            raise WebToolboxError(e.error_dict["message"]) from e

        # Google-based engines use GoogleSearch with engine parameter
        # These work with GoogleSearch class + engine parameter
        # NOTE: YouTube should use YoutubeSearch class, not GoogleSearch with engine="youtube"
        google_engines = ["google", "google_scholar", "google_news", "google_shopping", "google_images"]

        # Engines that should use specific classes (if available)
        specific_class_engines = {
            "youtube": "YoutubeSearch",  # YouTube should use YoutubeSearch class
            "bing": "BingSearch",
            "yahoo": "YahooSearch",
            "duckduckgo": "DuckDuckGoSearch",
            "baidu": "BaiduSearch",
            "yandex": "YandexSearch",
            "ebay": "EbaySearch",
            "walmart": "WalmartSearch",
            "home_depot": "HomeDepotSearch",
            "apple_app_store": "AppleAppStoreSearch",
            "naver": "NaverSearch",
        }

        try:
            if engine in google_engines:
                # Use GoogleSearch with engine parameter for Google-based engines
                search_params = {
                    **kwargs,
                    "q": query,
                    "num": num,
                    "engine": engine,
                    "api_key": api_key,
                    "gl": country_code,
                    "hl": language_code,
                }
                search = GoogleSearch(search_params)
                data = search.get_dict()
            elif engine in specific_class_engines:
                # Use specific class for non-Google engines
                class_name = specific_class_engines[engine]
                try:
                    # Dynamically import the specific class
                    module = __import__("serpapi", fromlist=[class_name])
                    SearchClass = getattr(module, class_name)
                except AttributeError:
                    raise WebToolboxError(
                        f"The {class_name} class is not available in your serpapi package version. "
                        "Update: pip install --upgrade google-search-results"
                    )

                # Build params - different engines use different query parameter names
                serpapi_query_params = {
                    "yahoo": "p",
                    "ebay": "_nkw",
                    "youtube": "search_query",  # YouTube uses "search_query" not "q"
                }
                search_params = {**kwargs, "api_key": api_key}
                query_param = serpapi_query_params.get(engine, "q")
                search_params[query_param] = query
                if num:
                    search_params["num"] = num

                # Add localization for engines that support it (after kwargs so locale wins)
                if engine in ["bing", "yahoo", "duckduckgo", "youtube"]:
                    search_params["gl"] = country_code
                    search_params["hl"] = language_code

                search = SearchClass(search_params)
                data = search.get_dict()
            else:
                raise WebToolboxError(
                    f"Engine '{engine}' is not supported. Supported engines: "
                    f"{', '.join(google_engines + list(specific_class_engines.keys()))}"
                )
        except Exception as e:
            self._handle_api_request_error("SerpApi", e)

        # Normalize SerpApi response format
        # Different engines return results in different keys
        normalized = self._create_normalized_response(data)

        # Map engine to response key
        # Different engines return results in different keys
        engine_result_keys = {
            "google": "organic_results",
            "google_scholar": "organic_results",
            "bing": "organic_results",
            "yahoo": "organic_results",
            "duckduckgo": "organic_results",
            "youtube": "video_results",  # YouTube returns video_results
            "google_news": "news_results",
            "google_shopping": "shopping_results",
            "google_images": "images_results",
            "ebay": "organic_results",  # eBay returns organic_results
            "walmart": "organic_results",
            "home_depot": "organic_results",
            "apple_app_store": "organic_results",
            "naver": "organic_results",
            "baidu": "organic_results",
            "yandex": "organic_results",
        }

        # Check for API errors first
        if "error" in data:
            raise WebToolboxError(
                f"SerpApi {engine} search error: {data.get('error', 'Unknown API error')}. "
                "Check your API key and query parameters."
            )

        # Get the appropriate results key for this engine
        results_key = engine_result_keys.get(engine, "organic_results")
        results = data.get(results_key, [])

        # If no results found in the expected key, fail loudly with debug info
        if not results:
            available_keys = [k for k in data.keys() if isinstance(data.get(k), list)]
            all_keys = list(data.keys())
            raise WebToolboxError(
                f"SerpApi response did not contain results in expected key '{results_key}'. "
                f"Available list keys: {available_keys}. All keys: {all_keys[:20]}"
            )

        # Extract and normalize results
        for result in results:
            normalized["results"].append(self._normalize_result_item(result, engine=engine))

        return normalized
