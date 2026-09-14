"""Linkup search, answer, structured output and fetch."""

import json
import os

import requests

from ..results import ApiKeyError, WebToolboxError

# The JSON Schema type names a field map may use. Anything else is a typo or a
# shape that needs the full schema form.
_SIMPLE_SCHEMA_TYPES = ("string", "integer", "number", "boolean", "array", "object", "null")


def _normalize_structured_schema(schema):
    """
    Normalize the schema argument of Web.structured_output.

    Accepts a simple field map ({"name": "string", "founded": "integer"}) and
    converts it to {"type": "object", "properties": {...}, "required": [...]}
    with all fields required. Values may also be property-schema dicts
    ({"tags": {"type": "array", "items": {"type": "string"}}}). Anything else
    (full JSON schemas, Pydantic classes, JSON strings) passes through untouched.

    Detection is a single rule: a dict with a "properties" mapping is a full
    JSON schema; any other dict is a field map. One caveat: a field literally
    named "properties" requires the full-schema form.
    """
    if not isinstance(schema, dict):
        return schema
    if isinstance(schema.get("properties"), dict):
        return schema
    if not schema:
        raise WebToolboxError(
            "Empty schema: define at least one field, e.g. schema={'name': 'string'}. "
            "Or pass a full JSON schema {'type': 'object', 'properties': {...}}."
        )
    properties = {}
    for field, spec in schema.items():
        if isinstance(spec, str):
            if spec not in _SIMPLE_SCHEMA_TYPES:
                raise WebToolboxError(
                    f"Invalid type '{spec}' for field '{field}'. Valid types: "
                    f"{', '.join(_SIMPLE_SCHEMA_TYPES)}. For constraints, nesting, or "
                    "descriptions, pass a full JSON schema {'type': 'object', 'properties': {...}}."
                )
            properties[field] = {"type": spec}
        elif isinstance(spec, dict) and spec.get("type") in _SIMPLE_SCHEMA_TYPES:
            properties[field] = spec
        else:
            raise WebToolboxError(
                f"Invalid spec for field '{field}': expected a type name "
                f"({', '.join(_SIMPLE_SCHEMA_TYPES)}) or a property schema like "
                "{'type': 'string'}. For complex shapes, pass a full JSON schema "
                "{'type': 'object', 'properties': {...}}."
            )
    return {"type": "object", "properties": properties, "required": list(schema.keys())}


def _encode_schema_for_sdk(schema):
    """
    Encode a schema the way the LinkUp SDK wants it.

    It takes a Pydantic model CLASS, a JSON STRING, or None — never a dict —
    so dicts are JSON-encoded here and everything else passes through.
    """
    is_pydantic = False
    try:
        # Check if it's a Pydantic class (v1 or v2)
        if isinstance(schema, type):
            # Try to import any version of Pydantic to check inheritance
            try:
                from pydantic import BaseModel as BM2

                if issubclass(schema, BM2):
                    is_pydantic = True
            except ImportError:
                pass

            if not is_pydantic:
                try:
                    from pydantic.v1 import BaseModel as BM1

                    if issubclass(schema, BM1):
                        is_pydantic = True
                except ImportError:
                    pass
        elif hasattr(schema, "__pydantic_model__"):  # some wrappers
            is_pydantic = True
    except Exception:
        # If any check fails, treat as non-pydantic
        pass

    if not is_pydantic and isinstance(schema, dict):
        return json.dumps(schema)
    return schema


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
            structured_output_schema: JSON schema dict, JSON string, or Pydantic class
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
                "structured_output_schema": _encode_schema_for_sdk(structured_output_schema),
                **kwargs,
            }

            response = client.search(**search_params)
        except Exception as e:
            # A 400 here is almost always a schema problem, not an auth problem —
            # say so instead of sending the caller to check their API key.
            if "schema" in str(e).lower():
                raise WebToolboxError(
                    "LinkUp rejected the output schema. Pass a full JSON schema "
                    "({'type': 'object', 'properties': {...}}) or a simple field map "
                    f"({{'name': 'string'}}). Backend error: {e}"
                ) from e
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
        except Exception as e:
            self._handle_api_request_error("LinkUp", e)

        if not hasattr(client, "fetch"):
            # Old SDK versions (0.2.x) predate the fetch endpoint; without this
            # guard the AttributeError below is caught by the handler beneath
            # and reported as "check your API key", which it is not.
            raise WebToolboxError(
                "The installed linkup-sdk has no fetch support. "
                "Upgrade: pip install --upgrade linkup-sdk (or use backend='serper'/'tavily'/'direct')."
            )

        try:
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
