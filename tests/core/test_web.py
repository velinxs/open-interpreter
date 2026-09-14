import json
import os
import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from interpreter.core.toolbox.web.backends.linkup import _normalize_structured_schema
from interpreter.core.toolbox.web.web import ResultItem, StructuredOutputResult, Web, WebToolboxError


@contextmanager
def fake_sdk(module_name, **attributes):
    """Stand in for an optional SDK so a test runs where it is not installed.

    linkup-sdk and tavily-python are optional dependencies and absent from a
    default install, so patching `linkup.LinkupClient` skips the test instead
    of running it. The backends import their client inside the call, so a stub
    module in sys.modules is all they need.
    """
    module = types.ModuleType(module_name)
    for name, value in attributes.items():
        setattr(module, name, value)
    with patch.dict(sys.modules, {module_name: module}):
        yield


class TestWebToolbox(unittest.TestCase):
    def setUp(self):
        self.mock_toolbox = MagicMock()
        self.web = Web(self.mock_toolbox)

    def test_structured_output_linkup(self):

        pytest.importorskip("linkup", reason="linkup-sdk not installed")
        # Mock API key
        with patch.dict(os.environ, {"LINKUP_API_KEY": "fake_key"}):
            # Mock LinkupClient
            with patch("linkup.LinkupClient") as MockClient:
                mock_instance = MockClient.return_value

                # Mock successful response
                mock_response = MagicMock()
                mock_response.structured_output = {
                    "author_last_name": "Vaswani",
                    "year": 2017,
                    "title": "Attention is All You Need",
                }
                mock_response.sources = [
                    {
                        "title": "Paper on arXiv",
                        "url": "https://arxiv.org/abs/1706.03762",
                        "snippet": "We propose a new simple network architecture...",
                    }
                ]
                mock_instance.search.return_value = mock_response

                # Define schema
                schema = {
                    "type": "object",
                    "properties": {
                        "author_last_name": {"type": "string"},
                        "year": {"type": "integer"},
                        "title": {"type": "string"},
                    },
                }

                # Call the method
                result = self.web.structured_output("Attention is All You Need", schema=schema)

                # Verify call parameters
                MockClient.assert_called_once_with(api_key="fake_key")
                mock_instance.search.assert_called_once()
                call_kwargs = mock_instance.search.call_args.kwargs
                self.assertEqual(call_kwargs["output_type"], "structured")
                # Backend receives JSON string for dict schemas
                self.assertEqual(call_kwargs["structured_output_schema"], json.dumps(schema))

                # Verify result structure
                self.assertIsInstance(result, StructuredOutputResult)
                self.assertEqual(result["structured_output"]["author_last_name"], "Vaswani")
                self.assertEqual(result["structured_output"]["year"], 2017)
                self.assertEqual(len(result["sources"]), 1)
                self.assertEqual(result["sources"][0]["title"], "Paper on arXiv")

    def test_structured_output_pydantic_flexibility(self):

        pytest.importorskip("linkup", reason="linkup-sdk not installed")
        # Mock API key
        with patch.dict(os.environ, {"LINKUP_API_KEY": "fake_key"}):
            # Mock a Pydantic-like model by inheriting from a real one if available
            try:
                from pydantic import BaseModel

                class MockModel(BaseModel):
                    test: str

                original_schema = MockModel
            except ImportError:
                # Fallback to a mock that doesn't inherit but simulates the behavior
                class MockModel:
                    @staticmethod
                    def model_json_schema():
                        return {"type": "object", "properties": {"test": {"type": "string"}}}

                original_schema = MockModel

            with patch("linkup.LinkupClient") as MockClient:
                mock_instance = MockClient.return_value
                mock_instance.search.return_value = MagicMock(structured_output={"test": "val"}, sources=[])

                # Call with pydantic-like object
                result = self.web.structured_output("query", schema=original_schema)

                # Verify call parameters - Linkup SDK receives the class itself
                call_kwargs = mock_instance.search.call_args.kwargs
                self.assertEqual(call_kwargs["structured_output_schema"], original_schema)

    def test_structured_output_no_backend_available(self):

        pytest.importorskip("linkup", reason="linkup-sdk not installed")
        # Ensure no API keys are set
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(WebToolboxError) as context:
                self.web.structured_output("test", schema={"name": "string"})
            # It might raise the specific ApiKeyError message or the aggregate No backends message
            err_msg = str(context.exception)
            self.assertTrue("No structured output backends are working" in err_msg or "LINKUP_API_KEY" in err_msg)


if __name__ == "__main__":
    unittest.main()


class TestKeylessFetch(unittest.TestCase):
    """fetch must work on a fresh install, with no API key for anything."""

    def setUp(self):
        self.web = Web(MagicMock())

    def _response(self, text, content_type="text/html", url="https://example.com"):
        response = MagicMock()
        response.text = text
        response.url = url
        response.headers = {"Content-Type": content_type}
        response.raise_for_status = MagicMock()
        return response

    def test_fetch_falls_back_to_the_keyless_backend(self):
        """With no keys set, fetch uses `direct` instead of refusing.

        Every other backend needs a key from a commercial service, so the most
        obvious first thing to try — fetching a URL — failed on a fresh install
        with a wall of signup links.
        """
        html = "<html><head><title>Example Domain</title></head><body><h1>Hi</h1></body></html>"
        with patch.dict(os.environ, {}, clear=True):
            with patch("requests.get", return_value=self._response(html)) as get:
                result = self.web.fetch("https://example.com")

        assert result.backend == "direct"
        assert result.title == "Example Domain"
        assert "Hi" in result.content
        assert get.call_args.kwargs["timeout"] > 0, "a fetch with no timeout can hang the session"

    def test_the_keyless_backend_strips_scripts_and_styles(self):
        """Script and style bodies are removed, not handed to the model as content.

        They are the bulk of a modern page and none of it is readable, so
        leaving them in wastes the context window the fetch exists to fill.
        """
        html = (
            "<html><head><title>T</title><style>body{color:red}</style></head>"
            "<body><script>var secret = 1;</script><p>Real text</p></body></html>"
        )
        with patch.dict(os.environ, {}, clear=True):
            with patch("requests.get", return_value=self._response(html)):
                result = self.web.fetch("https://example.com/x")

        assert "Real text" in result.content
        assert "var secret" not in result.content
        assert "color:red" not in result.content

    def test_the_keyless_backend_passes_non_html_through(self):
        """JSON and plain text are already what the caller wanted.

        Running them through an HTML-to-markdown pass mangles them.
        """
        payload = '{"answer": 42}'
        with patch.dict(os.environ, {}, clear=True):
            with patch("requests.get", return_value=self._response(payload, "application/json")):
                result = self.web.fetch("https://example.com/api.json")

        assert result.content == payload

    def test_a_non_http_url_is_refused_with_a_useful_message(self):
        """A path or an unsupported scheme is named as the problem, not passed to requests.

        A bare hostname is no longer one of these: it now gains an https://
        scheme instead of being refused.
        """
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(WebToolboxError) as excinfo:
                self.web.fetch("ftp://example.com/files")
        assert "http" in str(excinfo.value).lower()


class TestResultItem(unittest.TestCase):
    """Result entries answer to whichever key the model guessed.

    Ported from classic/develop: models kept writing `r.title` on a plain dict
    and getting AttributeError, or `r["content"]` on a search hit whose key is
    `snippet`. Both now work, and a wrong guess still fails loudly.
    """

    def setUp(self):
        self.web = Web(MagicMock())

    def test_result_item_model_style_access(self):
        """Normalized items support the attribute access models kept guessing (r.title)."""
        item = self.web._normalize_result_item({"title": "T", "link": "http://x", "snippet": "S"})
        self.assertIsInstance(item, ResultItem)
        # Attribute access (the style that raised AttributeError on plain dicts)
        self.assertEqual(item.title, "T")
        self.assertEqual(item.url, "http://x")
        self.assertEqual(item.snippet, "S")
        # Key access still works
        self.assertEqual(item["title"], "T")
        self.assertEqual(item.get("url"), "http://x")

    def test_result_item_key_aliases(self):
        """Common key guesses resolve: content<->snippet, link/href->url, name->title."""
        item = self.web._normalize_result_item({"title": "T", "link": "http://x", "snippet": "S"})
        self.assertEqual(item["content"], "S")
        self.assertEqual(item.content, "S")
        linky = ResultItem({"title": "T", "link": "http://x", "description": "D"})
        self.assertEqual(linky.url, "http://x")
        self.assertEqual(linky["url"], "http://x")
        self.assertEqual(linky.snippet, "D")
        self.assertEqual(ResultItem({"name": "N", "url": "http://x", "snippet": "S"}).title, "N")
        # Reverse direction: fetch-style entries expose snippet as an alias of content
        page = ResultItem({"url": "http://x", "title": "T", "content": "C"})
        self.assertEqual(page.snippet, "C")
        self.assertEqual(page["snippet"], "C")
        self.assertEqual(page.get("snippet"), "C")

    def test_result_item_truly_missing_keys_still_error(self):
        """Forgiveness has limits: unknown keys raise KeyError/AttributeError, get() defaults."""
        item = ResultItem({"title": "T"})
        with self.assertRaises(KeyError):
            item["nope"]
        with self.assertRaises(AttributeError):
            item.nope
        self.assertIsNone(item.get("nope"))
        self.assertEqual(item.get("nope", "fallback"), "fallback")


class TestResultIndexGuidance(unittest.TestCase):
    """fetch(i) says what went wrong instead of leaking a Python error.

    Ported from classic/develop: models passed lists, strings and
    out-of-range indices, and got TypeError or IndexError with a full Jupyter
    traceback, which says nothing about how to call it correctly.
    """

    def setUp(self):
        self.web = Web(MagicMock())

    def _search_result(self):
        from interpreter.core.toolbox.web.web import SearchResult

        return SearchResult(
            {"results": [{"title": "T", "url": "http://a", "snippet": "S"}], "backend": "serper"},
            web=self.web,
        )

    def test_fetch_list_index_guides(self):
        """fetch([0, 1]) fails with guidance toward one-at-a-time fetching."""
        with self.assertRaises(WebToolboxError) as context:
            self._search_result().fetch([0, 1])
        self.assertIn("single result index", str(context.exception))

    def test_fetch_string_index_guides(self):
        """A non-integer index raises WebToolboxError, not TypeError."""
        from interpreter.core.toolbox.web.web import AnswerResult

        result = AnswerResult(
            {"answer": "A", "sources": [{"title": "T", "url": "http://b", "snippet": "S"}], "backend": "linkup"},
            web=self.web,
        )
        with self.assertRaises(WebToolboxError):
            result.fetch("0")

    def test_fetch_out_of_range_guides(self):
        """An out-of-range index names the valid range instead of leaking IndexError."""
        result = StructuredOutputResult(
            {
                "structured_output": {},
                "sources": [{"title": "T", "url": "http://c", "snippet": "S"}],
                "backend": "linkup",
            },
            web=self.web,
        )
        with self.assertRaises(WebToolboxError) as context:
            result.fetch(5)
        self.assertIn("out of range", str(context.exception))

    def test_fetch_bool_index_rejected(self):
        """bool is not silently accepted as an integer index (True would fetch result 1)."""
        with self.assertRaises(WebToolboxError):
            self._search_result().fetch(True)


class TestStructuredOutputSchema(unittest.TestCase):
    """A field map is a schema too.

    Ported from classic/develop: the full JSON Schema form is a lot of syntax
    to get right from memory, and getting it wrong came back as an opaque
    backend 400 that read like an authentication failure.
    """

    def setUp(self):
        self.web = Web(MagicMock())

    def test_simple_field_map_becomes_a_schema_with_all_fields_required(self):
        """{"name": "string"} expands to the object schema LinkUp expects."""
        self.assertEqual(
            _normalize_structured_schema({"name": "string", "founded": "integer"}),
            {
                "type": "object",
                "properties": {"name": {"type": "string"}, "founded": {"type": "integer"}},
                "required": ["name", "founded"],
            },
        )

    def test_full_schema_passes_through_untouched(self):
        """A dict with a "properties" mapping is already a schema, not a field map."""
        schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
        self.assertIs(_normalize_structured_schema(schema), schema)

    def test_field_named_type_is_read_as_a_field(self):
        """{"type": "string"} is ambiguous; it resolves to a field named "type"."""
        normalized = _normalize_structured_schema({"type": "string"})
        self.assertEqual(normalized["properties"], {"type": {"type": "string"}})

    def test_property_schema_values_are_kept(self):
        """A field map may still spell one field out in full, for arrays and the like."""
        normalized = _normalize_structured_schema({"tags": {"type": "array", "items": {"type": "string"}}})
        self.assertEqual(normalized["properties"]["tags"], {"type": "array", "items": {"type": "string"}})

    def test_unknown_type_name_fails_locally(self):
        """ "str" is not a JSON Schema type; say so here rather than as a backend 400."""
        with self.assertRaises(WebToolboxError) as context:
            _normalize_structured_schema({"name": "str"})
        self.assertIn("Valid types", str(context.exception))

    def test_empty_schema_raises(self):
        """An empty schema asks for nothing; the error says what to write instead."""
        with self.assertRaises(WebToolboxError) as context:
            self.web.structured_output("q", schema={})
        self.assertIn("at least one field", str(context.exception))

    def test_field_map_reaches_the_backend_as_a_json_schema(self):
        """The conversion happens before the request, not only in the caller's head."""
        client = MagicMock()
        client.return_value.search.return_value = MagicMock(structured_output={}, sources=[])
        with patch.dict(os.environ, {"LINKUP_API_KEY": "fake_key"}):
            with fake_sdk("linkup", LinkupClient=client):
                self.web.structured_output("Apple Inc", schema={"name": "string", "founded": "integer"})
        sent = client.return_value.search.call_args.kwargs["structured_output_schema"]
        self.assertEqual(
            json.loads(sent),
            {
                "type": "object",
                "properties": {"name": {"type": "string"}, "founded": {"type": "integer"}},
                "required": ["name", "founded"],
            },
        )

    def test_schema_rejection_blames_the_schema_not_the_api_key(self):
        """A backend 400 mentioning the schema used to read as "check your API key"."""
        client = MagicMock()
        client.return_value.search.side_effect = Exception(
            "Validation failed structuredOutputSchema: must be valid JSON schema of type object."
        )
        with patch.dict(os.environ, {"LINKUP_API_KEY": "fake_key"}):
            with fake_sdk("linkup", LinkupClient=client):
                with self.assertRaises(WebToolboxError) as context:
                    self.web.structured_output("q", schema={"name": "string"})
        msg = str(context.exception)
        self.assertIn("schema", msg.lower())
        self.assertNotIn("API key", msg)


class TestFetchUrlNormalization(unittest.TestCase):
    """What the model typed is turned into a URL once, before anything else.

    Ported from classic/develop: models write "example.com", and each backend
    decided separately what to do with it — one refused, others sent it and
    got a confusing failure back. Normalizing in fetch() also means the two
    spellings of one page share a cache entry.
    """

    def setUp(self):
        self.web = Web(MagicMock())
        self.page = {"url": "https://example.com", "title": "", "content": "hi"}

    def test_fetch_prepends_a_missing_scheme(self):
        """A bare hostname becomes https:// before any backend sees it."""
        with patch.object(self.web, "_fetch_direct", return_value=dict(self.page)) as fetch_direct:
            result = self.web.fetch("example.com", backend="direct")
        self.assertEqual(result["url"], "https://example.com")
        fetch_direct.assert_called_once_with("https://example.com")

    def test_fetch_rejects_malformed_urls(self):
        """Malformed URLs raise with an example instead of reaching a backend."""
        for bad in ["not a url", "", "ftp://example.com/files", "https://", "http://"]:
            with self.subTest(url=bad):
                with self.assertRaises(WebToolboxError) as context:
                    self.web.fetch(bad, backend="direct")
                self.assertIn("https://example.com", str(context.exception))

    def test_fetch_cache_is_shared_across_scheme_forms(self):
        """ "example.com" and "https://example.com" are one page, so one fetch."""
        with patch.object(self.web, "_fetch_direct", return_value=dict(self.page)) as fetch_direct:
            first = self.web.fetch("example.com")
            second = self.web.fetch("https://example.com")
        self.assertEqual(first["content"], "hi")
        self.assertTrue(second._cached)
        self.assertEqual(fetch_direct.call_count, 1)

    def test_multi_url_fetch_is_refused_by_backends_that_cannot_do_it(self):
        """urls=[...] is a tavily feature; the others were handed a list as the URL."""
        with self.assertRaises(WebToolboxError) as context:
            self.web.fetch(
                "https://example.com",
                backend="direct",
                urls=["https://example.com", "https://example.org"],
            )
        self.assertIn("tavily", str(context.exception))


class TestFetchResultLinks(unittest.TestCase):
    """links() reads the markdown backends actually produce.

    Ported from classic/develop: the old single regex stopped at the first
    ")", so a Wikipedia URL came back truncated, an optional link title was
    swallowed into the URL, and reference-style links were invisible.
    """

    def _page(self, content, backend="tavily"):
        from interpreter.core.toolbox.web.web import FetchResult

        return FetchResult({"url": "https://example.com", "title": "", "content": content, "backend": backend})

    def test_links_keep_balanced_parens_and_drop_titles(self):
        """A parenthesised Wikipedia title stays in the URL; a link title does not."""
        page = self._page(
            '[Python](https://en.wikipedia.org/wiki/Python_(programming_language) "Python") and [A](http://a)'
        )
        self.assertEqual(
            page.links(),
            [
                ("Python", "https://en.wikipedia.org/wiki/Python_(programming_language)"),
                ("A", "http://a"),
            ],
        )

    def test_links_resolve_reference_style(self):
        """[text][ref] and [text][] are links too, once their definitions are read."""
        page = self._page(
            "See [docs][d] and [home][]\n\n[d]: https://example.com/docs\n[home]: https://example.com/",
            backend="serper",
        )
        self.assertEqual(
            page.links(),
            [("docs", "https://example.com/docs"), ("home", "https://example.com/")],
        )

    def test_links_invent_nothing_for_dangling_references(self):
        """serper keeps [text][21] uses but strips the definitions; that is not a link."""
        page = self._page("See [docs][21] for details.", backend="serper")
        self.assertEqual(page.links(), [])


class TestLinkupFetchGuard(unittest.TestCase):
    """An SDK too old to fetch says so, instead of blaming the API key."""

    def test_old_linkup_sdk_asks_for_an_upgrade(self):
        """linkup-sdk 0.2.x has no fetch(); the AttributeError read as an auth failure.

        Ported from classic/develop: the AttributeError was caught by the
        handler that reports "Check your API key and internet connection", so
        the one fix that works — upgrading the SDK — was never suggested.
        """
        web = Web(MagicMock())
        client = MagicMock()
        del client.return_value.fetch
        with patch.dict(os.environ, {"LINKUP_API_KEY": "fake_key"}):
            with fake_sdk("linkup", LinkupClient=client):
                with self.assertRaises(WebToolboxError) as context:
                    web.fetch("https://example.com", backend="linkup")
        msg = str(context.exception)
        self.assertIn("--upgrade linkup-sdk", msg)
        self.assertNotIn("API key", msg)
