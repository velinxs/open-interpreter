import json
import os
import unittest
from unittest.mock import MagicMock, patch

import pytest

from interpreter.core.toolbox.web.web import StructuredOutputResult, Web, WebToolboxError


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
                self.web.structured_output("test", schema={})
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
        """A path or a bare hostname is named as the problem, not passed to requests."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(WebToolboxError) as excinfo:
                self.web.fetch("example.com")
        assert "http" in str(excinfo.value).lower()
