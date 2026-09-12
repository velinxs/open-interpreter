"""The objects the web toolbox hands back to code the model wrote.

Every one of these is a dict with a hand-written __repr__, and that repr is
what lands in the model's context window when it evaluates a search in the
REPL. The reason it exists is truncation: a raw dump of ten pages of scraped
text would cost more context than the answer is worth. So the tests are mostly
about what the repr keeps, what it cuts, and the attribute access that makes
the objects usable at all.
"""

import pytest

from interpreter.core.toolbox.web.results import (
    AnswerResult,
    ApiKeyError,
    FetchResult,
    SearchResult,
    StructuredOutputResult,
    WebToolboxError,
    _normalize_locale_country_for_gl,
    _normalize_locale_language_for_hl,
    _normalize_tavily_single_page,
)


class FakeWeb:
    def __init__(self):
        self.fetched = []

    def fetch(self, url):
        self.fetched.append(url)
        return FetchResult({"url": url, "content": "page body"})


def _search_result(count, web=None):
    return SearchResult(
        {
            "backend": "serper",
            "results": [
                {
                    "title": f"Result {i}",
                    "url": f"https://example{i}.com/page",
                    "snippet": f"snippet {i}",
                }
                for i in range(count)
            ],
        },
        web=web,
    )


# --- locale normalisation ---------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [("en", "en"), ("EN", "en"), ("en_GB", "en"), ("pt-BR", "pt"), ("", "en"), ("nonsense", "en")],
)
def test_language_codes_are_reduced_to_iso_639_1(value, expected):
    """Search APIs take a bare two-letter language; anything else is rejected or ignored.

    Passing "en_GB" as hl= silently returns unlocalised results on some
    backends and errors on others, so the normalisation has to happen here
    rather than at the API.
    """
    assert _normalize_locale_language_for_hl(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [("US", "US"), ("us", "US"), ("en_GB", "GB"), ("", "US"), ("nonsense", "US")],
)
def test_country_codes_are_reduced_to_uppercase_iso_3166(value, expected):
    """gl= needs an uppercase two-letter territory, with US as the safe fallback."""
    assert _normalize_locale_country_for_gl(value) == expected


def test_country_normalisation_survives_non_breaking_spaces():
    """Locale values copied from a web page can carry U+00A0.

    An unstripped non-breaking space makes the two-letter check fail and
    silently downgrades every search to US results.
    """
    assert _normalize_locale_country_for_gl(" GB ") == "GB"


# --- attribute access -------------------------------------------------------


@pytest.mark.parametrize(
    "result",
    [
        SearchResult({"results": [], "backend": "x"}),
        FetchResult({"content": "text"}),
        AnswerResult({"answer": "yes", "sources": []}),
        StructuredOutputResult({"structured_output": {}}),
    ],
)
def test_missing_attributes_raise_attribute_error_with_a_hint(result):
    """A typo'd field name explains itself instead of raising KeyError.

    These objects are handled by code the *model* wrote, and the traceback is
    what the model reads to fix itself. "KeyError: 'resluts'" teaches it
    nothing; naming the real accessor and .keys() does.
    """
    with pytest.raises(AttributeError) as excinfo:
        result.definitely_not_a_field
    assert "keys()" in str(excinfo.value)


def test_results_are_still_ordinary_dicts():
    """Each result subclasses dict, so it serialises and prints plainly.

    Code the model writes will json.dumps these. A custom class would fail
    there and force the model into a retry loop it cannot diagnose.
    """
    result = SearchResult({"backend": "serper", "results": []})
    assert result["backend"] == "serper"
    assert dict(result) == {"backend": "serper", "results": []}


# --- SearchResult -----------------------------------------------------------

def test_search_repr_shows_at_most_five_results_and_a_count(capsys):
    """Only the first five hits are rendered, with the remainder summarised.

    This is the context-window guard: a 50-result search rendered in full
    would push out the conversation it was meant to inform.
    """
    text = repr(_search_result(12))
    assert "SearchResult(12 results)" in text
    assert "11. " not in text
    assert "... 7 more" in text


def test_search_repr_shows_the_domain_not_the_whole_url():
    """Long URLs are reduced to their domain in the listing.

    A page of tracking-parameter URLs is pure noise; the domain is the part
    that tells the model whether the hit is worth fetching.
    """
    result = SearchResult(
        {"backend": "serper", "results": [{"title": "T", "url": "https://en.wikipedia.org/wiki/Foo?x=1", "snippet": ""}]}
    )
    text = repr(result)
    assert "en.wikipedia.org" in text
    assert "?x=1" not in text


def test_search_repr_names_the_follow_up_calls():
    """The repr documents fetch()/find()/links() inline.

    The model has no docs for these objects beyond what it sees. Without the
    hint it tends to re-run the search instead of fetching a result.
    """
    text = repr(_search_result(1))
    assert "result.fetch(i)" in text
    assert "page.find(term)" in text


def test_fetching_a_search_result_by_index_uses_its_url():
    """result.fetch(2) fetches the third hit's URL, not the query again."""
    web = FakeWeb()
    _search_result(3, web=web).fetch(2)
    assert web.fetched == ["https://example2.com/page"]


# --- FetchResult ------------------------------------------------------------


def test_find_returns_snippets_around_each_case_insensitive_match():
    """find() is the model's way to read a long page without printing it all.

    Returning the whole page instead would defeat the point; matching
    case-sensitively would miss most real-world searches.
    """
    page = FetchResult({"content": "alpha BETA gamma beta delta"})
    snippets = page.find("beta", context=5)
    assert len(snippets) == 2
    assert [snippet.lower() for snippet in snippets] == ["lpha beta gamm", "amma beta delt"]


def test_find_respects_a_result_cap():
    """max_results stops after N matches, so a common word cannot flood the context."""
    page = FetchResult({"content": "the " * 500})
    assert len(page.find("the", max_results=3)) == 3


def test_find_flattens_newlines_in_its_snippets():
    """Snippets are single-line, so a match is one readable row.

    A snippet spanning twenty lines of scraped markup is unreadable and
    expensive.
    """
    page = FetchResult({"content": "start\nmiddle TARGET middle\nend"})
    assert "\n" not in page.find("TARGET")[0]


def test_find_and_links_work_across_a_multi_page_result():
    """Multi-URL fetches search all pages, not just the first.

    Tavily returns several pages under "results"; ignoring the rest would make
    find() quietly miss most of what was fetched.
    """
    page = FetchResult(
        {
            "results": [
                {"content": "first page mentions widgets"},
                {"content": "second page has [a link](https://example.com)"},
            ]
        }
    )
    assert page.find("widgets")
    assert page.links() == [("a link", "https://example.com")]


def test_links_only_extracts_http_urls():
    """Relative and javascript: targets are not returned as links.

    Handing the model a relative href it cannot fetch produces a failed fetch
    and a wasted turn.
    """
    page = FetchResult(
        {"content": "[ok](https://example.com) [rel](/local/path) [js](javascript:alert(1))"}
    )
    assert page.links() == [("ok", "https://example.com")]


def test_single_page_repr_previews_the_content_and_states_its_length():
    """The repr shows the size and a short preview rather than the whole page.

    A fetched page is routinely 100k characters. Printing it is the single
    easiest way to blow the context window.
    """
    page = FetchResult({"backend": "serper", "title": "Doc", "content": "x" * 50000})
    text = repr(page)
    assert "50,000 chars" in text
    assert len(text) < 1000
    assert "Doc" in text


def test_a_page_with_no_title_says_so():
    """An untitled page is labelled, not rendered as an empty line."""
    assert "[no title]" in repr(FetchResult({"content": "body"}))


def test_multi_page_repr_lists_three_pages_and_counts_the_rest():
    """The multi-page listing is capped the same way the search listing is."""
    page = FetchResult(
        {
            "backend": "tavily",
            "results": [
                {"title": f"T{i}", "url": f"https://e{i}.com/p", "content": "y" * 100} for i in range(7)
            ],
        }
    )
    text = repr(page)
    assert "FetchResult(7 pages)" in text
    assert "... 4 more" in text


def test_a_cached_fetch_is_marked_as_such():
    """The repr says when the page came from cache.

    Otherwise a stale page is indistinguishable from a fresh one, and the
    model cannot tell why a just-changed page still looks old.
    """
    page = FetchResult({"content": "body"})
    page._cached = True
    assert "[cached]" in repr(page)


# --- AnswerResult and StructuredOutputResult --------------------------------


def test_answer_repr_shows_the_whole_answer_and_the_source_count():
    """The answer itself is never truncated; it is the payload.

    Sources are summarised by count because the model can fetch them by index
    if it wants them.
    """
    result = AnswerResult(
        {"backend": "tavily", "answer": "line one\nline two", "sources": [{"url": "https://a.com"}] * 4}
    )
    text = repr(result)
    assert "AnswerResult(4 sources)" in text
    assert "line one" in text
    assert "line two" in text


def test_fetching_an_answer_source_by_index_uses_its_url():
    """result.fetch(i) on an answer follows the i-th source."""
    web = FakeWeb()
    AnswerResult({"sources": [{"url": "https://a.com"}, {"url": "https://b.com"}]}, web=web).fetch(1)
    assert web.fetched == ["https://b.com"]


def test_structured_output_repr_lists_field_names_and_a_short_json_preview():
    """The repr names the fields and shows at most six lines of JSON.

    Structured output can be a large nested object; the field list is what
    tells the model how to index into it without printing it.
    """
    result = StructuredOutputResult(
        {"backend": "linkup", "structured_output": {f"field{i}": i for i in range(20)}}
    )
    text = repr(result)
    assert "field0" in text
    assert "field15" not in text.split("Keys inside")[1].split("\n")[0]
    assert text.rstrip().endswith("...")


def test_structured_output_with_nothing_in_it_says_empty():
    """An empty structured result is labelled rather than rendered as a blank line."""
    assert "(empty)" in repr(StructuredOutputResult({"structured_output": {}}))


def test_fetching_a_structured_result_with_no_sources_explains_itself():
    """Calling fetch() when the backend returned no sources raises a readable error.

    An IndexError here would surface to the model as a bare traceback from
    inside the library, with nothing to act on.
    """
    with pytest.raises(WebToolboxError) as excinfo:
        StructuredOutputResult({"structured_output": {}}).fetch(0)
    assert "No sources" in str(excinfo.value)


# --- errors -----------------------------------------------------------------


def test_api_key_errors_render_as_one_line_in_notebooks():
    """The Jupyter traceback is suppressed in favour of the message.

    A missing API key is a configuration problem, not a bug. Twenty frames of
    library internals bury the one line that says which key to set.
    """
    error = ApiKeyError({"error": "missing key", "message": "Set TAVILY_API_KEY"})
    assert error._render_traceback_() == ["ApiKeyError: Set TAVILY_API_KEY"]
    assert str(error) == "missing key"


def test_web_toolbox_errors_render_as_one_line_too():
    """Same treatment for the general web failures."""
    assert WebToolboxError("no backends configured")._render_traceback_() == [
        "WebToolboxError: no backends configured"
    ]


# --- backend normalisation --------------------------------------------------


def test_a_single_tavily_page_is_flattened_to_the_shape_other_backends_use():
    """Tavily always returns a list; a single-URL fetch is unwrapped to match serper.

    Without this, result.content works on one backend and raises on another —
    and which backend answered is not something the model chose.
    """
    flat = _normalize_tavily_single_page(
        {
            "results": [{"url": "https://a.com", "title": "T", "content": "body"}],
            "raw_response": {"meta": 1},
        }
    )
    assert flat["content"] == "body"
    assert flat["raw_response"] == {"meta": 1}


def test_an_empty_tavily_fetch_raises_a_readable_error():
    """No pages means the URL was blocked or unreachable, and the error says so.

    Returning an empty dict instead would surface later as a missing-key
    AttributeError somewhere unrelated.
    """
    with pytest.raises(WebToolboxError) as excinfo:
        _normalize_tavily_single_page({"results": []})
    assert "no results" in str(excinfo.value)
