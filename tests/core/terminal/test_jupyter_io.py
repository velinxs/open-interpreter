"""What comes back out of the Python kernel, and what happens when it dies.

Two silent defects: the kernel's display formatter was configured with an
active_types list that left image/png and image/jpeg out, so every plot,
screenshot and PIL image reached the model as its text repr and the image
branch of the listener was unreachable; and a block that called exit() killed
the kernel, after which every later run() waited forever for channels that a
dead kernel never brings back.

These tests drive a real ipykernel, like tests/core/terminal/test_process_lifetime.py.
"""

import time
import types

import pytest

from interpreter.core.terminal.languages.jupyter_language import JupyterLanguage

DISPLAY_AN_IMAGE = (
    "from PIL import Image\nfrom IPython.display import display\ndisplay(Image.new('RGB', (4, 4), 'red'))"
)


def _interpreter(supports_vision=True):
    """The little of the interpreter that JupyterLanguage actually reads."""
    return types.SimpleNamespace(
        llm=types.SimpleNamespace(supports_vision=supports_vision),
        verbose=False,
        debug=False,
    )


def _text(chunks):
    return "".join(str(chunk.get("content")) for chunk in chunks)


@pytest.fixture(scope="module")
def kernel():
    """One kernel shared by the display tests, which leave it healthy."""
    language = JupyterLanguage(_interpreter())
    yield language
    language.terminate()


@pytest.fixture
def doomed_kernel():
    """A kernel of its own for the tests that kill one."""
    language = JupyterLanguage(_interpreter())
    yield language
    language.terminate()


@pytest.mark.timeout(180)
def test_a_displayed_image_reaches_a_vision_model_as_an_image(kernel):
    """display(pil_image) yields a base64 image chunk, not just a text repr.

    With image/png missing from active_types the kernel never rendered one, so
    a model that can see was told `<PIL.Image.Image image mode=RGB size=4x4>`
    and the whole image path — plots, toolbox.display.view() — was dead.
    """
    kernel.interpreter.llm.supports_vision = True
    chunks = list(kernel.run(DISPLAY_AN_IMAGE))
    images = [chunk for chunk in chunks if chunk["type"] == "image"]
    assert images, chunks
    assert images[0]["format"] == "base64.png"
    assert images[0]["content"].startswith("iVBORw0KGgo")  # base64 of the PNG magic number


@pytest.mark.timeout(180)
def test_a_plot_reaches_a_vision_model_as_an_image(kernel):
    """plt.show() sends the figure itself, which is the only useful form of it."""
    kernel.interpreter.llm.supports_vision = True
    chunks = list(kernel.run("import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])\nplt.show()"))
    assert [chunk for chunk in chunks if chunk["type"] == "image"], chunks


@pytest.mark.timeout(180)
def test_an_image_is_dropped_for_a_model_that_cannot_see_it(kernel):
    """A text-only model keeps the text repr instead of base64 it can't receive.

    Message conversion discards image messages when the model has no vision, so
    forwarding one would spend the context on nothing and lose the repr too.
    """
    kernel.interpreter.llm.supports_vision = False
    chunks = list(kernel.run(DISPLAY_AN_IMAGE))
    assert not [chunk for chunk in chunks if chunk["type"] == "image"], chunks
    assert "PIL.Image.Image" in _text(chunks)


@pytest.mark.timeout(180)
def test_markdown_and_unhandled_mimetypes_still_say_something(kernel):
    """Markdown is read, and a mimetype with no branch falls back to its text.

    text/markdown was the formatter's preferred type but nothing read it, so
    Markdown objects arrived as `<IPython.core.display.Markdown object>`; latex
    and json displays produced no chunk at all.
    """
    kernel.interpreter.llm.supports_vision = False
    chunks = list(kernel.run("from IPython.display import display, Markdown\ndisplay(Markdown('**bold**'))"))
    assert "**bold**" in _text(chunks), chunks
    chunks = list(kernel.run("from IPython.display import display\ndisplay({'text/latex': 'x^2'}, raw=True)"))
    assert "x^2" in _text(chunks), chunks


def test_markdown_is_preferred_over_html():
    """When a display offers both, take the markdown.

    An HTML chunk is handed on as executable code — the reason active_types was
    narrowed in the first place — so it must stay the last resort.
    """
    language = object.__new__(JupyterLanguage)  # skip kernel startup
    language.interpreter = _interpreter(supports_vision=False)
    chunk = language._display_chunk({"text/plain": "b", "text/html": "<b>b</b>", "text/markdown": "**b**"})
    assert chunk == {"type": "console", "format": "output", "content": "**b**"}


@pytest.mark.timeout(180)
def test_exit_restarts_the_kernel_instead_of_wedging_the_session(doomed_kernel):
    """A block calling exit() costs the session its state, not the session.

    run() waited on `self.kc.is_alive()`, which a dead kernel never satisfies,
    so one exit() — routine in script-style code an LLM writes — hung every
    later block forever, with no message and no recovery but Ctrl-C.
    """
    list(doomed_kernel.run("keep = 1"))
    list(doomed_kernel.run("exit()"))
    assert not doomed_kernel.km.is_alive()

    chunks = list(doomed_kernel.run("print('back')"))
    text = _text(chunks)
    assert "restarted" in text, chunks
    assert "back" in text, chunks
    # The new kernel is empty, which is why the notice above has to be said.
    assert "False" in _text(list(doomed_kernel.run("print('keep' in dir())")))


@pytest.mark.timeout(180)
def test_a_kernel_killed_outright_does_not_stall_the_block(doomed_kernel, monkeypatch):
    """os._exit() ends the block in about a second, not at the idle timeout.

    A kernel killed without warning (os._exit, a segfault, the OOM killer)
    never sends the idle status the listener waits for, so the block sat there
    for the full idle timeout — two minutes by default — before giving up.
    """
    monkeypatch.setenv("INTERPRETER_COMMAND_IDLE_TIMEOUT", "30")
    started = time.monotonic()
    chunks = list(doomed_kernel.run("import os\nos._exit(0)"))
    elapsed = time.monotonic() - started
    assert elapsed < 15, f"took {elapsed:.0f}s: {chunks}"
    assert "kernel exited" in _text(chunks), chunks
    assert "back" in _text(list(doomed_kernel.run("print('back')")))


HELPER = '''def summarize(rows, key='value'):
    """Average one column of a list of dicts."""
    total = 0
    count = 0
    for row in rows:
        total += row[key]
        count += 1
    return total / count if count else 0
'''


@pytest.mark.timeout(180)
def test_an_identical_redefinition_is_stripped_before_it_runs(kernel):
    """A helper the kernel already holds verbatim is dropped from the next block.

    The kernel fingerprints what it ran — source that carries the injected
    active_line markers — and the client fingerprints the raw definition the
    model wrote. The two hash the same thing only if the normalisation on both
    sides lines up, and if it ever stops lining up nothing is stripped and the
    saving silently becomes zero, which no fake-fingerprint test would notice.
    """
    kernel.interpreter.llm.supports_vision = False
    list(kernel.run(HELPER + "\nTHRESHOLD = 0.75\nprint('defined')"))
    assert kernel.function_fingerprints.get("summarize"), kernel.function_fingerprints
    assert kernel.variable_fingerprints.get("THRESHOLD"), kernel.variable_fingerprints

    resent = HELPER + "\nTHRESHOLD = 0.75\nprint(summarize([{'value': 1}, {'value': 3}]))"
    stripped, notice = kernel.strip_boilerplate(resent)
    assert "def summarize" not in stripped, stripped
    assert "THRESHOLD" not in stripped, stripped
    assert "already defined identically" in notice, notice
    assert "already set to that value" in notice, notice
    # What survives still runs against the definitions left in the kernel.
    assert "2.0" in _text(list(kernel.run(stripped)))


@pytest.mark.timeout(180)
def test_a_changed_helper_survives_and_replaces_the_old_one(kernel):
    """Only a byte-identical redefinition goes; an edited one runs and rebinds."""
    kernel.interpreter.llm.supports_vision = False
    list(kernel.run(HELPER + "\nprint('defined')"))
    edited = HELPER.replace("total / count", "total / count / 2")
    stripped, notice = kernel.strip_boilerplate(edited)
    assert stripped == edited
    assert notice is None
    list(kernel.run(edited))
    assert "1.0" in _text(list(kernel.run("print(summarize([{'value': 1}, {'value': 3}]))")))


@pytest.mark.timeout(180)
def test_the_fingerprint_marker_never_reaches_the_output(kernel):
    """The hidden ##oi_fp## line is bookkeeping; the model and the terminal must never see it."""
    kernel.interpreter.llm.supports_vision = False
    text = _text(list(kernel.run("def marked():\n    return 1\nprint('ran')")))
    assert "##oi_fp##" not in text, text
    # The REPL-state line is still delivered whole, newlines and all.
    assert "\n[Python REPL State: " in text, text
