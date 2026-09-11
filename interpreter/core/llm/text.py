"""Running a model that has no tool calling, and reading code out of its prose.

A text model answers in markdown, so a code block is both the thing shown to
the user and the thing to execute. FenceParser pulls the two apart as the
stream arrives, which is fiddly because providers split the stream on
arbitrary token boundaries: a fence, its language tag, and the code may each
be cut in half, and a lone backtick in prose must not be mistaken for the
start of one.
"""

from .utils.stream_usage import record_stream_chunk_usage


def run_text_llm(llm, params):
    skip_execution_instructions = params.pop("skip_execution_instructions", False)
    plain_title_stream = params.pop("conversation_title_plain_stream", False)

    ## Setup

    if llm.execution_instructions and not skip_execution_instructions:
        try:
            # Add the system message
            params["messages"][0]["content"] += "\n" + llm.execution_instructions
            llm.interpreter._last_rendered_system_message = params["messages"][0]["content"]
        except:
            print('params["messages"][0]', params["messages"][0])
            raise

    if plain_title_stream:
        for chunk in llm.completions(**params):
            if llm.interpreter.verbose:
                print("Chunk in coding_llm", chunk)

            record_stream_chunk_usage(llm, chunk)

            if "choices" not in chunk or len(chunk["choices"]) == 0:
                continue
            delta = chunk["choices"][0]["delta"]
            if "reasoning_content" in delta and delta["reasoning_content"]:
                continue
            content = delta.get("content", "")
            if content is None or content == "":
                continue
            yield {"type": "message", "content": content}

        return

    ## Convert output to LMC format

    # OS mode was meant to default bare fences to "text", but the original
    # condition never fired, so "python" has always been the effective default.
    parser = FenceParser(default_language="python")

    for chunk in llm.completions(**params):
        if llm.interpreter.verbose:
            print("Chunk in coding_llm", chunk)

        record_stream_chunk_usage(llm, chunk)

        if "choices" not in chunk or len(chunk["choices"]) == 0:
            # This happens sometimes
            continue
        delta = chunk["choices"][0]["delta"]

        # Stream reasoning_content as it arrives
        if "reasoning_content" in delta and delta["reasoning_content"]:
            yield {"role": "assistant", "type": "message", "format": "reasoning", "content": delta["reasoning_content"]}
            continue

        content = delta.get("content", "")
        if content is None:
            continue

        yield from parser.feed(content)
        if parser.finished:
            # Stop at the first closing fence: respond() runs the code and asks again.
            return

    yield from parser.flush()


class FenceParser:
    """Incremental parser: text pieces in, LMC message/code chunks out.

    Providers stream on arbitrary token boundaries, so the parser must cope
    with a fence split across pieces ("``" + "`python"), a language tag split
    across pieces ("```py" + "thon\n"), code glued to the closing fence, and
    lone backticks in prose. It stops at the first closing fence.
    """

    def __init__(self, default_language="python"):
        self.default_language = default_language
        self.inside_code = False
        self.language = None
        self.finished = False
        self._header = ""  # the language line, accumulated until its newline
        self._pending = ""  # trailing backticks that may be the start of a fence

    def feed(self, piece):
        if self.finished or not piece:
            return []
        out = []
        buf = self._pending + piece
        self._pending = ""
        while buf:
            if not self.inside_code:
                idx = buf.find("```")
                if idx == -1:
                    text, self._pending = _split_trailing_backticks(buf)
                    if text:
                        out.append({"type": "message", "content": text})
                    break
                if idx:
                    out.append({"type": "message", "content": buf[:idx]})
                buf = buf[idx + 3 :]
                self.inside_code = True
                self.language = None
                self._header = ""
                continue

            if self.language is None:
                newline = buf.find("\n")
                if newline == -1:
                    self._header += buf
                    break
                self._header += buf[:newline]
                buf = buf[newline + 1 :]
                # Drop hallucinated spaces, digits and punctuation from the tag.
                self.language = "".join(c for c in self._header if c.isalpha()) or self.default_language
                continue

            idx = buf.find("```")
            if idx == -1:
                code, self._pending = _split_trailing_backticks(buf)
                if code:
                    out.append({"type": "code", "format": self.language, "content": code})
                break
            if idx:
                out.append({"type": "code", "format": self.language, "content": buf[:idx]})
            self.finished = True
            break
        return out

    def flush(self):
        """Emit backticks held back at end of stream that never became a fence."""
        pending, self._pending = self._pending, ""
        if not pending or self.finished:
            return []
        if self.inside_code and self.language is not None:
            return [{"type": "code", "format": self.language, "content": pending}]
        if not self.inside_code:
            return [{"type": "message", "content": pending}]
        return []


def _split_trailing_backticks(text):
    """Return (emit_now, hold_back): up to two trailing backticks may start a fence."""
    held = len(text) - len(text.rstrip("`"))
    if held == 0:
        return text, ""
    return text[:-held], text[-held:]


def stream_to_lmc(pieces, default_language="python"):
    """Parse an iterable of text pieces into LMC message/code chunks (test seam)."""
    parser = FenceParser(default_language)
    for piece in pieces:
        yield from parser.feed(piece)
        if parser.finished:
            return
    yield from parser.flush()
