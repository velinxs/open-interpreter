"""One headless turn of the interpreter, approvals decided by a callback.

The loop (respond.py) yields a "confirmation" chunk before running code or
applying an edit when the auto-run policy says a human must decide. The
terminal answers with a prompt; the server answers on the next HTTP request;
a channel answers with the next chat message. drive() is the one place that
knows how to continue, decline, or pause around that chunk.
"""

RUN = "run"  # continue: the loop executes the pending code
SKIP = "skip"  # record a decline and end the turn; the code is not run
PAUSE = "pause"  # end the turn with the code still pending; drive(None) resumes it

DECLINED_NOTICE = "[User declined to run this code.]"
EDIT_DECLINED_NOTICE = "[User declined to apply this edit.]"


def drive(interpreter, message=None, *, approve=None):
    """Yield the turn's LMC chunks. approve(chunk) -> RUN | SKIP | PAUSE.

    message: the user's message for a new turn, or None to resume a turn that
    was paused with code pending (the loop skips the model call when the last
    message is code, so resuming re-yields the confirmation).
    """
    approve = approve or (lambda chunk: RUN)
    if message is None:
        interpreter.last_messages_count = len(interpreter.messages)
        chunks = interpreter._respond_and_store()
    else:
        chunks = interpreter.chat(message, display=False, stream=True)
    try:
        for chunk in chunks:
            if chunk.get("type") != "confirmation":
                yield chunk
                continue
            decision = approve(chunk)
            if decision == RUN:
                continue  # resuming the generator runs the code
            yield chunk
            if decision == SKIP:
                notice = EDIT_DECLINED_NOTICE if chunk.get("format") == "edit" else DECLINED_NOTICE
                interpreter.messages.append({"role": "user", "type": "message", "content": notice, "source": "session"})
            return
    finally:
        chunks.close()
