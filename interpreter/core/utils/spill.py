"""Archiving console output that was too long to keep in the conversation.

Output over max_output is truncated for the model; the full text is appended
to a per-session file, one block per command, so it can still be read.
Every function takes the interpreter, which holds the file handle and offsets.
"""

import os
import tempfile


def _spill_file_path(interpreter):
    if interpreter._spill_path is None:
        interpreter._spill_path = os.path.join(tempfile.gettempdir(), f"oi_outputs_{os.getpid()}.log")
    return interpreter._spill_path


def _open_spill(interpreter, path):
    """Open the archive for update, creating it private to this user.

    Console output can contain credentials and the file lives in a shared
    temp directory, so the default 0644 would expose it to every other user
    on the machine.
    """
    if not os.path.exists(path):
        os.close(os.open(path, os.O_CREAT | os.O_WRONLY, 0o600))
    return open(path, "r+b")


def _record_full_output(interpreter, message, chunk_content):
    """Archive the untruncated console output; return a note naming it.

    truncate_output() rewrites the message content in place, and the next
    chunk is appended to that already-truncated string. So the accumulated
    content is NOT the real output once it overflows even once - the middle
    is already gone by the time anything could read it. The true stream is
    recorded here instead, chunk by chunk, before truncation touches it.

    Blocks are appended, so earlier commands stay readable for the whole
    session rather than being overwritten by the next big output.

    Each chunk appends only the bytes that are new, rewriting nothing but
    the block's own footer, which is fixed-size. Chunks arrive per line, so
    re-writing the whole block every time would make a large output cost
    O(n^2) in disk writes.

    Returns None while the output still fits, so nothing is created in the
    common case.
    """
    if message is not interpreter._spill_message:
        # New console message: start a new block at the current end of file.
        interpreter._spill_message = message
        interpreter._spill_pending = ""
        interpreter._spill_size = 0
        interpreter._spill_body_end = None
        interpreter._spill_index += 1

    chunk_content = chunk_content or ""
    interpreter._spill_pending += chunk_content
    interpreter._spill_size += len(chunk_content)
    if interpreter._spill_size <= interpreter.max_output:
        # Still small enough to read in full; keep holding it in memory.
        return None

    path = interpreter._spill_file_path()
    index = interpreter._spill_index
    footer = f"\n===== END BLOCK {index} ({interpreter._spill_size:,} chars) =====\n"
    try:
        # Binary so the recorded offset stays valid regardless of encoding.
        with interpreter._open_spill(path) as f:
            if interpreter._spill_body_end is None:
                # First overflow for this block: append a header after
                # whatever earlier blocks are already in the file.
                f.seek(0, os.SEEK_END)
                f.write(f"\n===== OI OUTPUT BLOCK {index} =====\n".encode("utf-8", errors="replace"))
            else:
                # Resume where the body ended, overwriting the old footer.
                f.seek(interpreter._spill_body_end)
            f.write(interpreter._spill_pending.encode("utf-8", errors="replace"))
            interpreter._spill_body_end = f.tell()
            f.write(footer.encode("utf-8", errors="replace"))
            # The footer grows as the character count does; truncate so a
            # shorter one can never leave a stale tail behind.
            f.truncate()
    except OSError:
        # Read-only or full disk: truncation must still work, just without
        # the archive.
        return None
    interpreter._spill_pending = ""
    return (
        f"If you need the part that was cut, the full output is archived at "
        f"{path} as block {index} (earlier blocks from this session are still "
        f"in that file). Grep it for what you need, or read just this block "
        f"with `awk '/^===== OI OUTPUT BLOCK {index} =/,/^===== END BLOCK {index} /' {path}`."
    )
