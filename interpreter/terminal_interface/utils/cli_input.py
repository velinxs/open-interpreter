def cli_input(prompt: str = "") -> str:
    start_marker = '"""'
    end_marker = '"""'
    message = input(prompt)

    # Multi-line input mode
    if start_marker in message:
        # The whole quoted block may already be here: with bracketed paste the
        # terminal delivers a multi-line paste in a single input() call, and a
        # one-line '"""..."""' used to hang forever waiting for a closing line
        # the user had already typed.
        if message.count(start_marker) >= 2:
            return message
        lines = [message]
        while True:
            line = input()
            lines.append(line)
            if end_marker in line:
                break
        return "\n".join(lines)

    # Single-line input mode
    return message
