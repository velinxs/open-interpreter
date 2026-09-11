"""API-key check for every HTTP and websocket request."""

import os


def authenticate_function(key):
    """
    This function checks if the provided key is valid for authentication.

    Returns True if the key is valid, False otherwise.
    """
    # Fetch the API key from the environment variables. If it's not set, return True.
    api_key = os.getenv("INTERPRETER_API_KEY", None)

    # If the API key is not set in the environment variables, return True.
    # Otherwise, check if the provided key matches the fetched API key.
    # Return True if they match, False otherwise.
    if api_key is None:
        return True
    else:
        return key == api_key
