"""One mixin per search provider, combined into Web.

Each mixin holds only that provider's request and response shapes and calls
back into Web for the shared key check, error reporting and normalization.
"""

from .base import BackendPlumbing
from .brave import BraveBackend
from .linkup import LinkupBackend
from .serpapi import SerpApiBackend
from .serper import SerperBackend
from .tavily import TavilyBackend

__all__ = ["BackendPlumbing", "BraveBackend", "SerperBackend", "SerpApiBackend", "TavilyBackend", "LinkupBackend"]
