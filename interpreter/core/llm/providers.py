"""Provider-specific routing and metadata.

DashScope and DeepSeek defaults, Ollama model management, and the OpenRouter
model registry that fills the gaps in litellm's own.
"""

import os

import requests

from .completions import _litellm

# Cache of OpenRouter /api/v1/models entries keyed by model slug. Reasoning
# contract and modalities are fetched at most once per process; the list rarely
# changes within a session and a network round-trip per probe is wasteful.
_openrouter_model_entries = {}


def openrouter_model_entry(model, verbose=False):
    """
    Fetch the OpenRouter /api/v1/models entry for an openrouter model.

    OpenRouter proxies any provider's models, so LiteLLM's registry often
    doesn't list new ones (e.g. openrouter/qwen/qwen3.7-plus). OpenRouter's
    model list is authoritative for input modalities AND for the reasoning
    contract (mandatory reasoning, supported_efforts, default_effort), so a
    single cached fetch serves both the vision probe and the reasoning
    param-guarding below. Returns the entry dict, or None if the model is
    not openrouter/ or the list can't be fetched.
    """
    if not model.lower().startswith("openrouter/"):
        return None
    slug = model.split("openrouter/", 1)[-1]
    if slug in _openrouter_model_entries:
        return _openrouter_model_entries[slug]
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={
                "HTTP-Referer": os.environ.get("OR_SITE_URL", ""),
                "X-Title": os.environ.get("OR_APP_NAME", "Open Interpreter"),
            },
            timeout=10,
        )
        response.raise_for_status()
        for entry in response.json().get("data", []):
            if entry.get("id") == slug:
                _openrouter_model_entries[slug] = entry
                return entry
    except Exception as e:
        if verbose:
            print(f"Could not fetch OpenRouter model entry: {e}")
    _openrouter_model_entries[slug] = None
    return None


def configure(llm):
    """Apply provider defaults and probes to an Llm before its first request.

    DashScope and DeepSeek endpoint defaults, Ollama tag lookup / pull / warm-up
    and num_ctx-derived context window, then litellm's registry for the context
    window of anything else. Mutates llm in place.
    """
    litellm = _litellm()

    # Route explicit DashScope models to DashScope defaults (OpenAI-compatible).
    # Prefixes avoid ambiguous auto-routing vs other providers (e.g. deepseek/*).
    # Slugs:
    # - dashscope-intl/<model> (Singapore ap-southeast-1)
    # - dashscope-us/<model> (Virginia, US us-east-1)
    model_lower = llm.model.lower()
    dashscope_route = None
    if model_lower.startswith("dashscope-intl/"):
        dashscope_route = (
            llm.model.split("/", 1)[1].lower(),
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
    elif model_lower.startswith("dashscope-us/"):
        dashscope_route = (
            llm.model.split("/", 1)[1].lower(),
            "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        )
    if dashscope_route is not None:
        model_name, _dashscope_default_base = dashscope_route
        if llm.api_base is None:
            llm.api_base = _dashscope_default_base
        if llm.api_key is None:
            llm.api_key = os.environ.get("DASHSCOPE_API_KEY")
        # Qwen3.5 is a unified vision-language architecture — there are no separate
        # VL variants because every model in the family natively supports image input.
        # LiteLLM's registry does not know this yet, so we set it explicitly.
        if model_name.startswith("qwen3.5") and llm.supports_vision is None:
            llm.supports_vision = True
        # Route through OpenAI-compatible formatting for DashScope's compatible endpoint.
        llm.model = f"openai/{model_name}"

    # DeepSeek API (OpenAI-compatible). Keep deepseek/<model> for LiteLLM routing.
    if model_lower.startswith("deepseek/"):
        if llm.api_base is None:
            llm.api_base = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        if llm.api_key is None:
            llm.api_key = os.environ.get("DEEPSEEK_API_KEY")

    if llm.model.startswith("ollama/") and not ":" in llm.model:
        llm.model = llm.model + ":latest"

    llm._is_loaded = True

    if llm.model.startswith("ollama/"):
        model_name = llm.model.replace("ollama/", "")
        api_base = getattr(llm, "api_base", None) or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        names = []
        try:
            # List out all downloaded ollama models. Will fail if ollama isn't installed
            response = requests.get(f"{api_base}/api/tags")
            if response.ok:
                data = response.json()
                names = [model["name"] for model in data["models"] if "name" in model and model["name"]]

        except Exception as e:
            print(str(e))
            llm.interpreter.display_message(
                f"> Ollama not found\n\nPlease download Ollama from [ollama.com](https://ollama.com/) to use `{model_name}`.\n"
            )
            exit()

        # Download model if not already installed
        if model_name not in names:
            llm.interpreter.display_message(f"\nDownloading {model_name}...\n")
            requests.post(f"{api_base}/api/pull", json={"name": model_name})

        # Get context window if not set
        if llm.context_window == None:
            response = requests.post(f"{api_base}/api/show", json={"name": model_name})
            model_info = response.json().get("model_info", {})
            context_length = None
            for key in model_info:
                if "context_length" in key:
                    context_length = model_info[key]
                    break
            if context_length is not None:
                llm.context_window = context_length
        if llm.max_tokens == None:
            if llm.context_window != None:
                llm.max_tokens = int(llm.context_window * 0.2)

        # Send a ping, which will actually load the model
        model_name = model_name.replace(":latest", "")
        print(f"Loading {model_name}...\n")

        old_max_tokens = llm.max_tokens
        llm.max_tokens = 1
        llm.interpreter.toolbox.ai.chat("ping")
        llm.max_tokens = old_max_tokens

        llm.interpreter.display_message("*Model loaded.*\n")

    # Validate LLM should be moved here!!

    if llm.context_window == None:
        try:
            model_info = litellm.get_model_info(model=llm.model)
            llm.context_window = model_info["max_input_tokens"]
            if llm.max_tokens == None:
                llm.max_tokens = min(int(llm.context_window * 0.2), model_info["max_output_tokens"])
        except:
            pass
