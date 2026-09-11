"""Reasoning request parameters, and the OpenRouter provider preferences that
ride along with them.

include_reasoning / reasoning_effort are mapped to each provider's contract,
guarded by OpenRouter's model metadata (mandatory reasoning, supported
efforts) so a bad combination is dropped with a note instead of a 400.
"""

from .completions import _litellm

# Models already warned about during this process (mandatory reasoning / an
# unsupported effort value), so the note is printed once instead of every turn.
_warned_mandatory_reasoning = set()
_warned_unsupported_effort = set()


def apply_reasoning_params(llm, params, stream_options, model):
    """Mutate params and stream_options in place for the request to `model`."""
    litellm = _litellm()

    # Reasoning tokens: Some models support separate reasoning content
    # Set to {"exclude": True} to disable, or remove this line to allow reasoning tokens
    # params["reasoning"] = {"exclude": True}

    # OpenRouter provider preferences: Ensure consistent behavior across providers
    # For OpenRouter models, require providers to support all parameters to avoid inconsistency
    if model.startswith("deepseek/"):
        # LiteLLM maps include_reasoning / reasoning_effort to DeepSeek's thinking API.
        if getattr(litellm, "supports_reasoning", None) and litellm.supports_reasoning(model=model or llm.model):
            if llm.include_reasoning is not False:
                params["include_reasoning"] = True if llm.include_reasoning is None else llm.include_reasoning
                stream_options["include_reasoning"] = params["include_reasoning"]

    if model.startswith("openrouter/"):
        params["provider"] = {
            "require_parameters": True,  # Only use providers that support all request parameters
            "allow_fallbacks": False,  # Disable fallbacks to maintain consistency
        }
        # Request streaming reasoning when the model supports it, so LiteLLM forwards
        # delta.reasoning_content from OpenRouter. Requires LiteLLM v1.63.5+ (BerriAI/litellm#8631).
        # Only set for models that support reasoning to avoid 400 on non-reasoning OpenRouter models.
        if getattr(litellm, "supports_reasoning", None) and litellm.supports_reasoning(model=model or llm.model):
            # We use extra_body to pass reasoning parameters for OpenRouter models.
            # This ensures the tokens are requested via both legacy and modern API formats.
            params["extra_body"] = params.get("extra_body", {})
            params["extra_body"]["include_reasoning"] = True
            params["extra_body"]["reasoning"] = {"enabled": True}

            # Also set standard params for LiteLLM to handle unified mapping
            params["include_reasoning"] = True
            stream_options["include_reasoning"] = True

    # Override reasoning settings if explicitly set on interpreter.llm
    if llm.include_reasoning is not None:
        # Some OpenRouter endpoints (e.g. z-ai/glm-5.3-flash) mandate reasoning:
        # sending reasoning.enabled:false there returns 400 "Reasoning is
        # mandatory for this endpoint and cannot be disabled". OpenRouter's
        # model metadata flags this, so refuse to send the disable rather than
        # fail the request. The model then uses its default effort (max for GLM).
        _reasoning_mandatory = False
        if model.startswith("openrouter/"):
            _reasoning_mandatory = bool(
                ((llm._openrouter_model_entry(model) or {}).get("reasoning") or {}).get("mandatory")
            )
        if llm.include_reasoning is False and _reasoning_mandatory:
            if model not in _warned_mandatory_reasoning:
                llm.interpreter.display_message(
                    f"> **Note:** `{model}` always reasons and cannot have reasoning "
                    "disabled, so `include_reasoning: false` is being ignored."
                )
                _warned_mandatory_reasoning.add(model)
        else:
            params["include_reasoning"] = llm.include_reasoning
            stream_options["include_reasoning"] = llm.include_reasoning
            if model.startswith("openrouter/"):
                params["extra_body"] = params.get("extra_body", {})
                params["extra_body"]["include_reasoning"] = llm.include_reasoning
                params["extra_body"]["reasoning"] = {"enabled": llm.include_reasoning}

    # A reasoning_effort only makes sense when reasoning is enabled; sending
    # one alongside include_reasoning=false is contradictory and some backends
    # reject it, so skip it whenever the caller disabled reasoning explicitly.
    if llm.reasoning_effort and llm.include_reasoning is not False:
        # Guard against sending an unsupported effort level. GLM 5.3 only
        # accepts low/high/max and 400s on anything else (including "medium").
        _effort_ok = True
        if model.startswith("openrouter/"):
            _supported = ((llm._openrouter_model_entry(model) or {}).get("reasoning") or {}).get("supported_efforts")
            if _supported and llm.reasoning_effort not in _supported:
                _effort_ok = False
                if model not in _warned_unsupported_effort:
                    llm.interpreter.display_message(
                        f"> **Note:** `{model}` only supports reasoning_effort "
                        f"{_supported}, so `{llm.reasoning_effort}` is being ignored."
                    )
                    _warned_unsupported_effort.add(model)
        if _effort_ok:
            params["reasoning_effort"] = llm.reasoning_effort
            if model.startswith("openrouter/"):
                params["extra_body"] = params.get("extra_body", {})
                if "reasoning" not in params["extra_body"]:
                    params["extra_body"]["reasoning"] = {}
                params["extra_body"]["reasoning"]["effort"] = llm.reasoning_effort
