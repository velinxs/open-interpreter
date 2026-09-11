import os

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
# Note: litellm in DEV mode will load .env files from the current directory
# and all parent directories. This can lead to unexpected API keys being loaded
# if there are .env files in parent folders.
import json
import logging
import re
import sys
import uuid

import requests
from rich import print as rich_print
from rich.markdown import Markdown
from rich.panel import Panel

from .text import run_text_llm

# Appended to the system message in tool-calling mode. Tool names, parameters, languages,
# and execution modes live only in request.tools JSON — not duplicated here.
_TOOL_CALLING_INSTRUCTIONS = (
    "Use only the tools in this request (`execute`, `edit`, and `view_image` when present). "
    "Read each tool's JSON schema for parameters and languages.\n\n"
    '(What appears in the conversation log as {"role": "assistant", "type": "code", ...} '
    'or {"type": "edit", ...} is our internal storage derived from your tool call; '
    "you do not output that structure.) "
    "Code in message content is only shown and is not run. "
    "Do not call any other name as a tool (e.g. toolbox.web.answer). "
    'Those are Python APIs: use them inside the "code" string you pass to execute.'
)

# from .run_function_calling_llm import run_function_calling_llm
from .completions import _litellm, fixed_litellm_completions  # noqa: E402
from .errors import AccessDeniedError, FunctionCallingNotSupportedError, ModelNotFoundError  # noqa: E402
from .providers import configure, openrouter_model_entry  # noqa: E402
from .tool_calling import run_tool_calling_llm
from .utils.cache_aware_trim import cache_aware_trim
from .utils.convert_to_openai_messages import convert_to_openai_messages
from .utils.sanitize_secrets import sanitize_messages, should_sanitize_for_model

# Models already warned about during this process (mandatory reasoning / an
# unsupported effort value), so the note is printed once instead of every turn.
_warned_mandatory_reasoning = set()
_warned_unsupported_effort = set()

# Create or get the logger
logger = logging.getLogger("LiteLLM")


class SuppressDebugFilter(logging.Filter):
    def filter(self, record):
        # Suppress only the specific message containing the keywords
        if "cost map" in record.getMessage():
            return False  # Suppress this log message
        return True  # Allow all other messages


class Llm:
    """
    A stateless LMC-style LLM with some helpful properties.
    """

    def __init__(self, interpreter):
        # Add the filter to the logger
        logger.addFilter(SuppressDebugFilter())

        # Store a reference to parent interpreter
        self.interpreter = interpreter

        # OpenAI-compatible chat completions "endpoint"
        self.completions = fixed_litellm_completions

        # Settings
        self.model = "gpt-4o-mini"
        self.temperature = 0

        self.supports_vision = None  # Will try to auto-detect
        self.vision_renderer = self.interpreter.toolbox.vision.query  # Will only use if supports_vision is False

        self.supports_functions = None  # Will try to auto-detect
        self.execution_instructions = "To execute code on the user's machine, write a markdown code block. Specify the language after the ```. You will receive the output. Use any programming language."  # If supports_functions is False, this will be added to the system message
        # Appended to the system message only when supports_functions is True (tool-calling mode).
        # Mirrors execution_instructions: profiles/local models can set this to False to suppress it.
        self.tool_calling_instructions = _TOOL_CALLING_INSTRUCTIONS

        # Optional settings
        self.context_window = None
        self.max_tokens = None
        self.api_base = None
        self.api_key = None
        self.api_version = None
        self._is_loaded = False

        # Sanitize secrets (API keys, passwords) from messages before sending to API LLMs.
        # "auto" = sanitize for remote models, skip for local (ollama/local/jan). "on"/"off" override.
        self.sanitize_secrets = "auto"

        # Budget manager powered by LiteLLM
        self.max_budget = None

        # Cache-aware truncation: when the prompt outgrows the context window,
        # drop a *variable* number of whole turns down to `retention_ratio` of
        # the budget (0.8 keeps 80% of the window, dropping the oldest 20%).
        # Because the cut overshoots the limit by the retention margin and lands
        # on a complete-turn boundary, the prefix stays stable for several
        # consecutive turns and the provider's KV prefix cache stays warm —
        # unlike a per-turn sliding window which busts the cache every call.
        # None (default) falls back to tokentrim's sliding-window behaviour.
        # See: https://github.com/character-ai/prompt-poet#cache-aware-truncation-explained
        self.retention_ratio = None

        # Reasoning settings: Use include_reasoning=True to request reasoning tokens
        # (delta.reasoning_content) from OpenRouter/LiteLLM. Use reasoning_effort
        # ("low", "medium", "high") to control the intensity of thinking for OpenAI o1/o3 models.
        self.include_reasoning = None  # None (auto), True, or False
        self.reasoning_effort = None  # "low", "medium", "high"

        # Filled from the final streaming chunk when the API sends usage (see stream_usage.record_stream_chunk_usage).
        self.last_completion_usage = None

    def run(self, messages, *, auxiliary_title_request=False):
        """
        We're responsible for formatting the call into the llm.completions object,
        starting with LMC messages in interpreter.messages, going to OpenAI compatible messages into the llm,
        respecting whether it's a vision or function model, respecting its context window and max tokens, etc.

        And then processing its output, whether it's a function or non function calling model, into LMC format.

        auxiliary_title_request: one-off naming completion — text path only, tight max_tokens,
        no reasoning request, LiteLLM timeout, and no code-execution system suffix on the prompt.
        """
        litellm = _litellm()
        import tokentrim as tt

        if not self._is_loaded:
            self.load()

        if self.max_tokens is not None and self.context_window is not None and self.max_tokens > self.context_window:
            print(
                "Warning: max_tokens is larger than context_window. Setting max_tokens to be 0.2 times the context_window."
            )
            self.max_tokens = int(0.2 * self.context_window)

        # Assertions
        assert messages[0]["role"] == "system", "First message must have the role 'system'"
        for msg in messages[1:]:
            assert msg["role"] != "system", "No message after the first can have the role 'system'"

        model = self.model
        if model in [
            "claude-3.5",
            "claude-3-5",
            "claude-3.5-sonnet",
            "claude-3-5-sonnet",
        ]:
            model = "claude-sonnet-4-6"
            self.model = "claude-sonnet-4-6"
        # Setup our model endpoint
        if model == "i":
            model = "openai/i"
            if not hasattr(self.interpreter, "conversation_id"):  # Only do this once
                self.context_window = 7000
                self.api_key = "x"
                self.max_tokens = 1000
                self.api_base = "https://api.openinterpreter.com/v0"
                self.interpreter.conversation_id = str(uuid.uuid4())

        # Detect function support
        if self.supports_functions == None:
            try:
                if litellm.supports_function_calling(model):
                    self.supports_functions = True
                else:
                    self.supports_functions = False
            except:
                self.supports_functions = False

        # Detect vision support
        if self.supports_vision == None:
            try:
                if litellm.supports_vision(model):
                    self.supports_vision = True
                else:
                    # LiteLLM's registry can lag behind provider releases (e.g.
                    # openrouter/qwen/qwen3.7-plus is not listed yet). OpenRouter's
                    # model list is authoritative for modalities, so consult it
                    # before concluding the model can't receive images.
                    self.supports_vision = self._openrouter_supports_vision(model)
            except:
                self.supports_vision = False

        # Trim image messages if they're there
        image_messages = [msg for msg in messages if msg["type"] == "image"]
        if self.supports_vision:
            if self.interpreter.os:
                # Keep only the last two images if the interpreter is running in OS mode
                if len(image_messages) > 1:
                    for img_msg in image_messages[:-2]:
                        messages.remove(img_msg)
                        if self.interpreter.verbose:
                            print("Removing image message!")
            else:
                # Delete all the middle ones (leave only the first and last 2 images) from messages_for_llm
                if len(image_messages) > 3:
                    for img_msg in image_messages[1:-2]:
                        messages.remove(img_msg)
                        if self.interpreter.verbose:
                            print("Removing image message!")
                # Idea: we could set detail: low for the middle messages, instead of deleting them
        elif self.supports_vision == False and self.vision_renderer:
            for img_msg in image_messages:
                if img_msg["format"] != "description":
                    self.interpreter.display_message("\n  *Viewing image...*\n")

                    if img_msg["format"] == "path":
                        precursor = f"The image I'm referring to ({img_msg['content']}) contains the following: "
                        if self.interpreter.toolbox.import_toolbox_api:
                            postcursor = f"\nIf you want to ask questions about the image, run `toolbox.vision.query(path='{img_msg['content']}', query='(ask any question here)')` and a vision AI will answer it."
                        else:
                            postcursor = ""
                    else:
                        precursor = "Imagine I have just shown you an image with this description: "
                        postcursor = ""

                    try:
                        image_description = self.vision_renderer(lmc=img_msg)
                        ocr = self.interpreter.toolbox.vision.ocr(lmc=img_msg)

                        # It would be nice to format this as a message to the user and display it like: "I see: image_description"

                        img_msg["content"] = (
                            precursor
                            + image_description
                            + "\n---\nI've OCR'd the image, this is the result (this may or may not be relevant. If it's not relevant, ignore this): '''\n"
                            + ocr
                            + "\n'''"
                            + postcursor
                        )
                        img_msg["format"] = "description"
                        if img_msg.get("role") == "computer":
                            img_msg["role"] = "user"

                    except ImportError:
                        # For toolbox-generated images, use a simple message instead of showing installation prompt
                        if img_msg.get("role") == "computer":
                            img_msg["format"] = "description"
                            img_msg["content"] = "An image was generated by the code execution."
                            img_msg["role"] = "user"
                        else:
                            print("\nTo use local vision, run `pip install 'open-interpreter[local]'`.\n")
                            img_msg["format"] = "description"
                            img_msg["content"] = ""

        # Convert to OpenAI messages format
        messages = convert_to_openai_messages(
            messages,
            function_calling=self.supports_functions,
            vision=self.supports_vision,
            shrink_images=self.interpreter.shrink_images,
            interpreter=self.interpreter,
        )

        system_message = messages[0]["content"]
        messages = messages[1:]

        # Trim messages
        try:
            if self.retention_ratio and self.context_window:
                # Cache-aware truncation: when the prompt outgrows the window,
                # drop a variable number of whole turns down to `retention_ratio`
                # of the budget so the prefix stays stable for the next several
                # turns and the provider's KV prefix cache stays warm — unlike a
                # per-turn sliding window which invalidates the cache on every
                # call once the context fills up.
                token_limit = self.context_window - (self.max_tokens or 0) - 25
                messages = cache_aware_trim(
                    messages,
                    system_message=system_message,
                    token_limit=token_limit,
                    retention_ratio=self.retention_ratio,
                    model=model,
                )
            elif self.context_window and self.max_tokens:
                trim_to_be_this_many_tokens = self.context_window - self.max_tokens - 25  # arbitrary buffer
                messages = tt.trim(
                    messages,
                    system_message=system_message,
                    max_tokens=trim_to_be_this_many_tokens,
                )
            elif self.context_window and not self.max_tokens:
                # Just trim to the context window if max_tokens not set
                messages = tt.trim(
                    messages,
                    system_message=system_message,
                    max_tokens=self.context_window,
                )
            else:
                try:
                    messages = tt.trim(messages, system_message=system_message, model=model)
                except:
                    if len(messages) == 1:
                        if self.interpreter.in_terminal_interface:
                            self.interpreter.display_message(
                                """
**We were unable to determine the context window of this model.** Defaulting to 8000.

If your model can handle more, run `interpreter --context_window {token limit} --max_tokens {max tokens per response}`.

Continuing...
                            """
                            )
                        else:
                            self.interpreter.display_message(
                                """
**We were unable to determine the context window of this model.** Defaulting to 8000.

If your model can handle more, run `self.context_window = {token limit}`.

Also please set `self.max_tokens = {max tokens per response}`.

Continuing...
                            """
                            )
                    messages = tt.trim(messages, system_message=system_message, max_tokens=8000)
        except:
            # If we're trimming messages, this won't work.
            # If we're trimming from a model we don't know, this won't work.
            # Better not to fail until `messages` is too big, just for frustrations sake, I suppose.

            # Reunite system message with messages
            messages = [{"role": "system", "content": system_message}] + messages

            pass

        # If there should be a system message, there should be a system message!
        # Empty system messages appear to be deleted :(
        if system_message == "":
            if messages[0]["role"] != "system":
                messages = [{"role": "system", "content": system_message}] + messages

        if should_sanitize_for_model(model, self.sanitize_secrets):
            sanitize_messages(messages)

        ## Start forming the request

        params = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        # OpenAI-compatible: final stream chunk may include usage (prompt/completion/cached breakdown).
        stream_options = {"include_usage": True}

        # Reasoning tokens: Some models support separate reasoning content
        # Set to {"exclude": True} to disable, or remove this line to allow reasoning tokens
        # params["reasoning"] = {"exclude": True}

        # OpenRouter provider preferences: Ensure consistent behavior across providers
        # For OpenRouter models, require providers to support all parameters to avoid inconsistency
        if model.startswith("deepseek/"):
            # LiteLLM maps include_reasoning / reasoning_effort to DeepSeek's thinking API.
            if getattr(litellm, "supports_reasoning", None) and litellm.supports_reasoning(model=model or self.model):
                if self.include_reasoning is not False:
                    params["include_reasoning"] = True if self.include_reasoning is None else self.include_reasoning
                    stream_options["include_reasoning"] = params["include_reasoning"]

        if model.startswith("openrouter/"):
            params["provider"] = {
                "require_parameters": True,  # Only use providers that support all request parameters
                "allow_fallbacks": False,  # Disable fallbacks to maintain consistency
            }
            # Request streaming reasoning when the model supports it, so LiteLLM forwards
            # delta.reasoning_content from OpenRouter. Requires LiteLLM v1.63.5+ (BerriAI/litellm#8631).
            # Only set for models that support reasoning to avoid 400 on non-reasoning OpenRouter models.
            if getattr(litellm, "supports_reasoning", None) and litellm.supports_reasoning(model=model or self.model):
                # We use extra_body to pass reasoning parameters for OpenRouter models.
                # This ensures the tokens are requested via both legacy and modern API formats.
                params["extra_body"] = params.get("extra_body", {})
                params["extra_body"]["include_reasoning"] = True
                params["extra_body"]["reasoning"] = {"enabled": True}

                # Also set standard params for LiteLLM to handle unified mapping
                params["include_reasoning"] = True
                stream_options["include_reasoning"] = True

        # Override reasoning settings if explicitly set on interpreter.llm
        if self.include_reasoning is not None:
            # Some OpenRouter endpoints (e.g. z-ai/glm-5.3-flash) mandate reasoning:
            # sending reasoning.enabled:false there returns 400 "Reasoning is
            # mandatory for this endpoint and cannot be disabled". OpenRouter's
            # model metadata flags this, so refuse to send the disable rather than
            # fail the request. The model then uses its default effort (max for GLM).
            _reasoning_mandatory = False
            if model.startswith("openrouter/"):
                _reasoning_mandatory = bool(
                    ((self._openrouter_model_entry(model) or {}).get("reasoning") or {}).get("mandatory")
                )
            if self.include_reasoning is False and _reasoning_mandatory:
                if model not in _warned_mandatory_reasoning:
                    self.interpreter.display_message(
                        f"> **Note:** `{model}` always reasons and cannot have reasoning "
                        "disabled, so `include_reasoning: false` is being ignored."
                    )
                    _warned_mandatory_reasoning.add(model)
            else:
                params["include_reasoning"] = self.include_reasoning
                stream_options["include_reasoning"] = self.include_reasoning
                if model.startswith("openrouter/"):
                    params["extra_body"] = params.get("extra_body", {})
                    params["extra_body"]["include_reasoning"] = self.include_reasoning
                    params["extra_body"]["reasoning"] = {"enabled": self.include_reasoning}

        # A reasoning_effort only makes sense when reasoning is enabled; sending
        # one alongside include_reasoning=false is contradictory and some backends
        # reject it, so skip it whenever the caller disabled reasoning explicitly.
        if self.reasoning_effort and self.include_reasoning is not False:
            # Guard against sending an unsupported effort level. GLM 5.3 only
            # accepts low/high/max and 400s on anything else (including "medium").
            _effort_ok = True
            if model.startswith("openrouter/"):
                _supported = ((self._openrouter_model_entry(model) or {}).get("reasoning") or {}).get(
                    "supported_efforts"
                )
                if _supported and self.reasoning_effort not in _supported:
                    _effort_ok = False
                    if model not in _warned_unsupported_effort:
                        self.interpreter.display_message(
                            f"> **Note:** `{model}` only supports reasoning_effort "
                            f"{_supported}, so `{self.reasoning_effort}` is being ignored."
                        )
                        _warned_unsupported_effort.add(model)
            if _effort_ok:
                params["reasoning_effort"] = self.reasoning_effort
                if model.startswith("openrouter/"):
                    params["extra_body"] = params.get("extra_body", {})
                    if "reasoning" not in params["extra_body"]:
                        params["extra_body"]["reasoning"] = {}
                    params["extra_body"]["reasoning"]["effort"] = self.reasoning_effort

        params["stream_options"] = stream_options

        # Optional inputs
        if self.api_key:
            params["api_key"] = self.api_key
        if self.api_base:
            params["api_base"] = self.api_base
        if self.api_version:
            params["api_version"] = self.api_version
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        if self.temperature:
            params["temperature"] = self.temperature

        # Ollama sizes its KV cache from `num_ctx`, which Open Interpreter never
        # sent, so the model ran at the server's own default context regardless of
        # `context_window` - while OI trimmed messages to `context_window`. When
        # `context_window` is the larger of the two the request silently overflows
        # what the server actually allocated. Send both so they agree.
        if self.context_window and re.match(r"^ollama(_chat)?/", model or ""):
            params.setdefault("num_ctx", self.context_window)
        if hasattr(self.interpreter, "conversation_id"):
            params["conversation_id"] = self.interpreter.conversation_id

        if auxiliary_title_request:
            # Avoid huge reasoning streams and hung streams with no wall clock (OpenRouter etc.).
            # Not a litellm re-import — this path still uses the same completions() generator.
            params["timeout"] = 120
            # OpenRouter (and other providers) thinking/reasoning models consume max_tokens
            # with internal reasoning tokens even when reasoning streaming is disabled.
            # With max_tokens=128, finish_reason='length' was observed with zero visible output
            # (all tokens consumed by internal thinking) for deepseek/deepseek-v4-flash.
            # litellm.supports_reasoning() does not detect all reasoning models reliably, so
            # we can't use it to branch here. A large cap is safe for all models: non-reasoning
            # models stop at "stop" well within this budget; reasoning models need the headroom
            # for thinking tokens before the title is produced.
            params["max_tokens"] = min(int(params.get("max_tokens") or 2048), 2048)
            params["include_reasoning"] = False
            params.pop("reasoning_effort", None)
            if "stream_options" in params:
                params["stream_options"].pop("include_reasoning", None)
            if model.startswith("openrouter/") and params.get("extra_body"):
                eb = params["extra_body"]
                eb.pop("include_reasoning", None)
                eb.pop("reasoning", None)
            params["skip_execution_instructions"] = True
            # ``run_text_llm`` treats ```...``` as executable code blocks and can yield zero
            # chunks when the model returns a one-line fenced title (no newline after the
            # fence). Plain streaming maps ``delta.content`` directly to messages for
            # ``%rename`` / auto-title only.
            params["conversation_title_plain_stream"] = True

        # Set some params directly on LiteLLM
        if self.max_budget:
            litellm.max_budget = self.max_budget
        # NOTE: We don't enable litellm.set_verbose here because it causes massive output
        # showing every single streaming chunk. Open Interpreter's verbose mode should
        # only show our own debug messages, not LiteLLM's internal logging.
        # if self.interpreter.verbose:
        #     litellm.set_verbose = True

        if (
            self.interpreter.debug == True and False  # DISABLED
        ):  # debug will equal "server" if we're debugging the server specifically
            print("\n\n\nOPENAI COMPATIBLE MESSAGES:\n\n\n")
            for message in messages:
                if len(str(message)) > 5000:
                    print(str(message)[:200] + "...")
                else:
                    print(message)
                print("\n")
            print("\n\n\n")

        if self.supports_functions and not auxiliary_title_request:
            # yield from run_function_calling_llm(self, params)
            try:
                yield from run_tool_calling_llm(self, params)
            except FunctionCallingNotSupportedError as e:
                # Model doesn't support function calling, fall back to text mode
                message = (
                    "Model doesn't support function calling, falling back to text mode.\n\n"
                    "Tip: Use `--no-llm_supports_functions` to skip function calling and avoid this message."
                )
                panel = Panel(message, border_style="yellow", title="Warning", title_align="left")
                rich_print(panel)
                print("")  # Add space after message
                self.supports_functions = False
                # Re-convert messages for text mode
                messages = convert_to_openai_messages(
                    self.interpreter.messages,
                    function_calling=False,
                    vision=self.supports_vision,
                    shrink_images=self.interpreter.shrink_images,
                    interpreter=self.interpreter,
                )
                params["messages"] = messages
                # Remove tools parameter if present (it was added by run_tool_calling_llm)
                params.pop("tools", None)
                yield from run_text_llm(self, params)
        else:
            yield from run_text_llm(self, params)
        # Let ModelNotFoundError and AccessDeniedError bubble up to respond.py for proper handling

    # If you change model, set _is_loaded to false
    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
        self._is_loaded = False

    def _openrouter_model_entry(self, model):
        return openrouter_model_entry(model, verbose=self.interpreter.verbose)

    def _openrouter_supports_vision(self, model):
        """
        Returns True if an openrouter model accepts image input.

        Consulted when LiteLLM's registry can't confirm vision support, since the
        registry often lags provider releases. Falls back to False if the model
        list can't be fetched.
        """
        entry = self._openrouter_model_entry(model)
        if entry is None:
            return False
        modalities = (entry.get("architecture") or {}).get("input_modalities") or []
        return "image" in modalities

    def load(self):
        if self._is_loaded:
            return
        configure(self)
