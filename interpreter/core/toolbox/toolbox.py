import inspect
import json
import platform
import re

from .actions.actions import Actions
from .ai.ai import Ai
from .ai2 import Ai2
from .browser.browser import Browser
from .calendar.calendar import Calendar
from .clipboard.clipboard import Clipboard
from .contacts.contacts import Contacts
from .display.display import Display
from .docs.docs import Docs
from .files.files import Files
from .keyboard.keyboard import Keyboard
from .mail.mail import Mail
from .mouse.mouse import Mouse
from .os.os import Os
from .sms.sms import SMS
from .vision.vision import Vision
from .web.web import Web


class Toolbox:
    def __init__(self, interpreter):
        self.interpreter = interpreter

        self.offline = False
        self.verbose = False
        self.debug = False

        self.mouse = Mouse(self)
        self.keyboard = Keyboard(self)
        self.display = Display(self)
        self.clipboard = Clipboard(self)
        self.mail = Mail(self)
        self.sms = SMS(self)
        self.calendar = Calendar(self)
        self.contacts = Contacts(self)
        self.browser = Browser(self)
        self.os = Os(self)
        self.web = Web(self)
        self.vision = Vision(self)
        self.docs = Docs(self)
        self.ai = Ai(self)
        self._ai2 = None
        self.files = Files(self)
        self.actions = Actions(self)

        self.emit_images = True
        self.api_base = "https://api.openinterpreter.com/v0"

        self.import_toolbox_api = False  # Defaults to false
        # "names" lists the callables and their required arguments; "full"
        # restores the exact signatures and the one-line descriptions, which
        # cost about three times as many tokens.
        self.api_listing = "names"
        self._has_imported_toolbox_api = False  # Because we only want to do this once

        self.max_output = self.interpreter.max_output  # Should mirror interpreter.max_output

        self._system_message_override = None

    @property
    def ai2(self):
        """Lazily initialize ai2 so LiteLLM setup only happens on first real use."""
        if self._ai2 is None:
            self._ai2 = Ai2(self)
        return self._ai2

    @ai2.setter
    def ai2(self, value):
        self._ai2 = value

    @property
    def system_message(self):
        if self._system_message_override is not None:
            return self._system_message_override

        # Names and required arguments, by default. The full catalogue of exact
        # signatures and descriptions costs about 1,400 tokens in every request,
        # forever, to describe methods a given session mostly will not touch.
        # Everything beyond the argument list is one help() call away at the
        # moment it is needed, and that call returns the live docstring, which
        # cannot go stale the way a baked listing can. The argument list itself
        # could not stay behind help(), because a model shown a bare name does
        # not ask — it invents the arguments and burns a turn on the TypeError.
        if self.api_listing == "full":
            catalogue = "\n".join(self._get_all_toolbox_tools_signature_and_description())
        else:
            catalogue = "\n".join(self._get_all_toolbox_tool_names())

        notes = []
        if platform.system() != "Darwin":
            notes.append("`toolbox.mail`, `toolbox.sms`, `toolbox.calendar` and `toolbox.contacts` are macOS-only.")
        if getattr(self.interpreter.llm, "supports_vision", None) is False:
            notes.append(
                "The `view_image` tool is unavailable on this non-vision model; "
                "use `toolbox.vision.query(path=..., query='...')` to describe an image."
            )
        note_block = ("\n" + "\n".join(f"Note: {n}" for n in notes) + "\n") if notes else ""

        # Actions are described only when some exist. A session with none pays
        # nothing, which is the point of keeping them out of the catalogue: the
        # long tail of specific capabilities should not be a standing cost in
        # every request the way a toolbox method is.
        available_actions = self.actions.list()
        if available_actions:
            listed = "\n".join(f"{a['name']} — {a['summary']}" for a in available_actions)
            action_block = f"""

### Actions

Project-specific Python modules, loaded only when used. Read one with
`toolbox.actions.show(name)`; load it with `toolbox.actions.load(name)`, which
returns the module and runs nothing until you call one of its functions.

```
{listed}
```
"""
        else:
            action_block = ""

        return f"""
## The `toolbox` API

`toolbox` is already a variable in your Python namespace. Never import it.

```python
{catalogue}
```
{note_block}
`...` stands for optional arguments. Call `help(toolbox.module.method)` for
those, the return shape and examples. Never guess a signature or a return format.
{action_block}""".strip()

    @system_message.setter
    def system_message(self, value):
        self._system_message_override = value

    # Backward compatibility: allow profiles to use import_computer_api
    @property
    def import_computer_api(self):
        return self.import_toolbox_api

    @import_computer_api.setter
    def import_computer_api(self, value):
        self.import_toolbox_api = value

    # Shortcut for interpreter.terminal.languages
    @property
    def languages(self):
        return self.interpreter.terminal.languages

    @languages.setter
    def languages(self, value):
        self.interpreter.terminal.languages = value

    def _get_all_toolbox_tools_list(self):
        tools = [
            self.mouse,
            self.keyboard,
            self.display,
            self.clipboard,
            self.browser,
            self.os,
            self.web,
            self.vision,
            self.docs,
            self.ai,
            self.ai2,
            self.files,
        ]
        if platform.system() == "Darwin":
            tools = tools[:4] + [self.mail, self.sms, self.calendar, self.contacts] + tools[4:]
        return tools

    def _get_all_toolbox_tool_names(self):
        """Every callable as `toolbox.module.method(required, ...)`, with no prose.

        The part of the catalogue a model cannot reconstruct for itself: the
        name, and what it has to pass. Defaults, types and return shapes stay
        behind help(), where they cannot go stale; what could not stay there is
        the argument list, because a model that has never been shown one
        invents it instead of asking — `toolbox.os.notify(title=..., message=...)`,
        borrowed from plyer, for a method that takes one positional string.
        """
        names = []
        for tool in self._get_all_toolbox_tools_list():
            for method in self._extract_tool_info(tool)["methods"]:
                names.append(method["brief"])
        return names

    def _get_all_toolbox_tools_signature_and_description(self):
        """
        This function returns a list of all the toolbox tools that are available with their signature and description from the function docstrings.
        for example:
        toolbox.browser.search(query) # Searches the web for the specified query and returns the results.
        toolbox.calendar.create_event(title: str, start_date: datetime.datetime, end_date: datetime.datetime, location: str = "", notes: str = "", calendar: str = None) -> str # Creates a new calendar event in the default calendar with the given parameters using AppleScript.
        """
        tools = self._get_all_toolbox_tools_list()
        tools_signature_and_description = []
        for tool in tools:
            tool_info = self._extract_tool_info(tool)

            # Add module docstring as section header if available
            if tool_info.get("module_doc"):
                tools_signature_and_description.append(f"\n# {tool.__class__.__name__}: {tool_info['module_doc']}")

            for method in tool_info["methods"]:
                # Format as tool_signature # tool_description (first line only)
                formatted_info = f"{method['signature']} # {method['description']}"
                tools_signature_and_description.append(formatted_info)
        return tools_signature_and_description

    def _extract_tool_info(self, tool):
        """
        Helper function to extract the signature and description of a tool's methods.
        """
        tool_info = {"signature": tool.__class__.__name__, "methods": []}

        # Extract module docstring (first line only)
        if tool.__class__.__module__:
            try:
                import sys

                module = sys.modules.get(tool.__class__.__module__)
                if module and module.__doc__:
                    # Get first line of module docstring
                    first_line = module.__doc__.strip().split("\n")[0].strip()
                    if first_line:
                        tool_info["module_doc"] = first_line
            except:
                pass
        module = tool.__class__.__name__.lower()
        methods, properties = self._public_members(tool)

        for name, method in methods:
            if self._is_hidden_from_listing(f"{module}.{name}", method.__doc__):
                continue
            qualified = f"toolbox.{module}.{name}"
            tool_info["methods"].append(
                {
                    "signature": qualified + self._render_parameters(method),
                    "brief": qualified + self._render_required_parameters(method),
                    "description": self._get_first_line(method.__doc__),
                }
            )

        # ------------------------------------------------------------------
        # Include read-only @property attributes (e.g., ai2.available_models)
        # ------------------------------------------------------------------
        for name, prop in properties:
            if self._is_hidden_from_listing(f"{module}.{name}", prop.fget.__doc__):
                continue
            qualified = f"toolbox.{module}.{name}"
            tool_info["methods"].append(
                {
                    "signature": qualified,
                    "brief": qualified,
                    "description": self._get_first_line(prop.fget.__doc__),
                }
            )
        return tool_info

    @staticmethod
    def _public_members(tool):
        """The public methods and properties of a tool, without evaluating any.

        Attributes are read off the class with getattr_static, so a lazy
        property is never triggered merely to be catalogued: `browser.driver`
        would otherwise launch Chrome every time the system message is built.
        That hazard is why Browser used to have its own branch here, which
        built signatures from `co_varnames` and so advertised `self` as the
        first parameter of every browser method.
        """
        methods, properties = [], []
        for name in sorted(dir(tool.__class__)):
            if name.startswith("_"):
                continue
            attr = inspect.getattr_static(tool, name, None)
            if isinstance(attr, property):
                properties.append((name, attr))
            elif inspect.isfunction(attr) and not hasattr(attr, "__wrapped__"):
                methods.append((name, getattr(tool, name)))
        return methods, properties

    # Callables that exist for the implementation rather than for the model:
    # lazy bootstraps that every real call already performs for itself. Their
    # docstrings carry no marker, so they are named here. Anything whose
    # docstring opens with "[INTERNAL" or "DEPRECATED" is dropped without
    # needing an entry.
    _LISTING_EXCLUDED = frozenset({"browser.driver", "browser.setup", "vision.load", "ai2.client"})

    @classmethod
    def _is_hidden_from_listing(cls, qualified_name, docstring):
        """Advertising a method the model must not call costs a turn and tokens."""
        if qualified_name in cls._LISTING_EXCLUDED:
            return True
        first_line = (docstring or "").strip()
        return first_line.startswith("[INTERNAL") or first_line.startswith("DEPRECATED")

    @staticmethod
    def _render_parameters(method):
        """The exact signature, `*args` and `**kwargs` included.

        Those two were dropped, which hid the *primary* argument of several
        tools: `keyboard.hotkey(*args, interval=0.1)` listed as
        `hotkey(interval=0.1)` gives a model no way to pass the keys, and
        `mouse.click` no way to pass the target. A signature with the only
        useful parameter missing is worse than no signature, because it looks
        complete.
        """
        rendered = []
        for param in inspect.signature(method).parameters.values():
            if param.kind == param.VAR_POSITIONAL:
                rendered.append(f"*{param.name}")
            elif param.kind == param.VAR_KEYWORD:
                rendered.append(f"**{param.name}")
            elif param.default == param.empty:
                rendered.append(param.name)
            else:
                rendered.append(f"{param.name}={param.default!r}")
        return "(" + ", ".join(rendered) + ")"

    @staticmethod
    def _render_required_parameters(method):
        """Just enough of the signature to stop a model guessing the rest.

        Required parameters and `*args` are named; everything optional
        (defaults and `**kwargs`) collapses to `...`, which says "there is
        more, ask help()" without paying per default value in every request.
        So `toolbox.os.notify(text)`, `toolbox.keyboard.hotkey(*args, ...)`,
        `toolbox.calendar.create_event(title, start_date, end_date, ...)` —
        each of those was being called wrongly for want of this one line.
        """
        rendered = []
        has_optional = False
        for param in inspect.signature(method).parameters.values():
            if param.kind == param.VAR_POSITIONAL:
                rendered.append(f"*{param.name}")
            elif param.kind == param.VAR_KEYWORD or param.default != param.empty:
                has_optional = True
            else:
                rendered.append(param.name)
        if has_optional:
            rendered.append("...")
        return "(" + ", ".join(rendered) + ")"

    def _get_first_line(self, docstring):
        """One short sentence for the API listing; help() has the rest."""
        if not docstring:
            return ""
        first_line = docstring.strip().split("\n\n")[0].split("\n")[0].strip()
        # The listing is re-sent with every request, so keep it to the first
        # sentence. Anything further (parameters, return shape, examples) is a
        # help(toolbox.module.method) away, which the prompt tells the model to use.
        sentence = re.split(r"(?<=[.!?])\s", first_line)[0]
        return sentence if len(sentence) >= 12 else first_line

    def _extract_return_format(self, docstring):
        """Extract return format information from docstring Returns: section."""
        if not docstring:
            return ""

        # Look for "Returns:" section
        lines = docstring.split("\n")
        in_returns_section = False
        return_lines = []
        section_keywords = [
            "Example:",
            "Examples:",
            "Args:",
            "Arguments:",
            "Parameters:",
            "Raises:",
            "Note:",
            "Notes:",
            "See also:",
            "See Also:",
        ]

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("Returns:"):
                in_returns_section = True
                # Get the content after "Returns:"
                content = stripped[8:].strip()  # Remove "Returns:"
                if content:
                    return_lines.append(content)
                continue

            if in_returns_section:
                # Stop at empty line if we already have content
                if stripped == "":
                    if return_lines:
                        break
                    continue
                # Check if this is a new section
                for keyword in section_keywords:
                    if stripped.startswith(keyword):
                        break
                else:
                    # Not a new section, continue collecting
                    if stripped:
                        return_lines.append(stripped)
                        continue
                # Found a new section, stop
                break

        if return_lines:
            return " ".join(return_lines).strip()
        return ""

    def run(self, *args, **kwargs):
        """
        Shortcut for interpreter.terminal.run
        """
        return self.interpreter.terminal.run(*args, **kwargs)

    def exec(self, code):
        """
        Shortcut for interpreter.terminal.run("bash", code)
        It has hallucinated this.
        """
        return self.interpreter.terminal.run("bash", code)

    def run_bash(self, code, **kwargs):
        return self.interpreter.terminal.run("bash", code, **kwargs)

    def run_cmd(self, code, **kwargs):
        return self.interpreter.terminal.run("cmd", code, **kwargs)

    def stop(self):
        """
        Shortcut for interpreter.terminal.stop
        """
        return self.interpreter.terminal.stop()

    def terminate(self):
        """
        Shortcut for interpreter.terminal.terminate
        """
        return self.interpreter.terminal.terminate()

    def screenshot(self, *args, **kwargs):
        """
        Shortcut for toolbox.display.screenshot
        """
        return self.display.screenshot(*args, **kwargs)

    def view(self, *args, **kwargs):
        """
        Shortcut for toolbox.display.screenshot
        """
        return self.display.screenshot(*args, **kwargs)

    def to_dict(self):
        def json_serializable(obj):
            try:
                json.dumps(obj)
                return True
            except:
                return False

        return {k: v for k, v in self.__dict__.items() if json_serializable(v)}

    def load_dict(self, data_dict):
        for key, value in data_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
