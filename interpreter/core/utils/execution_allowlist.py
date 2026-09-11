import os
import re

import yaml

from ...terminal_interface.utils.oi_dir import oi_dir

AUTO_RUN_MODES = ("prompt", "all", "allowlist", "denylist")

SHELL_LANGUAGES = frozenset({"bash", "shell", "sh"})

DEFAULT_ALLOWLIST_FILE = os.path.join(oi_dir, "allowlist.yaml")
DEFAULT_DENYLIST_FILE = os.path.join(oi_dir, "denylist.yaml")

# PoC builtin preset — exact match only (see plan: v2 AST, v3 safe-chains).
BUILTIN_STRICT_RULES = [
    {"language": "bash", "match": "exact", "pattern": "ls"},
    {"language": "shell", "match": "exact", "pattern": "ls"},
    {"language": "cmd", "match": "exact", "pattern": "dir"},
    {"language": "python", "match": "exact", "pattern": "help(os)"},
]

# Denylist mode inverts the allowlist: everything runs unmatched, and only these
# stop for confirmation. That is fail-OPEN by construction — a command nobody
# anticipated runs without asking. It exists for headless/yolo operation where
# an allowlist is impractical; the trade is deliberate.
#
# Exact matching is useless here (you cannot enumerate every spelling of a
# destructive command), so these are regexes, matched case-insensitively
# against the whole code block.
BUILTIN_DENY_RULES = [
    # rm -rf against a root-ish target: /, ~, $HOME, or a bare glob
    {
        "language": "bash",
        "match": "regex",
        "pattern": r"\brm\s+(-[a-z]*\s+)*-[a-z]*r[a-z]*f|\brm\s+(-[a-z]*\s+)*-[a-z]*f[a-z]*r",
    },
    # Filesystem creation / raw device writes
    {"language": "bash", "match": "regex", "pattern": r"\bmkfs(\.\w+)?\b"},
    {"language": "bash", "match": "regex", "pattern": r"\bdd\b[^\n]*\bof=\s*/dev/"},
    {"language": "bash", "match": "regex", "pattern": r">\s*/dev/(sd[a-z]|nvme\d|vd[a-z])"},
    # Recursive ownership/permission changes rooted at /
    {"language": "bash", "match": "regex", "pattern": r"\bch(mod|own)\s+(-[a-z]*\s+)*-[a-z]*R[a-z]*\s+[^\n]*\s/(\s|$)"},
    # Fork bomb
    {"language": "bash", "match": "regex", "pattern": r":\s*\(\s*\)\s*\{.*\|.*&.*\}\s*;\s*:"},
    # Disk/partition table rewrites
    {"language": "bash", "match": "regex", "pattern": r"\b(fdisk|parted|sgdisk|wipefs)\b[^\n]*/dev/"},
    # Python equivalents of the above
    {"language": "python", "match": "regex", "pattern": r"\bshutil\.rmtree\s*\("},
    {"language": "python", "match": "regex", "pattern": r"\bos\.(remove|unlink|rmdir|removedirs)\s*\("},
]


def normalize_auto_run_mode(value):
    if value is True or value == "all" or value == "true":
        return "all"
    if value is False or value == "prompt" or value == "false" or value is None:
        return "prompt"
    if value == "allowlist":
        return "allowlist"
    if value == "denylist":
        return "denylist"
    raise ValueError(
        f"Invalid auto_run mode: {value!r}. "
        "Expected prompt, all, allowlist, denylist, or a boolean."
    )


def _expand_path(path):
    if not path:
        return path
    return os.path.expanduser(path)


def _rule_key(rule):
    return (
        rule.get("language", "").lower(),
        rule.get("match", ""),
        rule.get("pattern", ""),
    )


def _languages_compatible(rule_language, code_language):
    rule_language = (rule_language or "").lower()
    code_language = (code_language or "").lower()
    if rule_language == code_language:
        return True
    if rule_language in SHELL_LANGUAGES and code_language in SHELL_LANGUAGES:
        return True
    return False


def match_exact(code, pattern):
    return code.strip() == pattern


def match_regex(code, pattern):
    """
    Search (not fullmatch) the whole block, case-insensitively.

    Denylist rules have to fire on a dangerous command buried inside a larger
    script, so this deliberately matches anywhere in the text.
    """
    return re.search(pattern, code, re.IGNORECASE | re.MULTILINE) is not None


MATCHERS = {"exact": match_exact, "regex": match_regex}


def _validate_rule(rule, source="allowlist"):
    if not isinstance(rule, dict):
        raise ValueError(f"Invalid rule in {source}: expected mapping, got {type(rule)}")
    match_type = rule.get("match")
    if match_type not in MATCHERS:
        raise ValueError(
            f"Unsupported match type {match_type!r} in {source}. "
            f"Supported: {', '.join(sorted(MATCHERS))}."
        )
    if not rule.get("language"):
        raise ValueError(f"Rule in {source} missing 'language'")
    if "pattern" not in rule:
        raise ValueError(f"Rule in {source} missing 'pattern'")
    if match_type == "regex":
        try:
            re.compile(rule["pattern"])
        except re.error as error:
            raise ValueError(
                f"Invalid regex in {source}: {rule['pattern']!r} ({error})"
            ) from error


def _rule_matches(rule, language, code):
    if not _languages_compatible(rule["language"], language):
        return False
    return MATCHERS[rule["match"]](code, rule["pattern"])


def _load_rules_from_file(path):
    path = _expand_path(path)
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Rule file must be a YAML mapping: {path}")
    rules = data.get("rules") or []
    if not isinstance(rules, list):
        raise ValueError(f"'rules' in rule file must be a list: {path}")
    for rule in rules:
        _validate_rule(rule, source=path)
    return rules


def load_allowlist_rules(interpreter):
    rules = []
    seen = set()

    def add_rules(rule_list):
        for rule in rule_list:
            _validate_rule(rule, "allowlist")
            key = _rule_key(rule)
            if key in seen:
                continue
            seen.add(key)
            rules.append(dict(rule))

    if interpreter.auto_run_mode != "allowlist":
        return rules

    if not getattr(interpreter, "auto_run_allowlist_replace_builtin", False):
        add_rules(BUILTIN_STRICT_RULES)

    profile_rules = getattr(interpreter, "auto_run_allowlist_rules", None) or []
    add_rules(profile_rules)

    allowlist_file = getattr(interpreter, "auto_run_allowlist_file", None) or DEFAULT_ALLOWLIST_FILE
    add_rules(_load_rules_from_file(allowlist_file))

    session_rules = getattr(interpreter, "_session_allowlist_rules", None) or []
    add_rules(session_rules)

    return rules


def is_execution_allowlisted(interpreter, language, code):
    if interpreter.auto_run_mode != "allowlist":
        return False
    for rule in load_allowlist_rules(interpreter):
        if _rule_matches(rule, language, code):
            return True
    return False


def load_denylist_rules(interpreter):
    """
    Built-ins first, then profile rules, then the denylist file.

    Unlike the allowlist there is no session-rules tier: "allow this once" is a
    safe thing to accumulate at runtime, "stop blocking this" is not.
    """
    rules = []
    seen = set()

    def add_rules(rule_list, source):
        for rule in rule_list:
            _validate_rule(rule, source)
            key = _rule_key(rule)
            if key in seen:
                continue
            seen.add(key)
            rules.append(dict(rule))

    if interpreter.auto_run_mode != "denylist":
        return rules

    if not getattr(interpreter, "auto_run_denylist_replace_builtin", False):
        add_rules(BUILTIN_DENY_RULES, "builtin denylist")

    profile_rules = getattr(interpreter, "auto_run_denylist_rules", None) or []
    add_rules(profile_rules, "denylist")

    denylist_file = getattr(interpreter, "auto_run_denylist_file", None) or DEFAULT_DENYLIST_FILE
    add_rules(_load_rules_from_file(denylist_file), denylist_file)

    return rules


def is_execution_denied(interpreter, language, code):
    if interpreter.auto_run_mode != "denylist":
        return False
    for rule in load_denylist_rules(interpreter):
        if _rule_matches(rule, language, code):
            return True
    return False


def should_require_execution_confirmation_for_code(interpreter, language, code):
    if interpreter.auto_run_mode == "all":
        return False
    if interpreter.auto_run_mode == "denylist":
        # Inverted: confirm only what a rule catches.
        return is_execution_denied(interpreter, language, code)
    if is_execution_allowlisted(interpreter, language, code):
        return False
    return True


def _execution_target_from_confirmation_chunk(chunk):
    if chunk.get("type") != "confirmation":
        return None, None
    if chunk.get("format") == "edit":
        return None, None
    content = chunk.get("content") or {}
    if chunk.get("format") == "execution" or content.get("type") == "code":
        return content.get("format"), content.get("content")
    return content.get("format"), content.get("content")


def should_require_execution_confirmation(interpreter, chunk):
    if chunk.get("type") != "confirmation":
        return True
    # "all" is the yolo setting and it covers edits too. This check has to come
    # BEFORE the edit check below: edits used to be confirmed unconditionally,
    # which hung any headless run (server mode has no TTY, so nothing could
    # answer the prompt).
    if interpreter.auto_run_mode == "all":
        return False
    if chunk.get("format") == "edit":
        return True
    language, code = _execution_target_from_confirmation_chunk(chunk)
    if language is None:
        return True
    return should_require_execution_confirmation_for_code(interpreter, language, code)


def persist_allowlist_rule(interpreter, language, code):
    rule = {
        "language": language,
        "match": "exact",
        "pattern": code.strip(),
    }
    _validate_rule(rule, "session")

    session_rules = getattr(interpreter, "_session_allowlist_rules", None)
    if session_rules is None:
        interpreter._session_allowlist_rules = []
        session_rules = interpreter._session_allowlist_rules

    key = _rule_key(rule)
    if any(_rule_key(existing) == key for existing in session_rules):
        return rule, False

    session_rules.append(dict(rule))

    allowlist_file = _expand_path(
        getattr(interpreter, "auto_run_allowlist_file", None) or DEFAULT_ALLOWLIST_FILE
    )
    os.makedirs(os.path.dirname(allowlist_file), exist_ok=True)

    file_rules = _load_rules_from_file(allowlist_file)
    if any(_rule_key(existing) == key for existing in file_rules):
        return rule, False

    file_rules.append(dict(rule))
    with open(allowlist_file, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            {
                "rules": file_rules,
            },
            file,
            default_flow_style=False,
            sort_keys=False,
        )

    return rule, True
