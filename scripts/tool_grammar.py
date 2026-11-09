"""Tool-call JSON grammar shared by the SA-4 format gate (scratch/sa4_format_check.py) and the SA-7 runtime.

Moved unchanged from scratch/sa4_format_check.py (2026-10-27); see that file's docstring for the rules
(MAX_STR, MAX_RUN, no repeated key or call, MAX_CALLS). gfeed/gstep are pure Python, so the grammar can be
checked on CPU; ToolCallGrammar is the greedy LogitsProcessor built on them. grammar()/ToolCallGrammar take
max_str (default MAX_STR) so the SA-7 agent loop can let write_file carry a whole solution.

schema=True (SA-7 agent, 2026-10-27) also enforces each tool's argument schema from its tool line: the arguments
object holds exactly the listed parameter keys (any order), a "type": "str" parameter takes a string, and
values[(tool, parameter)] narrows a string to a fixed set. Off by default, so the SA-4 format gate is unchanged.
"""
import json
import re

import torch
from transformers import LogitsProcessor

TOOL_LINE = re.compile(r'^# (\{"name": .*\})$')
WS = " \t\n\r"
MAX_CALLS = 8
MAX_DEPTH = 8  # nested objects/arrays inside "arguments"
MAX_NUM = 32  # chars per number literal
POST_WS = 2  # whitespace chars allowed after the closing "]"
MAX_STR = 128  # chars per string (raw, escapes included); SA-4 val refs <= 54, 7 SA-4 train strings longer
MAX_RUN = 16  # same char in a row inside a string; stops "1000000..." runaways
# (container stack, mode, calls) with calls = (calls opened, canonical JSON of closed calls, current call text or None)
START = ((), ("pre",), (0, frozenset(), None))
OBJ0 = ("OBJ", frozenset())  # object frame: keys seen so far
STR_KEY = ("str", True, 0, 0, 0, "", "")  # (kind, is_key, escape phase, length, run, previous char, key text)
STR_VAL = ("str", False, 0, 0, 0, "", "")


def tool_names(prompt: str) -> set[str]:
    names = set()
    for line in prompt.split("\n"):
        match = TOOL_LINE.match(line)
        if match:
            try:
                names.add(json.loads(match.group(1))["name"])
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
    return names


def judge(reply: str, prompt: str, reference: str) -> dict:
    try:
        calls = json.loads(reply)
    except json.JSONDecodeError:
        return {"valid": False, "why": "json"}
    if not (isinstance(calls, list) and calls and all(
            isinstance(c, dict) and set(c) == {"name", "arguments"} and isinstance(c["arguments"], dict)
            for c in calls)):
        return {"valid": False, "why": "schema"}
    if not all(c["name"] in tool_names(prompt) for c in calls):
        return {"valid": False, "why": "unknown_tool"}
    key = lambda cs: sorted(json.dumps(c, sort_keys=True) for c in cs)
    names = lambda cs: sorted(c["name"] for c in cs)
    keys = lambda cs: sorted((c["name"], sorted(c["arguments"])) for c in cs)
    ref = json.loads(reference)
    return {"valid": True, "exact": key(calls) == key(ref), "names": names(calls) == names(ref),
            "keys": keys(calls) == keys(ref)}


def _plain(s) -> bool:
    return isinstance(s, str) and '"' not in s and "\\" not in s


def tool_schema(prompt: str, values: dict | None = None) -> dict:
    """{tool: {parameter: spec}} from the tool lines; spec None = any JSON value, True = any string, frozenset = one
    of these strings. A tool whose parameters cannot be read (or have quotes/backslashes in names) is left out."""
    values = values or {}
    schema = {}
    for line in prompt.split("\n"):
        match = TOOL_LINE.match(line)
        try:
            tool = json.loads(match.group(1)) if match else None
            name, params = tool["name"], tool["parameters"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if not (_plain(name) and isinstance(params, dict) and all(map(_plain, params))):
            continue
        schema[name] = {k: frozenset(values[(name, k)]) if (name, k) in values
                        else True if isinstance(p, dict) and p.get("type") == "str" else None
                        for k, p in params.items()}
    return schema


def grammar(prompt: str, max_str: int = MAX_STR, schema: bool = False, values: dict | None = None) -> tuple:
    """(tool names, their prefixes, max_str, {tool: {parameter: spec}} or None); see the module docstring."""
    names = frozenset(n for n in tool_names(prompt) if _plain(n))
    return (names, frozenset(n[:i] for n in names for i in range(len(n) + 1)), max_str,
            tool_schema(prompt, values) if schema else None)


def _close(stack: tuple) -> tuple:
    stack = stack[:-1]
    return stack, ("call_end",) if stack[-1][0] == "CALL" else ("after",)


def _spec(stack: tuple, gram: tuple):
    """Parameter specs of the call when the top frame is its schema-checked arguments object, else None."""
    return gram[3][stack[-2][1]] if stack[-1][0] == "ARGS" else None


def _value(stack: tuple, ch: str):
    """First char of a JSON value inside "arguments"; returns (stack, mode) or None."""
    if ch == '"':
        return stack, STR_VAL
    if ch in "{[":
        if len(stack) >= 3 + MAX_DEPTH:
            return None
        return (stack + (OBJ0,), ("obj_open",)) if ch == "{" else (stack + ("ARR",), ("arr_open",))
    if ch in "tfn":
        return stack, ("lit", {"t": "rue", "f": "alse", "n": "ull"}[ch], "after")
    if ch == "-":
        return stack, ("num", "minus", 1)
    if ch.isdigit() and ch.isascii():
        return stack, ("num", "zero" if ch == "0" else "int", 1)
    return None


_NUM = {  # phase -> {char class: next phase}; None key = phase may end here
    "minus": {"0": "zero", "d": "int"},
    "zero": {".": "frac0", "e": "exp0", None: True},
    "int": {"0": "int", "d": "int", ".": "frac0", "e": "exp0", None: True},
    "frac0": {"0": "frac", "d": "frac"},
    "frac": {"0": "frac", "d": "frac", "e": "exp0", None: True},
    "exp0": {"+": "expsign", "0": "exp", "d": "exp"},
    "expsign": {"0": "exp", "d": "exp"},
    "exp": {"0": "exp", "d": "exp", None: True},
}
_SIMPLE = {  # mode -> {char: next mode} for the fixed call skeleton (whitespace is skipped in all of these)
    "name_key": {'"': ("lit", 'name"', "name_colon")}, "name_colon": {":": ("name_val",)},
    "name_val": {'"': ("enum", "")}, "name_after": {",": ("args_key",)},
    "args_key": {'"': ("lit", 'arguments"', "args_colon")}, "args_colon": {":": ("args_val",)},
    "obj_key": {'"': STR_KEY}, "obj_colon": {":": ("value",)},
}


def _schar(state: tuple, ch: str, esc: int) -> tuple:
    stack, (_, key, _, n, run, prev, text), ncalls = state
    return stack, ("str", key, esc, n + 1, run + 1 if ch == prev else 1, ch, text + ch if key else ""), ncalls


def gstep(state: tuple, ch: str, gram: tuple):
    """Advance the tool-call grammar by one char; returns the new state, or None if ch is illegal here."""
    new = _gstep(state, ch, gram)
    if new is None or new is state or new[2][2] is None:  # illegal, skipped whitespace, or outside a call
        return new
    opened, seen, text = new[2]
    text += ch
    if new[1] == ("calls_after",):  # ch was the call's closing "}"
        key = json.dumps(json.loads(text), sort_keys=True)
        return None if key in seen else (new[0], new[1], (opened, seen | {key}, None))
    return new[0], new[1], (opened, seen, text)


def _gstep(state: tuple, ch: str, gram: tuple):
    stack, mode, ncalls = state
    kind = mode[0]
    if kind == "lit":
        if ch != mode[1][0]:
            return None
        return stack, ((mode[2],) if len(mode[1]) == 1 else ("lit", mode[1][1:], mode[2])), ncalls
    if kind == "str":
        esc = mode[2]
        spec = _spec(stack, gram) if mode[1] else None
        if esc == 0:
            if ch == '"':
                if not mode[1]:
                    return stack, ("after",), ncalls
                keys = stack[-1][1]
                if mode[6] in keys:
                    return None
                if spec is not None:
                    return (stack[:-1] + (("ARGS", keys | {mode[6]}, mode[6]),), ("obj_colon",), ncalls) \
                        if mode[6] in spec else None
                return stack[:-1] + (("OBJ", keys | {mode[6]}),), ("obj_colon",), ncalls
            if ord(ch) < 0x20 or mode[3] >= gram[2] or (ch == mode[5] and mode[4] >= MAX_RUN):
                return None
            if spec is not None and not any(k.startswith(mode[6] + ch) for k in spec if k not in stack[-1][1]):
                return None  # schema keys are plain, so this also rules out escapes
            return _schar(state, ch, 1 if ch == "\\" else 0)
        if esc == 1:
            if ch in '"\\/bfnrt':
                return _schar(state, ch, 0)
            return _schar(state, ch, 2) if ch == "u" else None
        if ch not in "0123456789abcdefABCDEF":
            return None
        return _schar(state, ch, 0 if esc == 5 else esc + 1)
    if kind == "enum":
        if ch == '"':  # the call frame keeps the tool name for the schema check
            return (stack[:-1] + (("CALL", mode[1]),), ("name_after",), ncalls) if mode[1] in gram[0] else None
        return (stack, ("enum", mode[1] + ch), ncalls) if mode[1] + ch in gram[1] else None
    if kind == "venum":  # a string value restricted to the set in mode[2]
        if ch == '"':
            return (stack, ("after",), ncalls) if mode[1] in mode[2] else None
        text = mode[1] + ch
        return (stack, ("venum", text, mode[2]), ncalls) if any(v.startswith(text) for v in mode[2]) else None
    if kind == "num":
        cls = "0" if ch == "0" else "d" if ch in "123456789" else "e" if ch in "eE" else "+" if ch in "+-" else ch
        nxt = _NUM[mode[1]].get(cls)
        if nxt is not None and nxt is not True:
            return (stack, ("num", nxt, mode[2] + 1), ncalls) if mode[2] < MAX_NUM else None
        if None in _NUM[mode[1]]:  # the number ended; ch belongs to what follows it
            return _gstep((stack, ("after",), ncalls), ch, gram)
        return None
    if kind == "post":
        return (stack, ("post", mode[1] + 1), ncalls) if ch in WS and mode[1] < POST_WS else None
    if ch in WS:
        return state
    if kind in _SIMPLE:
        nxt = _SIMPLE[kind].get(ch)
        return None if nxt is None else (stack, nxt, ncalls)
    if kind == "pre":
        return (("CALLS",), ("call_open",), ncalls) if ch == "[" else None
    if kind == "call_open":
        return (stack + (("CALL", None),), ("name_key",), (ncalls[0] + 1, ncalls[1], "")) if ch == "{" else None
    if kind == "args_val":
        if ch != "{":
            return None
        checked = gram[3] is not None and stack[-1][1] in gram[3]
        return stack + (("ARGS", frozenset(), None) if checked else OBJ0,), ("obj_open",), ncalls
    if kind == "call_end":
        return (stack[:-1], ("calls_after",), ncalls) if ch == "}" else None
    if kind == "calls_after":
        if ch == "," and ncalls[0] < MAX_CALLS:
            return stack, ("call_open",), ncalls
        return ((), ("post", 0), ncalls) if ch == "]" else None
    spec = _spec(stack, gram)
    if kind == "obj_open":
        if ch == "}":
            return (*_close(stack), ncalls) if not spec else None
        return (stack, STR_KEY, ncalls) if ch == '"' and spec != {} else None
    if kind == "arr_open" and ch == "]":
        return (*_close(stack), ncalls)
    if kind == "value" and spec is not None and spec[stack[-1][2]] is not None:
        allowed = spec[stack[-1][2]]
        if ch != '"':
            return None
        return stack, STR_VAL if allowed is True else ("venum", "", allowed), ncalls
    if kind in ("arr_open", "value"):
        nxt = _value(stack, ch)
        return None if nxt is None else (*nxt, ncalls)
    if kind == "after":
        obj = stack[-1] != "ARR"
        full = spec is None or len(stack[-1][1]) == len(spec)
        if ch == ",":
            return (stack, ("obj_key",) if obj else ("value",), ncalls) if not full or spec is None else None
        if ch == ("}" if obj else "]"):
            return (*_close(stack), ncalls) if full else None
        return None
    raise ValueError(f"unknown grammar mode {mode}")


def gfeed(state: tuple, text: str, gram: tuple):
    for ch in text:
        state = gstep(state, ch, gram)
        if state is None:
            return None
    return state


class ToolCallGrammar(LogitsProcessor):
    """Per-row grammar mask for greedy decoding: keeps only the best legal token (see module docstring)."""

    def __init__(self, pieces: list, eos: int, prompts: list[str], max_str: int = MAX_STR, schema: bool = False,
                 values: dict | None = None):
        self.pieces, self.eos = pieces, eos
        self.grams = [grammar(p, max_str, schema, values) for p in prompts]
        self.states = [START] * len(prompts)
        self.texts = [[] for _ in prompts]
        self.done = [False] * len(prompts)
        self.steps = self.interventions = 0

    def _legal(self, i: int, tid: int):
        if tid == self.eos:
            return self.states[i] if self.states[i][1][0] == "post" else None
        piece = self.pieces[tid] if tid < len(self.pieces) else None
        return None if piece is None else gfeed(self.states[i], piece, self.grams[i])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        out = torch.full_like(scores, float("-inf"))
        top = scores.argmax(-1).tolist()
        for i in range(scores.shape[0]):
            if self.done[i]:
                out[i, self.eos] = 0
                continue
            self.steps += 1
            tid, state = top[i], self._legal(i, top[i])
            if state is None:
                self.interventions += 1
                for tid in torch.argsort(scores[i], descending=True).tolist():
                    if (state := self._legal(i, tid)) is not None:
                        break
            if state is None:  # nothing legal (should not happen: every mode has a 1-char continuation)
                tid, self.done[i] = self.eos, True
            elif tid == self.eos:
                self.done[i] = True
            else:
                self.states[i] = state
                self.texts[i].append(self.pieces[tid])
            out[i, tid] = 0
        return out


def token_pieces(tok, vocab: int) -> list:
    """Text of every token id; None for ids the grammar never allows (specials, partial UTF-8 bytes, empty)."""
    special = set(tok.all_special_ids)
    pieces = []
    for i in range(vocab):
        piece = None if i >= len(tok) or i in special else tok.decode([i], clean_up_tokenization_spaces=False)
        pieces.append(piece if piece and "�" not in piece else None)
    return pieces
