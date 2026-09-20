"""Repair of answers that hold JSON plus something else.

A prompt asking for JSON is often answered with the JSON wrapped in a Markdown
fence, introduced by a sentence, or followed by an explanation. ``json.loads``
then rejects the whole answer, so the row drops out of every downstream
consumer. These helpers find the JSON inside such an answer; they never invent
one.
"""

import json

from llmexer.common import strip_code_fence

_DECODER = json.JSONDecoder()

# A JSON answer is an object or an array; a bare string or number in an
# otherwise prose reply is a coincidence, not the answer.
_OPENING_CHARS = "{["


def is_json(text: str) -> bool:
    """Tell whether ``text`` parses as JSON exactly as it stands."""

    try:
        json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False
    return True


def needs_fix(value) -> bool:
    """Tell whether ``value`` is a non-empty answer that does not parse as JSON.

    An unrun row (``None``) and an empty answer are left alone: there is nothing
    to repair, and an empty cell is not a broken one.
    """

    if not isinstance(value, str) or not value.strip():
        return False

    return not is_json(value)


def extract_json(text: str) -> str | None:
    """Return the JSON embedded in ``text``, or ``None`` when there is none.

    The Markdown fence is peeled off first, then the FIRST complete JSON value
    in what is left is taken and whatever trails it is dropped. The matched span
    is returned verbatim - the model's own formatting is kept, nothing is
    re-serialised.
    """

    if not isinstance(text, str):
        return None

    candidate = strip_code_fence(text)

    for index, char in enumerate(candidate):
        if char not in _OPENING_CHARS:
            continue
        try:
            _, end = _DECODER.raw_decode(candidate, index)
        except (json.JSONDecodeError, ValueError):
            continue
        return candidate[index:end]

    return None


def fixed_value(value) -> str | None:
    """Return the repaired answer for ``value``, or ``None`` if there is nothing to do.

    ``None`` covers both an answer that is already valid JSON and one that holds
    no JSON at all - the caller tells the two apart with ``needs_fix``.
    """

    if not needs_fix(value):
        return None

    return extract_json(value)
