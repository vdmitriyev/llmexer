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

# Why a rename was skipped, for the caller to report.
SKIP_NOT_AN_OBJECT = "answer is not a JSON object"
SKIP_NO_SUCH_FIELD = "no such field"
SKIP_NAME_TAKEN = "the new name is already used"


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


def rename_json_field(value, old: str, new: str) -> tuple:
    """Return the answer with the top-level key ``old`` renamed to ``new``.

    Shaped like ``transform.parse_json_payload``: ``(text, "")`` when the rename
    happened and ``(None, reason)`` when there was nothing to do.

    Only the outermost object is touched. A key of the same name nested deeper
    answers a different question and is left alone.

    The result is re-serialised, so the model's own formatting does not survive
    a rename - unlike :func:`extract_json`, which returns the matched span
    verbatim. The text being replaced is what the caller stores as the old value.
    """

    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError, ValueError):
        return None, SKIP_NOT_AN_OBJECT

    if not isinstance(parsed, dict):
        return None, SKIP_NOT_AN_OBJECT

    if old not in parsed:
        return None, SKIP_NO_SUCH_FIELD

    if new in parsed:
        # Renaming onto a name in use would drop one of the two answers.
        return None, SKIP_NAME_TAKEN

    # Rebuilt in order, so the field keeps its place among its siblings.
    renamed = {(new if key == old else key): item for key, item in parsed.items()}

    return json.dumps(renamed, ensure_ascii=False, indent=2), ""
