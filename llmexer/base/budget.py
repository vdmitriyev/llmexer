"""Spend tracking for a single run.

The cap exists because an ``experiment run`` over a large cross join can spend
real money through a paid provider. A :class:`SessionBudget` is built once per
run and thrown away with it, so the amount spent resets every time the command
exits -- there is no stored tally and no state shared between runs.
"""

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from llmexer.configs import logger
from llmexer.constants import DEFAULT_OPENROUTER_MAX_SPEND_USD

# Environment variable overriding the default cap.
MAX_SPEND_ENV_VAR = "PROVIDER_OPENROUTER_MAX_SPEND"


def resolve_max_spend() -> float:
    """Resolve the spend ceiling from ``PROVIDER_OPENROUTER_MAX_SPEND`` (else the default).

    Read at call time (not import time) so ``.env`` values loaded by the CLI are honored.
    Falls back to the default on an unset, non-positive, or non-numeric value: the cap is
    a safety net, so a typo must not silently turn it off, and must never raise from a
    worker thread mid-run.
    """

    raw = os.getenv(MAX_SPEND_ENV_VAR)
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
            logger.warning(
                "Non-positive %s=%r; using default %s USD",
                MAX_SPEND_ENV_VAR,
                raw,
                DEFAULT_OPENROUTER_MAX_SPEND_USD,
            )
        except ValueError:
            logger.warning(
                "Invalid %s=%r; using default %s USD",
                MAX_SPEND_ENV_VAR,
                raw,
                DEFAULT_OPENROUTER_MAX_SPEND_USD,
            )
    return DEFAULT_OPENROUTER_MAX_SPEND_USD


def as_cost(value: Any) -> Optional[float]:
    """Return ``value`` as a USD amount, or ``None`` when it is not one.

    Deliberately strict: no ``float()`` coercion. A provider reporting nothing
    leaves a stand-in object behind (a ``MagicMock`` under test, a sentinel in an
    SDK), and ``float()`` would turn that into a plausible-looking amount booked
    against the cap. Booleans are excluded because ``bool`` is a subclass of
    ``int`` and ``True`` would otherwise read as one dollar.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value >= 0 else None


@dataclass
class SessionBudget:
    """What one run is allowed to spend, and what it has spent so far.

    Shared across the worker threads of a parallel run, so every read and write
    of the tally goes through a lock. The lock keeps the counter itself correct;
    it cannot stop the cap being overshot, because a call's cost is only known
    once that call has returned -- see :meth:`is_exhausted`.
    """

    limit_usd: float
    spent_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, cost_usd: float) -> float:
        """Book one call's cost against the budget; return the new total."""

        with self._lock:
            self.spent_usd += cost_usd
            return self.spent_usd

    def is_exhausted(self) -> bool:
        """Whether the cap has been reached and no further call may be made.

        Checked before a call, but a cost is only reported after one, so up to
        ``--parallel-calls`` calls may already be in flight when this first turns
        true. The final amount spent can exceed ``limit_usd`` by that much.
        """

        with self._lock:
            return self.spent_usd >= self.limit_usd
