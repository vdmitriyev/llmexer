"""Experiment management: ``Experiment`` data class and ``ExperimentsManager``.

An experiment is generated into a per-provider SQLite database (see
:mod:`llmexer.base.dao`). ``run_experiment_row`` executes a single generated
row (a plain dict from the DAO) against the right LLM provider and returns an
:class:`Experiment` carrying the result and provider state. ``ExperimentsManager``
is a thin DAO-backed convenience wrapper that runs a row by id and reports
aggregate statistics.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import llmexer.base.llm_provider as llm_module
from llmexer.base.dao import ExperimentDAO
from llmexer.base.llm_provider import CallerState, ProviderAuth
from llmexer.exceptions import LLMExerException
from llmexer.logger import get_logger

logger = get_logger()


@dataclass
class Experiment:
    """A single generated-experiment combination (one row of a provider table).

    Carries the rendered prompt, the resolved model/provider/parameters, the
    LLM result, and the provider execution state. The full original row is kept
    in :attr:`raw` so that nothing from the source is lost on round-trips.
    """

    experiment_id: str = ""  # maps to the unique ``code`` column
    row_id: Any = None  # the numeric ``ID`` column
    prompt: str = ""
    model_name: str = ""
    provider_name: str = ""

    profile_name: str = ""
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None

    # Execution results / provider state
    response_text: str = ""
    usage_tokens: Optional[int] = None
    status: Optional[str] = None
    state: Optional[str] = None
    call_count: int = 0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0
    timestamp: Optional[str] = None

    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Experiment":
        """Build an :class:`Experiment` from a row dict.

        Tolerates rows that have not been run yet (NULL result columns).
        """

        row = dict(row)
        experiment_id = str(row.get("code") or row.get("ID") or "")
        return cls(
            experiment_id=experiment_id,
            row_id=row.get("ID"),
            prompt=str(row.get("prompt") or ""),
            model_name=str(row.get("model_name") or ""),
            provider_name=str(row.get("provider_name") or ""),
            profile_name=str(row.get("profile_name") or ""),
            temperature=row.get("temperature"),
            top_p=row.get("top_p"),
            max_tokens=row.get("max_tokens"),
            response_text=str(row.get("response_text") or ""),
            usage_tokens=row.get("usage_tokens"),
            status=row.get("status"),
            state=row.get("state"),
            call_count=int(row.get("call_count") or 0),
            total_tokens=int(row.get("total_tokens") or 0),
            elapsed_seconds=float(row.get("elapsed_seconds") or 0.0),
            timestamp=row.get("timestamp"),
            raw=row,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Flat dict combining the original row with current result/state fields."""

        merged = dict(self.raw)
        merged.update(
            {
                "response_text": self.response_text,
                "usage_tokens": self.usage_tokens,
                "status": self.status,
                "state": self.state,
                "call_count": self.call_count,
                "total_tokens": self.total_tokens,
                "elapsed_seconds": self.elapsed_seconds,
                "timestamp": self.timestamp,
            }
        )
        return merged

    def _default_path(self, ext: str) -> str:
        name = self.experiment_id or "experiment"
        # Make the id filesystem-safe (model/profile codes may contain "/" or ":").
        safe = name.replace("/", "-").replace(":", "-")
        return f"{safe}.{ext}"

    def to_json(self, file: Optional[str] = None) -> str:
        """Write this experiment to a formatted JSON file (``indent=4``).

        If ``file`` is omitted, a default name based on ``experiment_id`` is used.
        Returns the path written.
        """

        path = file or self._default_path("json")
        payload = {k: v for k, v in asdict(self).items() if k != "raw"}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=4, ensure_ascii=False)
        logger.debug(f"Experiment '{self.experiment_id}' written to JSON: '{path}'")
        return path

    def to_yaml(self, file: Optional[str] = None) -> str:
        """Write this experiment to a formatted YAML file.

        If ``file`` is omitted, a default name based on ``experiment_id`` is used.
        Returns the path written.
        """

        import yaml

        path = file or self._default_path("yaml")
        payload = {k: v for k, v in asdict(self).items() if k != "raw"}
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(payload, fh, default_flow_style=False, sort_keys=False)
        logger.debug(f"Experiment '{self.experiment_id}' written to YAML: '{path}'")
        return path


def build_response_payload(experiment: Experiment, provider: str) -> Dict[str, Any]:
    """Build the per-call JSON payload exported to ``responses/`` and the DB."""

    return {
        "model": experiment.model_name,
        "provider": provider,
        "prompt": experiment.prompt,
        "profile": experiment.profile_name,
        "response_text": experiment.response_text,
        "usage_tokens": experiment.usage_tokens,
        "status": experiment.status,
        "timestamp": experiment.timestamp,
    }


def result_values(experiment: Experiment, provider: str) -> Dict[str, Any]:
    """Build the result-column dict (incl. ``response_json``) for a DB update."""

    payload = build_response_payload(experiment, provider)
    return {
        "response_text": experiment.response_text,
        "usage_tokens": experiment.usage_tokens,
        "status": experiment.status,
        "state": experiment.state,
        "call_count": experiment.call_count,
        "total_tokens": experiment.total_tokens,
        "elapsed_seconds": experiment.elapsed_seconds,
        "timestamp": experiment.timestamp,
        "response_json": json.dumps(payload, ensure_ascii=False),
    }


def run_experiment_row(row: Dict[str, Any]) -> Experiment:
    """Execute a single generated row against its provider and return results.

    Resolves the provider from ``provider_name``, runs the LLM call, and copies
    the provider's :class:`CallerState`/:class:`CallerStats` into the returned
    :class:`Experiment`. Pure: it does not persist anything.
    """

    experiment = Experiment.from_row(row)
    provider = (experiment.provider_name or "").lower()
    base_url, api_key = llm_module.resolve_provider_config(provider)

    if provider == "ollama":
        caller = llm_module.OllamaProvider(
            provider=provider,
            auth=ProviderAuth(api_key=api_key),
            base_url=base_url or llm_module.URL_MAP["ollama"],
        )
        resp = caller.execute(experiment.prompt, row)
        experiment.response_text = resp.text
        experiment.usage_tokens = resp.usage_tokens
        experiment.status = (
            f"Error: {resp.raw}" if caller.state == CallerState.ERROR else "success"
        )
        state = getattr(caller, "state", CallerState.SUCCESS)
        experiment.state = getattr(state, "value", str(state))
        stats = getattr(caller, "stats", None)
        experiment.call_count = getattr(stats, "call_count", 1)
        experiment.total_tokens = getattr(
            stats, "total_tokens", experiment.usage_tokens or 0
        )
        experiment.elapsed_seconds = getattr(stats, "elapsed_seconds", 0.0)
        experiment.timestamp = datetime.now(timezone.utc).isoformat()
    else:
        mapper = llm_module.LLMRequestsMapper(
            provider=provider, base_url=base_url, api_key=api_key
        )
        result = mapper.execute(experiment.prompt, row)
        experiment.response_text = result.response_text
        experiment.usage_tokens = result.usage_tokens
        experiment.status = result.status
        # LLMRequestsMapper has no CallerState/CallerStats; derive them.
        experiment.state = (
            CallerState.SUCCESS.value
            if result.status == "success"
            else CallerState.ERROR.value
        )
        experiment.call_count = 1
        experiment.total_tokens = result.usage_tokens or 0
        experiment.timestamp = result.timestamp

    return experiment


class ExperimentsManager:
    """DAO-backed convenience wrapper over a single experiment database.

    Typical lifecycle::

        mgr = ExperimentsManager(db_path)
        mgr.run(1)        # run a single combination by ID (or code), persist result
        mgr.stats()       # aggregate statistics across all provider tables
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path: Optional[str] = db_path
        self.dao: Optional[ExperimentDAO] = ExperimentDAO(db_path) if db_path else None

    def open(self, db_path: Optional[str] = None) -> ExperimentDAO:
        """Open (reflect) an existing experiment database."""

        path = db_path or self.db_path
        if not path:
            raise LLMExerException("No database provided to open experiments from.")
        self.dao = ExperimentDAO(path)
        self.db_path = path
        return self.dao

    def _require_open(self) -> None:
        if self.dao is None:
            raise LLMExerException(
                "No experiment database opened. Pass db_path or call open() first."
            )

    def run(self, id_experiment: Any) -> Experiment:
        """Run a single experiment combination by id, persisting the result."""

        self._require_open()
        rows = self.dao.fetch_rows(id_experiment=id_experiment)
        if not rows:
            raise LLMExerException(
                f"No experiment row found with id '{id_experiment}'."
            )

        row = rows[0]
        experiment = run_experiment_row(row)
        provider = row.get("_provider") or (experiment.provider_name or "").lower()
        self.dao.update_result(provider, row["ID"], result_values(experiment, provider))
        return experiment

    def stats(self) -> Dict[str, Any]:
        """Return aggregate statistics over all provider tables."""

        self._require_open()
        return self.dao.stats()

    def close(self) -> None:
        if self.dao is not None:
            self.dao.close()
