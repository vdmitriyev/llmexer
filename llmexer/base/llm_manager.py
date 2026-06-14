"""Experiment management: ``Experiment`` data class and ``ExperimentsManager`` mapper.

``ExperimentsManager`` owns a generated ``experiment_*.csv`` file as a pandas
DataFrame (rows = data x prompt x model x params combinations) and acts as a
*mapper* between that file on disk and in-memory state. It can run a single
combination by id directly through the existing LLM providers, copy the
provider's :class:`CallerState` / :class:`CallerStats` back into the row, and
report aggregate statistics over the whole file.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

import llmexer.base.llm_provider as llm_module
from llmexer.base.llm_provider import CallerState, ProviderAuth
from llmexer.exceptions import LLMExerException
from llmexer.logger import get_logger

logger = get_logger()

# Columns appended to a generated row once it has been run.
_RESULT_COLUMNS = [
    "response_text",
    "usage_tokens",
    "status",
    "state",
    "call_count",
    "total_tokens",
    "elapsed_seconds",
    "timestamp",
]


def _clean(value: Any) -> Any:
    """Normalise pandas/NaN values into plain Python (``None`` for missing)."""

    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


@dataclass
class Experiment:
    """A single generated-experiment combination (one row of the CSV).

    Carries the rendered prompt, the resolved model/provider/parameters, the
    LLM result, and the provider execution state. The full original row is kept
    in :attr:`raw` so that nothing from the source file is lost on round-trips.
    """

    experiment_id: str = ""  # maps to the unique ``code`` column
    row_id: Any = None  # the numeric ``ID`` column
    prompt: str = ""
    model_name: str = ""
    provider_name: str = ""

    profile_name: str = ""
    param_model_name: str = ""
    param_provider: str = ""
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
        """Build an :class:`Experiment` from a DataFrame row dict.

        Tolerates rows that have not been run yet (missing result columns).
        """

        row = {k: _clean(v) for k, v in dict(row).items()}
        experiment_id = str(row.get("code") or row.get("ID") or "")
        return cls(
            experiment_id=experiment_id,
            row_id=row.get("ID"),
            prompt=str(row.get("prompt") or ""),
            model_name=str(row.get("model_name") or ""),
            provider_name=str(row.get("provider_name") or ""),
            profile_name=str(row.get("profile_name") or ""),
            param_model_name=str(row.get("param_model_name") or ""),
            param_provider=str(row.get("param_provider") or ""),
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


class ExperimentsManager:
    """Mapper that manages a generated ``experiment_*.csv`` as a DataFrame.

    Typical lifecycle::

        mgr = ExperimentsManager()
        mgr.load("experiment_<eid>.csv")
        mgr.run(1)            # run a single combination by ID, write state back
        mgr.sync()            # flush changes back to the loaded file
        mgr.stats()           # aggregate statistics
    """

    def __init__(self, file: Optional[str] = None):
        self.file: Optional[str] = file
        self.df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ I/O
    def load(self, file: Optional[str] = None) -> pd.DataFrame:
        """Load the experiments CSV from disk into a DataFrame."""

        path = file or self.file
        if not path:
            raise LLMExerException("No file provided to load experiments from.")
        self.df = pd.read_csv(path, sep=";", encoding="utf-8")
        self.file = path
        logger.info(f"Loaded {len(self.df)} experiment row(s) from: '{path}'")
        return self.df

    def unload(self, file: Optional[str] = None) -> str:
        """Write the whole DataFrame to disk (defaults to the loaded file)."""

        self._require_loaded()
        path = file or self.file
        if not path:
            raise LLMExerException("No file provided to unload experiments to.")
        self.df.to_csv(path, index=False, sep=";", encoding="utf-8")
        logger.info(f"Unloaded {len(self.df)} experiment row(s) to: '{path}'")
        return path

    def sync(self, file: Optional[str] = None) -> str:
        """Flush current in-memory state back to the loaded source file.

        ``sync`` always persists to the originally loaded file (unless ``file``
        is given explicitly), whereas :meth:`unload` is a plain dump to an
        arbitrary path.
        """

        self._require_loaded()
        path = file or self.file
        if not path:
            raise LLMExerException(
                "Nothing has been loaded yet — call load() before sync()."
            )
        return self.unload(path)

    # -------------------------------------------------------------- running
    def run(self, id_experiment: Any) -> Experiment:
        """Run a single experiment combination by id, writing state back.

        Locates the row whose ``ID`` (or ``code``) matches ``id_experiment``,
        resolves the right provider, executes the LLM call, and copies the
        provider's :class:`CallerState` and :class:`CallerStats` into the row.
        """

        self._require_loaded()
        idx = self._locate(id_experiment)
        row = {k: _clean(v) for k, v in self.df.loc[idx].to_dict().items()}

        experiment = Experiment.from_row(row)
        provider = (experiment.param_provider or "").lower()
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
            from datetime import datetime, timezone

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

        self._write_back(idx, experiment)
        return experiment

    # -------------------------------------------------------------- stats
    def stats(self) -> Dict[str, Any]:
        """Return aggregate statistics over the loaded experiments."""

        self._require_loaded()
        df = self.df
        total = len(df)

        if "status" in df.columns:
            status_str = df["status"].astype("string")
            completed = int((status_str == "success").sum())
            errors = int(status_str.str.startswith("Error", na=False).sum())
            pending = int(status_str.isna().sum())
        else:
            completed = 0
            errors = 0
            pending = total

        running = 0
        if "state" in df.columns:
            running = int(
                (df["state"].astype("string") == CallerState.RUNNING.value).sum()
            )

        total_tokens = 0
        if "total_tokens" in df.columns:
            total_tokens = int(
                pd.to_numeric(df["total_tokens"], errors="coerce").fillna(0).sum()
            )
        elif "usage_tokens" in df.columns:
            total_tokens = int(
                pd.to_numeric(df["usage_tokens"], errors="coerce").fillna(0).sum()
            )

        providers = self._value_counts("param_provider")
        models = self._value_counts("param_model_name")

        return {
            "total": total,
            "completed": completed,
            "running": running,
            "errors": errors,
            "pending": pending,
            "total_tokens": total_tokens,
            "providers": providers,
            "models": models,
        }

    # ------------------------------------------------------------- helpers
    def _require_loaded(self) -> None:
        if self.df is None:
            raise LLMExerException(
                "No experiments loaded. Call load() before this operation."
            )

    def _locate(self, id_experiment: Any) -> Any:
        """Return the DataFrame index of the row matching ``id_experiment``."""

        for column in ("ID", "code"):
            if column in self.df.columns:
                matches = self.df.index[
                    self.df[column].astype("string") == str(id_experiment)
                ]
                if len(matches):
                    return matches[0]
        raise LLMExerException(f"No experiment row found with id '{id_experiment}'.")

    def _write_back(self, idx: Any, experiment: Experiment) -> None:
        for column in _RESULT_COLUMNS:
            if column not in self.df.columns:
                self.df[column] = pd.Series([None] * len(self.df), dtype=object)
                self.df[column] = self.df[column].astype(object)
            self.df.at[idx, column] = getattr(experiment, column)

    def _value_counts(self, column: str) -> Dict[str, int]:
        if column not in self.df.columns:
            return {}
        counts = self.df[column].astype("string").value_counts(dropna=True)
        return {str(k): int(v) for k, v in counts.items()}
