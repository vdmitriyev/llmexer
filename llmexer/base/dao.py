"""Data access layer for experiment storage backed by SQLite + SQLAlchemy.

Each LLM provider that appears in a generation run gets *two* tables:

``experiment_<provider>`` (e.g. ``experiment_ollama``) holds the common
identity/prompt columns, the ``(params_code, profile_name)`` join key, the
result columns (including ``response_json``) and finally the SHA-256 hash
columns — so the generated rows and their future results live together.

``params_<provider>`` (e.g. ``params_ollama``) holds one row per parameter set,
keyed by the composite primary key ``(params_code, profile_name)``, with the
common parameters and that provider's own parameter columns. Storing them here
instead of on every experiment row means a hyperparameter profile is written
once per run rather than repeated across the whole cross join.

:meth:`ExperimentDAO.fetch_rows` joins the two back together, so the rest of the
codebase keeps receiving one flat ``dict`` per experiment row. Databases written
before this split have no ``params_<provider>`` table and are rejected on open.

All SQLAlchemy access is funnelled through :class:`ExperimentDAO`; the rest of
the codebase passes and receives plain ``dict`` rows and never touches the
engine directly. SQLAlchemy *Core* is used (``MetaData`` + ``Table`` built at
runtime) because the set of tables and their columns is data-driven by which
providers are present.
"""

import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    and_,
    case,
    create_engine,
    func,
    insert,
    or_,
    select,
    update,
)

from llmexer.base.experiment import (
    COMMON_IDENTITY_COLUMNS,
    COMMON_PARAM_COLUMNS,
    HASH_COLUMNS,
    PARAMS_KEY_COLUMNS,
    PROVIDER_PARAM_COLUMNS,
    RESULT_COLUMNS,
)
from llmexer.exceptions import LLMExerException
from llmexer.logger import get_logger

logger = get_logger()

TABLE_PREFIX = "experiment_"
PARAMS_TABLE_PREFIX = "params_"
DB_PREFIX = "experiment"
DB_SUFFIX = ".db"

# SQLite is dynamically typed, so these affinities are advisory — but declaring
# them keeps the schema self-documenting and portable to other backends.
COLUMN_TYPES: Dict[str, Any] = {
    # identity / prompt
    "ID": Integer,
    "code": String,
    "prompt": Text,
    "tokens_estimate": Integer,
    "original_data": Text,
    "model_name": String,
    "provider_name": String,
    "prompt_hash": String(64),
    "original_data_hash": String(64),
    # params join key (present in BOTH the experiment and the params table)
    "params_code": String,
    "profile_name": String,
    # common params
    "temperature": Float,
    "top_p": Float,
    "max_tokens": Integer,
    # provider-specific params
    "ollama_context_window": Integer,
    "ollama_repeat_penalty": Float,
    "vllm_min_p": Float,
    "vllm_best_of": Integer,
    "openai_seed": Integer,
    "gemini_thinking_level": String,
    "litellm_min_p": Float,
    "litellm_best_of": Integer,
    # results
    "response_text": Text,
    "usage_tokens": Integer,
    "status": String,
    "state": String,
    "call_count": Integer,
    "total_tokens": Integer,
    "elapsed_seconds": Float,
    "timestamp": String,
    "response_json": Text,
}


def table_name_for(provider: str) -> str:
    """Return the table name for a provider (``experiment_<provider>``)."""

    return f"{TABLE_PREFIX}{str(provider).lower()}"


def provider_from_table_name(name: str) -> str:
    """Inverse of :func:`table_name_for`."""

    return name[len(TABLE_PREFIX) :] if name.startswith(TABLE_PREFIX) else name


def params_table_name_for(provider: str) -> str:
    """Return the params table name for a provider (``params_<provider>``)."""

    return f"{PARAMS_TABLE_PREFIX}{str(provider).lower()}"


def provider_from_params_table_name(name: str) -> str:
    """Inverse of :func:`params_table_name_for`."""

    return name[len(PARAMS_TABLE_PREFIX) :] if name.startswith(PARAMS_TABLE_PREFIX) else name


def _experiment_columns(provider: str) -> List[str]:
    """Ordered column list for a provider's ``experiment_<provider>`` table."""

    return list(COMMON_IDENTITY_COLUMNS) + list(PARAMS_KEY_COLUMNS) + list(RESULT_COLUMNS) + list(HASH_COLUMNS)


def _params_columns(provider: str) -> List[str]:
    """Ordered column list for a provider's ``params_<provider>`` table.

    A provider with no entry in ``PROVIDER_PARAM_COLUMNS`` still gets a table,
    carrying the join key and the common parameters only.
    """

    extra = PROVIDER_PARAM_COLUMNS.get(str(provider).lower(), [])
    return list(PARAMS_KEY_COLUMNS) + list(COMMON_PARAM_COLUMNS) + list(extra)


def _clean_value(value: Any) -> Any:
    """Normalise pandas/NumPy scalars and NaN into plain Python for SQLite."""

    if value is None:
        return None
    # NumPy / pandas scalars expose ``.item()``; unwrap to a Python scalar.
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def params_code_for(model_name: Any, provider: Any) -> str:
    """Build the parameter-set join key: ``<model_name>_<provider>``.

    The provider is lower-cased so the key always matches the table it is stored
    in, however the CSV spelled it; the model name keeps its case, matching the
    case-sensitive join between ``llms-for-experiment.csv`` and
    ``llm-params.csv``. A missing/NaN model name collapses to an empty string
    rather than the literal ``"None"``.
    """

    model = _clean_value(model_name)
    model = "" if model is None else str(model).strip()
    return f"{model}_{str(provider).strip().lower()}"


def list_db_files(folder: str) -> List[str]:
    """Return sorted ``experiment*.db`` filenames in ``folder`` (may be empty)."""

    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.startswith(DB_PREFIX) and f.endswith(DB_SUFFIX))


def _counter_of(filename: str) -> int:
    """Extract the trailing ``_NN`` counter from an experiment db filename."""

    stem = filename[: -len(DB_SUFFIX)] if filename.endswith(DB_SUFFIX) else filename
    token = stem.rsplit("_", 1)[-1]
    try:
        return int(token)
    except ValueError:
        return 0


def next_db_filename(
    folder: str,
    prefix: str = DB_PREFIX,
    suffix: str = DB_SUFFIX,
    date: Optional[str] = None,
) -> str:
    """Compute the next ``experiment_<date>_<NN>.db`` name for ``folder``.

    The counter is a zero-padded sequential number starting at ``01``,
    one greater than the highest counter among existing ``experiment*.db``
    files. ``date`` defaults to today's UTC ``YYYYMMDD``.
    """

    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
    existing = [f for f in list_db_files(folder) if f.startswith(prefix) and f.endswith(suffix)]
    counter = max((_counter_of(f) for f in existing), default=0) + 1
    return f"{prefix}_{date}_{counter:02d}{suffix}"


def latest_db(folder: str) -> Optional[str]:
    """Return the path of the highest-counter ``experiment*.db`` or ``None``."""

    files = list_db_files(folder)
    if not files:
        return None
    newest = max(files, key=_counter_of)
    return os.path.join(folder, newest)


class ExperimentDAO:
    """SQLAlchemy-Core data access object for a single experiment database.

    Typical lifecycles::

        # generate
        with ExperimentDAO(db_path, create=True) as dao:
            dao.insert_rows("ollama", rows)

        # run / stats
        with ExperimentDAO(db_path) as dao:
            for row in dao.fetch_rows(provider="ollama"):
                ...
            dao.update_result("ollama", row["ID"], result)
    """

    def __init__(self, db_path: str, create: bool = False):
        self.db_path = db_path
        self.metadata = MetaData()
        self._tables: Dict[str, Table] = {}
        # Kept apart from ``_tables``: provider_tables(), fetch_rows() and
        # stats() all iterate the experiment tables and must never see these.
        self._params_tables: Dict[str, Table] = {}

        if not create and not os.path.exists(db_path):
            raise LLMExerException(f"Experiment database not found: '{db_path}'.")

        self.engine = create_engine(f"sqlite:///{db_path}")

        if not create:
            self.metadata.reflect(bind=self.engine)
            for table in self.metadata.tables.values():
                if table.name.startswith(PARAMS_TABLE_PREFIX):
                    self._params_tables[provider_from_params_table_name(table.name)] = table
                elif table.name.startswith(TABLE_PREFIX):
                    self._tables[provider_from_table_name(table.name)] = table
            self._require_params_tables()

    def _require_params_tables(self) -> None:
        """Reject databases written before parameters moved to their own table.

        There is no migration path: an ``experiment_<provider>`` table without
        its ``params_<provider>`` counterpart carries its parameters inline, in
        columns this version no longer knows about.
        """

        missing = sorted(set(self._tables) - set(self._params_tables))
        if not missing:
            return
        names = ", ".join(f"'{params_table_name_for(p)}'" for p in missing)
        raise LLMExerException(
            f"Experiment database '{self.db_path}' uses the previous single-table layout: "
            f"no {names} table(s) to join parameters from. Parameters now live in a separate "
            "'params_<provider>' table -- re-run `experiment generate` to create a database "
            "in the current format."
        )

    # ----------------------------------------------------------------- schema
    def _build_table(self, provider: str) -> Table:
        columns = [
            Column(name, COLUMN_TYPES[name], primary_key=(name == "ID")) for name in _experiment_columns(provider)
        ]
        return Table(table_name_for(provider), self.metadata, *columns)

    def _build_params_table(self, provider: str) -> Table:
        # Composite primary key: one (model, provider) pair can have several
        # profiles in llm-params.csv, so params_code alone is not unique.
        columns = [
            Column(name, COLUMN_TYPES[name], primary_key=(name in PARAMS_KEY_COLUMNS))
            for name in _params_columns(provider)
        ]
        return Table(params_table_name_for(provider), self.metadata, *columns)

    def ensure_provider_table(self, provider: str) -> Table:
        """Return the (built, not yet created) experiment Table for ``provider``.

        The matching ``params_<provider>`` table is registered at the same time:
        the two are always created together, so an experiment table can never be
        left without the params table :meth:`fetch_rows` joins against.
        """

        key = str(provider).lower()
        if key not in self._tables:
            self._tables[key] = self._build_table(key)
        self.ensure_params_table(key)
        return self._tables[key]

    def ensure_params_table(self, provider: str) -> Table:
        """Return the (built, not yet created) params Table for ``provider``."""

        key = str(provider).lower()
        if key not in self._params_tables:
            self._params_tables[key] = self._build_params_table(key)
        return self._params_tables[key]

    def create_tables(self) -> None:
        """Create all tables registered so far that do not yet exist."""

        self.metadata.create_all(bind=self.engine)

    def provider_tables(self) -> Dict[str, Table]:
        """Mapping of provider name -> ``experiment_<provider>`` Table."""

        return dict(self._tables)

    def params_tables(self) -> Dict[str, Table]:
        """Mapping of provider name -> ``params_<provider>`` Table."""

        return dict(self._params_tables)

    def _table_for(self, provider: str) -> Table:
        key = str(provider).lower()
        if key not in self._tables:
            raise LLMExerException(f"No experiment table for provider '{provider}' in '{self.db_path}'.")
        return self._tables[key]

    def _params_table_for(self, provider: str) -> Table:
        key = str(provider).lower()
        if key not in self._params_tables:
            raise LLMExerException(
                f"No params table '{params_table_name_for(key)}' for provider '{provider}' "
                f"in '{self.db_path}'. Re-run `experiment generate`."
            )
        return self._params_tables[key]

    # ---------------------------------------------------------------- generate
    def _split_rows(self, key, rows, table, params_table):
        """Split flat rows into experiment payloads and unique parameter sets.

        Returns ``(experiment_rows, params_rows_by_key)`` where the second is
        keyed by ``(params_code, profile_name)`` — the same parameter set recurs
        on every row of the cross join, so it collapses to one entry here.
        """

        exp_valid = set(table.c.keys())
        params_valid = set(params_table.c.keys())

        exp_payload: List[dict] = []
        params_payload: Dict[tuple, dict] = {}
        for row in rows:
            cleaned = {k: _clean_value(v) for k, v in row.items()}
            # Derive from the ``provider`` argument rather than the row's own
            # provider_name, so the key always matches the table it lands in.
            code = cleaned.get("params_code") or params_code_for(cleaned.get("model_name"), key)
            # profile_name is half of the params PRIMARY KEY and so cannot be
            # NULL; normalise it on BOTH payloads or the join would silently
            # miss and every parameter would come back NULL.
            profile = cleaned.get("profile_name")
            profile = "" if profile is None else str(profile)
            cleaned["params_code"] = code
            cleaned["profile_name"] = profile

            exp_payload.append({k: v for k, v in cleaned.items() if k in exp_valid})

            params_row = {k: v for k, v in cleaned.items() if k in params_valid}
            if params_payload.setdefault((code, profile), params_row) != params_row:
                logger.warning(
                    f"Conflicting parameter values for ('{code}', '{profile}') in "
                    f"'{params_table.name}'; keeping the first set."
                )
        return exp_payload, params_payload

    def insert_rows(self, provider: str, rows: List[dict]) -> int:
        """Bulk-insert generated rows, splitting each flat row across two tables.

        Callers pass one flat dict per experiment combination. Identity, result
        and hash values go to ``experiment_<provider>``; the parameter values go
        to ``params_<provider>``, deduplicated on ``(params_code, profile_name)``
        both within this call and against parameter sets an earlier call already
        stored. Each dict is filtered to its table's columns (extra keys ignored,
        missing columns left NULL) and cleaned of NaN/NumPy scalars. Both tables
        are created on first insert if they do not yet exist.

        Returns the number of experiment rows inserted.
        """

        if not rows:
            return 0
        key = str(provider).lower()
        table = self.ensure_provider_table(key)
        params_table = self.ensure_params_table(key)
        self.create_tables()

        exp_payload, params_payload = self._split_rows(key, rows, table, params_table)

        with self.engine.begin() as conn:
            stored = {
                (r.params_code, r.profile_name)
                for r in conn.execute(select(params_table.c.params_code, params_table.c.profile_name))
            }
            new_params = [values for pk, values in params_payload.items() if pk not in stored]
            if new_params:
                conn.execute(insert(params_table), new_params)
            conn.execute(insert(table), exp_payload)

        logger.info(
            f"Inserted {len(exp_payload)} row(s) into '{table.name}' and {len(new_params)} "
            f"parameter set(s) into '{params_table.name}' of '{self.db_path}'."
        )
        return len(exp_payload)

    # --------------------------------------------------------------------- run
    def fetch_rows(
        self,
        provider: Optional[str] = None,
        id_experiment: Optional[Any] = None,
    ) -> List[dict]:
        """Return rows across all (or one) provider tables, ordered by ID.

        Each experiment row is LEFT-joined to its parameter set, so the returned
        dict is flat and shaped exactly as it was before parameters moved to
        their own table — providers and :class:`~llmexer.base.llm_manager.Experiment`
        read parameters straight off it by name.

        Every row dict carries an extra ``_provider`` key identifying its table.
        ``id_experiment`` matches the numeric ``ID`` or the ``code`` column.
        """

        if provider is not None:
            key = str(provider).lower()
            tables = {key: self._tables[key]} if key in self._tables else {}
        else:
            tables = self._tables

        results: List[dict] = []
        with self.engine.connect() as conn:
            for prov, table in tables.items():
                params_table = self._params_table_for(prov)
                # Select the experiment table plus only the params *value*
                # columns: params_code/profile_name exist on both sides, and
                # selecting both would yield 'params_code_1' keys in .mappings().
                param_value_cols = [
                    params_table.c[name] for name in params_table.c.keys() if name not in PARAMS_KEY_COLUMNS
                ]
                # Outer join: a row whose parameter set went missing still comes
                # back (with NULL parameters) instead of vanishing from `run`.
                joined = table.join(
                    params_table,
                    and_(
                        table.c.params_code == params_table.c.params_code,
                        table.c.profile_name == params_table.c.profile_name,
                    ),
                    isouter=True,
                )
                stmt = select(table, *param_value_cols).select_from(joined)
                if id_experiment is not None:
                    conditions = [table.c.code == str(id_experiment)]
                    try:
                        conditions.append(table.c.ID == int(id_experiment))
                    except (TypeError, ValueError):
                        pass
                    stmt = stmt.where(or_(*conditions))
                stmt = stmt.order_by(table.c.ID)
                for mapping in conn.execute(stmt).mappings():
                    row = dict(mapping)
                    row["_provider"] = prov
                    results.append(row)
        results.sort(key=lambda r: (r.get("ID") if r.get("ID") is not None else 0))
        return results

    def update_result(self, provider: str, row_id: Any, result: dict) -> None:
        """Write result columns back onto a single row, keyed by its ``ID``.

        Only ``experiment_<provider>`` columns are writable here; a generated
        parameter set is immutable, and any parameter key is filtered out.
        """

        table = self._table_for(provider)
        valid = set(table.c.keys())
        values = {k: _clean_value(v) for k, v in result.items() if k in valid}
        if not values:
            return
        with self.engine.begin() as conn:
            conn.execute(update(table).where(table.c.ID == row_id).values(**values))

    # ------------------------------------------------------------------- stats
    def stats(self) -> Dict[str, Any]:
        """Aggregate statistics across every provider table in the database.

        Reads only identity and result columns, so no params join is needed.
        """

        total = finished = running = errors = total_tokens = 0
        providers: Dict[str, int] = {}
        models: Dict[str, Dict[str, Any]] = {}

        with self.engine.connect() as conn:
            for prov, table in self._tables.items():

                def count(condition=None) -> int:
                    stmt = select(func.count()).select_from(table)
                    if condition is not None:
                        stmt = stmt.where(condition)
                    return int(conn.execute(stmt).scalar() or 0)

                rows = count()
                total += rows
                providers[prov] = providers.get(prov, 0) + rows
                finished += count(table.c.status == "success")
                errors += count(table.c.status.like("Error%"))
                running += count(table.c.state == "running")

                token_sum = conn.execute(
                    select(func.sum(func.coalesce(table.c.total_tokens, table.c.usage_tokens, 0))).select_from(table)
                ).scalar()
                total_tokens += int(token_sum or 0)

                # Per-model aggregates: counts of finished (status "success") and
                # open (pending/unrun, NULL status) rows, plus tokens and elapsed
                # time accumulated over the model's *finished* rows only.
                is_finished = table.c.status == "success"
                finished_tokens = func.coalesce(table.c.total_tokens, table.c.usage_tokens, 0)
                for (
                    name,
                    cnt,
                    fin,
                    opn,
                    toks,
                    secs,
                ) in conn.execute(
                    select(
                        table.c.model_name,
                        func.count(),
                        func.sum(case((is_finished, 1), else_=0)),
                        func.sum(case((table.c.status.is_(None), 1), else_=0)),
                        func.sum(case((is_finished, finished_tokens), else_=0)),
                        func.sum(
                            case(
                                (
                                    is_finished,
                                    func.coalesce(table.c.elapsed_seconds, 0),
                                ),
                                else_=0,
                            )
                        ),
                    ).group_by(table.c.model_name)
                ):
                    agg = models.setdefault(
                        str(name),
                        {
                            "requests": 0,
                            "finished": 0,
                            "open": 0,
                            "tokens": 0,
                            "elapsed_seconds": 0.0,
                        },
                    )
                    agg["requests"] += int(cnt or 0)
                    agg["finished"] += int(fin or 0)
                    agg["open"] += int(opn or 0)
                    agg["tokens"] += int(toks or 0)
                    agg["elapsed_seconds"] += float(secs or 0.0)

        # Mean elapsed time per finished request (over the cross-table totals).
        for agg in models.values():
            agg["avg_elapsed_seconds"] = agg["elapsed_seconds"] / agg["finished"] if agg["finished"] else 0.0

        return {
            "total": total,
            "finished": finished,
            "running": running,
            "errors": errors,
            "total_tokens": total_tokens,
            "providers": providers,
            "models": models,
        }

    # ----------------------------------------------------------------- cleanup
    def close(self) -> None:
        self.engine.dispose()

    def __enter__(self) -> "ExperimentDAO":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
