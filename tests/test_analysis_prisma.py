"""Tests for the PRISMA 2020 flow diagram: `base/analysis/prisma.py`, `analysis draw-prisma`, `analysis add-prisma`."""

import json
import shutil
import xml.etree.ElementTree as ET
from unittest.mock import Mock

import nbformat
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

import llmexer.base.analysis.prisma as prisma
from llmexer.base.analysis import COPIED_MODULES
from llmexer.base.experiment_layout import Manifest, default_files, save_manifest
from llmexer.cli import app
from llmexer.common import strip_code_fence as shared_strip_code_fence
from llmexer.exceptions import UnexpectedCLIParamsException
from tests.db_helpers import OLLAMA_ROW, seed_db

runner = CliRunner()

PID = "prisma-exp"
RESULTS_COLUMNS = "title;doi;entry_source;language;pdf_filename;pdf_downloaded\n"


# ---------------------------------------------------------------------------
# A small project whose numbers are worked out by hand
# ---------------------------------------------------------------------------
#
# Search s1 (has a counts file): A, B (Semantic Scholar), C (OpenAlex), "Dee" (no DOI).
#   The counts file says Semantic Scholar 3, OpenAlex 2, of which 1 duplicate.
#   s1__filtered.csv drops "Dee" (language).
# Search s2 (no counts file): B again (Semantic Scholar), E (OpenAlex).
#
# identified: Semantic Scholar 3 + 1 = 4, OpenAlex 2 + 1 = 3       -> 7
# duplicates: 1 between engines + (6 rows - 5 unique) between searches -> 2
# automation: unique before {a, b, c, dee, e} - after {a, b, c, e}   -> 1
# screened (default): 4                                              -> 7 - 2 - 1 = 4
#
# Screening (field "include", three models):
#   S01 (a) yes yes yes | S02 (b) yes no no | S03 (c) no no no | S04 (e) unparsable x3
#   majority: excluded S02, S03 -> 2; any: S03 -> 1; all: S02, S03 -> 2; S04 undecided -> kept
# Only A has its PDF, so of the kept S01 and S04, S04 is not retrieved -> 1 (without full text).
# With the full-text experiment, both kept records were assessed -> not retrieved 0.
#
# Full text (field "eligible", reason "reason"): S01 yes, S04 no ("Wrong population").


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _answer_rows(answers_by_id: dict, field: str, extra=None) -> list:
    rows = []
    models = ["m1", "m2", "m3"]
    for data_id, answers in answers_by_id.items():
        for model, answer in zip(models, answers):
            payload = answer if not isinstance(answer, str) or not answer.startswith("{") else answer
            if isinstance(answer, str) and answer in ("yes", "no"):
                body = {field: answer}
                if extra and answer == "no":
                    body.update(extra)
                payload = "```json\n" + json.dumps(body) + "\n```"
            rows.append(
                {
                    **OLLAMA_ROW,
                    "ID": len(rows) + 1,
                    "code": f"{data_id}_prompt01_{model}_p-{model}",
                    "data_id": data_id,
                    "model_name": model,
                    "profile_name": f"p-{model}",
                    "response_text": payload,
                    "status": "success",
                }
            )
    return rows


@pytest.fixture()
def projects_dir(tmp_path, monkeypatch):
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "PROJECTS_PATH", str(tmp_path))
    monkeypatch.setattr("llmexer.cli.load_dotenv", Mock(return_value=True))
    monkeypatch.setenv("COLUMNS", "250")
    return tmp_path


@pytest.fixture()
def project(projects_dir):
    root = projects_dir / PID
    searches = root / "searches"
    _write(
        searches / "s1__results.csv",
        RESULTS_COLUMNS
        + "A;10.1/a;Semantic Scholar;en;a.pdf;True\n"
        + "B;10.1/b;Semantic Scholar;en;b.pdf;False\n"
        + "C;10.1/c;OpenAlex;en;c.pdf;False\n"
        + "Dee;;Semantic Scholar;de;d.pdf;False\n",
    )
    _write(
        searches / "s1__filtered.csv",
        RESULTS_COLUMNS
        + "A;10.1/a;Semantic Scholar;en;a.pdf;True\n"
        + "B;10.1/b;Semantic Scholar;en;b.pdf;False\n"
        + "C;10.1/c;OpenAlex;en;c.pdf;False\n",
    )
    _write(
        searches / "s2__results.csv",
        RESULTS_COLUMNS + "B;10.1/b;Semantic Scholar;en;b.pdf;False\n" + "E;10.1/e;OpenAlex;en;e.pdf;False\n",
    )
    _write(
        searches / "logs" / "s1__counts.json",
        json.dumps({"semantic_scholar": 3, "openalex": 2, "openalex_duplicates": 1}),
    )
    _write(
        searches / "logs" / "filters-applied.log",
        "2026-11-24 10:00:00 search file: s1__filtered.csv; filter applied: language=de ; "
        "input rows: 4; output rows: 3\n",
    )
    _write(root / "papers" / "a.pdf", "%PDF")
    # A merged file must never be counted as a search of its own.
    _write(searches / f"{PID}__merged_results.csv", RESULTS_COLUMNS)

    experiments = root / "experiments"
    _write(
        experiments / "abstracts" / "data.csv",
        "ID;Title;Abstract;year;doi;authors\n"
        "S01;A;x;2024;10.1/a;Ann\nS02;B;x;2024;10.1/b;Bo\nS03;C;x;2024;10.1/c;Cy\nS04;E;x;2024;10.1/e;Ed\n",
    )
    seed_db(
        experiments / "abstracts" / "experiment_20261124_01.db",
        {
            "ollama": _answer_rows(
                {
                    "S01": ["yes", "yes", "yes"],
                    "S02": ["yes", "no", "no"],
                    "S03": ["no", "no", "no"],
                    "S04": ["not json", "maybe?", "{broken"],
                },
                "include",
            )
        },
    )
    _write(
        experiments / "fulltext" / "data.csv",
        "ID;Title;Abstract;year;doi;authors\nS01;A;x;2024;10.1/a;Ann\nS04;E;x;2024;10.1/e;Ed\n",
    )
    seed_db(
        experiments / "fulltext" / "experiment_20261124_01.db",
        {
            "ollama": _answer_rows(
                {"S01": ["yes", "yes", "yes"], "S04": ["no", "no", "no"]},
                "eligible",
                extra={"reason": "Wrong population"},
            )
        },
    )
    for name in ("abstracts", "fulltext"):
        _write(experiments / name / "mapping.csv", "data_id;prompt_id\n")
        _write(experiments / name / "llms-for-experiment.csv", "provider;model_name;profile_name;notes\n")
    _write(experiments / "llm-params.csv", "provider;model_name;profile_name\n")
    save_manifest(
        str(experiments),
        Manifest(default="abstracts", experiments={"abstracts": default_files(), "fulltext": default_files()}),
    )
    return root


def _exp(project, name):
    return project / "experiments" / name / "data.csv", project / "experiments" / name / "experiment_20261124_01.db"


# ---------------------------------------------------------------------------
# Copies stay in step with llmexer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ['```json\n{"a": 1}\n```', "plain", "  ```\nx\n```  ", ""])
def test_strip_code_fence_matches_the_shared_helper(text):
    assert prisma.strip_code_fence(text) == shared_strip_code_fence(text)


def test_backup_naming_matches_llmexer(tmp_path):
    from llmexer.common import next_backup_name
    from llmexer.constants import ANALYSIS_BACKUP_DIR

    (tmp_path / "prisma_backup_x_03.svg").write_text("", encoding="utf-8")
    for _ in range(2):
        name = prisma.backup_name(tmp_path, "prisma", ".svg")
        assert name == next_backup_name(str(tmp_path), "prisma", ".svg")
        (tmp_path / name).write_text("", encoding="utf-8")
    assert prisma.BACKUP_DIR == ANALYSIS_BACKUP_DIR


def test_save_outputs_backs_up_only_changed_files(tmp_path):
    counts, svg = tmp_path / "prisma_counts.yaml", tmp_path / "prisma.svg"
    values = {key: 1 for key in prisma.BOXES}

    assert prisma.save_outputs(counts, svg, {}, {}, values) == []
    assert prisma.save_outputs(counts, svg, {}, {}, values) == []

    backups = prisma.save_outputs(counts, svg, {}, {}, values, title="New title")

    assert len(backups) == 1
    assert backups[0].parent == tmp_path / ".backup"
    assert backups[0].name.startswith("prisma_backup_") and backups[0].name.endswith("_01.svg")
    assert "New title" not in backups[0].read_text(encoding="utf-8")
    assert "New title" in svg.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "row",
    [{"doi": " 10.1/A ", "title": "x"}, {"doi": None, "title": " Some  Title "}, {"doi": "", "title": ""}],
)
def test_dedup_key_matches_search_merge(row):
    from llmexer.base.search import _dedup_key

    assert prisma.dedup_key(row) == _dedup_key(row)


def test_prisma_is_copied_into_analysis():
    assert "prisma.py" in COPIED_MODULES


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------


def test_list_searches_skips_merged(project):
    assert prisma.list_searches(project) == ["s1", "s2"]


def test_identified_by_database(project):
    by_database, note = prisma.identified_by_database(project, ["s1", "s2"])

    assert by_database == {"Semantic Scholar": 4, "OpenAlex": 3}
    assert "approximate for s2" in note


def test_duplicates_and_automation(project):
    duplicates, note = prisma.duplicates_removed(project, ["s1", "s2"])
    automation, automation_note = prisma.removed_by_automation(project, ["s1", "s2"])

    assert duplicates == 2
    assert "1 between the search engines" in note
    assert automation == 1
    assert "language=de" in automation_note
    assert prisma.records_after_identification(project, ["s1", "s2"]) == 4


# ---------------------------------------------------------------------------
# Screening decisions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "votes, rule, decision",
    [
        ([True, False, False], "majority", prisma.EXCLUDED),
        ([True, False], "majority", prisma.KEPT),
        ([True, False, False], "any", prisma.KEPT),
        ([True, True, False], "all", prisma.EXCLUDED),
        ([True, True, None], "all", prisma.KEPT),
        ([None, None], "majority", prisma.UNDECIDED),
        ([], "any", prisma.UNDECIDED),
    ],
)
def test_decide(votes, rule, decision):
    assert prisma.decide(votes, rule) == decision


def test_decide_rejects_unknown_rule():
    with pytest.raises(ValueError, match="rule must be"):
        prisma.decide([True], "most")


@pytest.mark.parametrize("value, vote", [("Yes", True), (" no ", False), (True, True), (0, False), ("maybe", None)])
def test_to_vote(value, vote):
    assert prisma.to_vote(value) is vote


def test_answer_field_reaches_nested_values():
    assert prisma.answer_field({"screening": {"include": "yes"}}, "screening.include") == "yes"
    assert prisma.answer_field({"a": 1}, "screening.include") is None


@pytest.mark.parametrize("rule, excluded", [("majority", {"S02", "S03"}), ("any", {"S03"}), ("all", {"S02", "S03"})])
def test_screen_rules(project, rule, excluded):
    data_csv, db = _exp(project, "abstracts")

    decisions, note = prisma.screen(data_csv, db, "include", rule)

    assert set(decisions.loc[decisions["decision"] == prisma.EXCLUDED, "data_id"]) == excluded
    s04 = decisions.set_index("data_id").loc["S04"]
    assert (s04["decision"], s04["unusable"]) == (prisma.UNDECIDED, 3)
    assert "1 undecided" in note


def test_screen_unknown_field_lists_the_fields(project):
    data_csv, db = _exp(project, "abstracts")
    with pytest.raises(ValueError, match="Fields found: include"):
        prisma.screen(data_csv, db, "relevant", "majority")


# ---------------------------------------------------------------------------
# Which answers count (filters) and manual decisions
# ---------------------------------------------------------------------------


def _answers(project, name="abstracts"):
    return prisma.load_answers(_exp(project, name)[1])


def test_filter_answers_by_model(project):
    answers = prisma.filter_answers(_answers(project), {"model": ["m2", " m3 "]})

    assert set(answers["model_name"]) == {"m2", "m3"}


def test_filter_answers_prompt_suffix_and_provider_case(project):
    answers = prisma.filter_answers(_answers(project), {"prompt": ["prompt01.txt"], "provider": ["OLLAMA"]})

    assert len(answers) == len(_answers(project))


def test_filter_answers_rejects_an_unknown_value(project):
    with pytest.raises(ValueError, match="No answer has model 'm9'. Found: m1, m2, m3"):
        prisma.filter_answers(_answers(project), {"model": ["m1", "m9"]})


def test_screen_counts_only_the_filtered_answers(project):
    data_csv, db = _exp(project, "abstracts")

    # m1 said yes to S02; m2 and m3 said no.
    only_m1, note = prisma.screen(data_csv, db, "include", "majority", filters={"model": ["m1"]})
    without_m1, _ = prisma.screen(data_csv, db, "include", "majority", filters={"model": ["m2", "m3"]})

    assert only_m1.set_index("data_id").loc["S02", "decision"] == prisma.KEPT
    assert without_m1.set_index("data_id").loc["S02", "decision"] == prisma.EXCLUDED
    assert "only model 'm1'" in note and "1 model/profile(s)" in note


def test_screen_returns_the_decision_columns(project):
    data_csv, db = _exp(project, "abstracts")

    decisions, _ = prisma.screen(data_csv, db, "include", "majority")

    assert list(decisions.columns) == prisma.DECISION_COLUMNS
    s02 = decisions.set_index("data_id").loc["S02"]
    assert (s02["stage"], s02["title"], s02["yes"], s02["no"]) == ("screening", "B", 1, 2)
    assert s02["llm_decision"] == s02["decision"] == prisma.EXCLUDED


def _manual(*rows):
    return pd.DataFrame(rows, columns=["stage", "data_id", "manual", "manual_reason"])


def test_manual_decision_wins(project):
    data_csv, db = _exp(project, "abstracts")

    decisions, note = prisma.screen(
        data_csv,
        db,
        "include",
        "majority",
        manual=_manual(("screening", "S03", "yes", ""), ("fulltext", "S02", "no", "")),
    )

    by_id = decisions.set_index("data_id")
    assert (by_id.loc["S03", "llm_decision"], by_id.loc["S03", "decision"]) == (prisma.EXCLUDED, prisma.KEPT)
    assert by_id.loc["S02", "decision"] == prisma.EXCLUDED  # the full-text row is another stage
    assert "1 manual decision(s)" in note


def test_manual_reason_counts_at_full_text(project):
    manual = _manual(("fulltext", "S01", "no", "Wrong outcome"), ("fulltext", "S04", "", "Wrong design"))
    counts, decisions = _derive_all(project, manual=manual)

    assert counts["excluded_reasons"]["value"] == {"Wrong design": 1, "Wrong outcome": 1}
    assert counts["included"]["value"] == 0
    s04 = decisions[(decisions["stage"] == "fulltext") & (decisions["data_id"] == "S04")].iloc[0]
    assert s04["llm_reason"] == "Wrong population"


def test_derive_keeps_manual_rows_of_unknown_records(project):
    _, decisions = _derive_all(project, manual=_manual(("screening", "S99", "no", "")))

    assert list(decisions["stage"].unique()) == ["screening", "fulltext"]
    last = decisions.iloc[-1]
    assert (last["data_id"], last["manual"], last["decision"]) == ("S99", "no", "")
    assert len(decisions) == 4 + 2 + 1


def test_decisions_file_round_trip(project, tmp_path):
    _, decisions = _derive_all(project)
    decisions.loc[decisions["data_id"] == "S02", "manual"] = "Yes"
    path = tmp_path / "prisma_decisions.csv"
    path.write_text(prisma.decisions_file_text(decisions), encoding="utf-8")

    manual = prisma.load_decisions_file(path)

    assert manual.to_dict("records") == [{"stage": "screening", "data_id": "S02", "manual": "Yes", "manual_reason": ""}]


def test_decisions_file_text_uses_plain_newlines(project, monkeypatch):
    """On Windows os.linesep is "\\r\\n"; written in text mode that would become "\\r\\r\\n"."""

    _, decisions = _derive_all(project)
    monkeypatch.setattr(prisma.os, "linesep", "\r\n")

    assert "\r" not in prisma.decisions_file_text(decisions)


def test_decisions_file_rejects_a_value_that_is_not_yes_or_no(tmp_path):
    path = tmp_path / "prisma_decisions.csv"
    path.write_text("stage;data_id;manual;manual_reason\nscreening;S01;maybe;\n", encoding="utf-8")

    with pytest.raises(ValueError, match="screening S01: 'maybe'"):
        prisma.load_decisions_file(path)


def test_settings_round_trip(tmp_path):
    path = tmp_path / "prisma_counts.yaml"
    settings = {"exp": None, "field": "include", "llm_vote_rule": "any", "filter_model": ["m1"], "prompt": []}
    path.write_text(prisma.counts_file_text({}, {}, settings, {"llmexer_version": "0.6.0"}), encoding="utf-8")

    assert prisma.load_settings(path) == {"field": "include", "llm_vote_rule": "any", "filter_model": ["m1"]}
    assert prisma.load_counts_file(path) == ({}, {})


def test_settings_reject_unknown_keys(tmp_path):
    path = tmp_path / "prisma_counts.yaml"
    path.write_text("settings:\n  feild: include\n", encoding="utf-8")

    with pytest.raises(ValueError, match="feild"):
        prisma.load_settings(path)


def test_answer_filters_per_stage():
    settings = {"prompt": ["p1"], "fulltext_prompt": ["p2"], "filter_model": ["m1"]}

    assert prisma.answer_filters(settings)["prompt"] == ["p1"]
    assert prisma.answer_filters(settings, prisma.STAGE_FULLTEXT) == {
        "prompt": ["p2"],
        "provider": [],
        "model": ["m1"],
        "profile": [],
    }


def _derive_all(project, manual=None, **options):
    screening_data, screening_db = _exp(project, "abstracts")
    fulltext_data, fulltext_db = _exp(project, "fulltext")
    return prisma.derive(
        project,
        screening_data=screening_data,
        screening_db=screening_db,
        field="include",
        fulltext_data=fulltext_data,
        fulltext_db=fulltext_db,
        fulltext_field="eligible",
        reason_field="reason",
        manual=manual,
        **options,
    )


# ---------------------------------------------------------------------------
# All boxes
# ---------------------------------------------------------------------------


def _derive(project, rule="majority", fulltext=True):
    screening_data, screening_db = _exp(project, "abstracts")
    fulltext_data, fulltext_db = _exp(project, "fulltext")
    return prisma.derive_counts(
        project,
        screening_data=screening_data,
        screening_db=screening_db,
        field="include",
        rule=rule,
        fulltext_data=fulltext_data if fulltext else None,
        fulltext_db=fulltext_db if fulltext else None,
        fulltext_field="eligible" if fulltext else None,
        reason_field="reason" if fulltext else None,
    )


def test_derive_counts(project):
    values = prisma.effective_counts(_derive(project), {})

    assert values == {
        "identified_databases": {"Semantic Scholar": 4, "OpenAlex": 3},
        "duplicates": 2,
        "automation": 1,
        "other_removed": 0,
        "screened": 4,
        "screen_excluded": 2,
        "sought": 2,
        "not_retrieved": 0,
        "assessed": 2,
        "excluded_reasons": {"Wrong population": 1},
        "included": 1,
    }


def test_derive_counts_without_fulltext_derives_assessed_only(project):
    values = prisma.effective_counts(_derive(project, fulltext=False), {})

    assert values["not_retrieved"] == 1
    assert values["assessed"] == 1
    assert values["excluded_reasons"] is None and values["included"] is None


def test_derive_counts_without_searches_or_experiment(projects_dir):
    (projects_dir / "empty").mkdir()

    counts = prisma.derive_counts(projects_dir / "empty")

    assert counts["identified_databases"]["value"] is None
    assert counts["other_removed"]["value"] == 0


def test_consistency_warnings():
    values = {"identified_databases": {"x": 5}, "duplicates": 1, "automation": 0, "other_removed": 0, "screened": 9}
    values.update({"screen_excluded": 10, "sought": 3, "not_retrieved": 1, "assessed": 1})
    values.update({"excluded_reasons": {"r": 1}, "included": 5})

    warnings = prisma.check_counts(values)

    assert len(warnings) == 4


def test_consistent_project_has_no_warnings(project):
    assert prisma.check_counts(prisma.effective_counts(_derive(project), {})) == []


def test_counts_file_round_trip_keeps_overrides(tmp_path):
    path = tmp_path / "prisma_counts.yaml"
    prisma.save_counts_file(path, {"screened": {"value": 4, "source": "x"}}, {"included": 3})

    derived, overrides = prisma.load_counts_file(path)

    assert derived["screened"]["value"] == 4
    assert overrides == {"included": 3}
    assert prisma.effective_counts(derived, overrides)["included"] == 3
    assert path.read_text(encoding="utf-8").startswith("# PRISMA 2020 counts")


def test_counts_file_rejects_unknown_override(tmp_path):
    path = tmp_path / "prisma_counts.yaml"
    path.write_text("overrides:\n  incuded: 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incuded"):
        prisma.load_counts_file(path)


# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------


def _texts(svg):
    root = ET.fromstring(svg)
    return [element.text for element in root.iter("{http://www.w3.org/2000/svg}text")]


def test_svg_is_valid_and_shows_every_box(project):
    svg = prisma.render_svg(prisma.effective_counts(_derive(project), {}), title="Review & <test>")

    texts = _texts(svg)
    assert "Review & <test>" in texts
    for expected in (
        "Databases (n = 7)",
        "Semantic Scholar (n = 4)",
        "Duplicate records removed (n = 2)",
        "Records screened (n = 4)",
        "Records excluded (n = 2)",
        "Reports not retrieved (n = 0)",
        "Wrong population (n = 1)",
        "Studies included in review (n = 1)",
    ):
        assert expected in texts, expected
    for band in ("Identification", "Screening", "Included"):
        assert band in texts


def test_svg_marks_missing_values():
    svg = prisma.render_svg({key: None for key in prisma.BOXES})

    assert "Records screened (n = ?)" in _texts(svg)
    assert 'stroke-dasharray="6 4"' in svg


# ---------------------------------------------------------------------------
# CLI: analysis draw-prisma
# ---------------------------------------------------------------------------


def _prisma(*options, group="analysis"):
    return runner.invoke(app, [group, "draw-prisma", "--pid", PID, *options])


def test_cli_writes_counts_and_svg(project):
    result = _prisma(
        "--field", "include", "--fulltext-exp", "fulltext", "--fulltext-field", "eligible", "--reason-field", "reason"
    )

    assert result.exit_code == 0, (result.output, result.exception)
    analysis = project / "analysis"
    svg = (analysis / "abstracts__prisma.svg").read_text(encoding="utf-8")
    assert "Studies included in review (n = 1)" in _texts(svg)
    counts = yaml.safe_load((analysis / "abstracts__prisma_counts.yaml").read_text(encoding="utf-8"))
    assert counts["derived"]["screen_excluded"]["value"] == 2
    assert counts["overrides"] == {}
    assert "PRISMA 2020 counts" in result.output


def test_cli_keeps_overrides_on_rerun(project):
    _prisma()
    counts_path = project / "analysis" / "abstracts__prisma_counts.yaml"
    content = yaml.safe_load(counts_path.read_text(encoding="utf-8"))
    content["overrides"] = {"screen_excluded": 3, "included": 1}
    counts_path.write_text(yaml.safe_dump(content), encoding="utf-8")

    result = _prisma()

    assert result.exit_code == 0, result.exception
    assert yaml.safe_load(counts_path.read_text(encoding="utf-8"))["overrides"] == {"screen_excluded": 3, "included": 1}
    texts = _texts((project / "analysis" / "abstracts__prisma.svg").read_text(encoding="utf-8"))
    assert "Records excluded (n = 3)" in texts


def test_cli_backs_up_changed_outputs(project):
    analysis = project / "analysis"
    _prisma()
    counts_path = analysis / "abstracts__prisma_counts.yaml"
    content = yaml.safe_load(counts_path.read_text(encoding="utf-8"))
    content["overrides"] = {"included": 1}
    counts_path.write_text(yaml.safe_dump(content), encoding="utf-8")
    old_counts = counts_path.read_text(encoding="utf-8")
    old_svg = (analysis / "abstracts__prisma.svg").read_text(encoding="utf-8")

    result = _prisma()

    assert result.exit_code == 0, result.exception
    backups = sorted(path.name for path in (analysis / ".backup").iterdir())
    assert len(backups) == 2
    counts_backup = next(name for name in backups if name.startswith("abstracts__prisma_counts_backup_"))
    svg_backup = next(name for name in backups if name.startswith("abstracts__prisma_backup_"))
    assert counts_backup.endswith("_01.yaml") and svg_backup.endswith("_01.svg")
    assert (analysis / ".backup" / counts_backup).read_text(encoding="utf-8") == old_counts
    assert (analysis / ".backup" / svg_backup).read_text(encoding="utf-8") == old_svg
    assert "Backed up" in result.output


def test_cli_rerun_without_changes_makes_no_backup(project):
    _prisma()
    result = _prisma()

    assert result.exit_code == 0, result.exception
    assert not (project / "analysis" / ".backup").exists()


def test_cli_dry_run_names_the_backup(project):
    _prisma()
    counts_path = project / "analysis" / "abstracts__prisma_counts.yaml"
    counts_path.write_text(counts_path.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    edited = counts_path.read_text(encoding="utf-8")

    result = runner.invoke(app, ["--dry-run", "analysis", "draw-prisma", "--pid", PID])

    assert result.exit_code == 0, result.exception
    assert ".backup/abstracts__prisma_counts_backup_" in result.output.replace("\n", "")
    assert counts_path.read_text(encoding="utf-8") == edited
    assert not (project / "analysis" / ".backup").exists()


def test_cli_without_field_leaves_screening_empty(project):
    result = _prisma()

    assert result.exit_code == 0, result.exception
    texts = _texts((project / "analysis" / "abstracts__prisma.svg").read_text(encoding="utf-8"))
    assert "Records excluded (n = ?)" in texts
    assert "Records screened (n = 4)" in texts


def test_cli_dry_run_writes_nothing(project):
    result = runner.invoke(app, ["--dry-run", "analysis", "draw-prisma", "--pid", PID])

    assert result.exit_code == 0, result.exception
    assert not (project / "analysis").exists()


@pytest.mark.parametrize(
    "options, match",
    [
        (["--field", "relevant"], "Fields found: include"),
        (["--search", "nope"], "nope"),
        (["--fulltext-exp", "fulltext"], "--fulltext-field"),
        (["--reason-field", "reason"], "--fulltext-exp"),
    ],
)
def test_cli_rejects_bad_options(project, options, match):
    result = _prisma(*options)

    assert isinstance(result.exception, UnexpectedCLIParamsException)
    assert match in str(result.exception)
    assert not (project / "analysis").exists()


def test_cli_selects_searches(project):
    result = _prisma("--search", "s2")

    assert result.exit_code == 0, result.exception
    texts = _texts((project / "analysis" / "abstracts__prisma.svg").read_text(encoding="utf-8"))
    assert "Databases (n = 2)" in texts


@pytest.mark.parametrize("group", ["analysis", "analyse", "analyze"])
def test_cli_aliases(project, group):
    assert _prisma(group=group).exit_code == 0


DECISIONS = "abstracts__prisma_decisions.csv"
COUNTS = "abstracts__prisma_counts.yaml"


def _counts_file(project):
    return yaml.safe_load((project / "analysis" / COUNTS).read_text(encoding="utf-8"))


def test_cli_saves_the_settings_and_reuses_them(project):
    first = _prisma("--field", "include", "--llm-vote-rule", "any", "--filter-model", "m1,m2", "--title", "My review")
    assert first.exit_code == 0, (first.output, first.exception)
    saved = _counts_file(project)

    assert saved["settings"] == {
        "exp": "abstracts",
        "field": "include",
        "llm_vote_rule": "any",
        "filter_model": ["m1", "m2"],
        "title": "My review",
    }
    assert saved["inputs"]["screening_db"] == "experiments/abstracts/experiment_20261124_01.db"
    assert "llmexer_version" in saved["inputs"]

    again = _prisma()

    assert again.exit_code == 0, again.exception
    assert "Settings from abstracts__prisma_counts.yaml" in again.output
    assert "--field include" in again.output and "--filter-model m1,m2" in again.output
    assert _counts_file(project) == saved
    assert not (project / "analysis" / ".backup").exists()


def test_cli_option_replaces_a_saved_setting(project):
    _prisma("--field", "include", "--llm-vote-rule", "any")

    result = _prisma("--llm-vote-rule", "all")

    assert result.exit_code == 0, result.exception
    counts = _counts_file(project)
    assert counts["settings"]["llm_vote_rule"] == "all"
    assert counts["derived"]["screen_excluded"]["value"] == 2  # S02 and S03


def test_cli_ignore_settings(project):
    _prisma("--field", "include")

    result = _prisma("--ignore-settings")

    assert result.exit_code == 0, result.exception
    assert "Settings from" not in result.output
    assert "field" not in _counts_file(project)["settings"]
    assert _counts_file(project)["derived"]["screen_excluded"]["value"] is None


def test_cli_filters_change_the_decisions(project):
    result = _prisma("--field", "include", "--filter-model", "m1")

    assert result.exit_code == 0, result.exception
    assert _counts_file(project)["derived"]["screen_excluded"]["value"] == 1  # only S03


def test_cli_rejects_an_unknown_filter_value(project):
    result = _prisma("--field", "include", "--prompt", "screening_v9")

    assert isinstance(result.exception, UnexpectedCLIParamsException)
    assert "No answer has prompt 'screening_v9'. Found: prompt01" in str(result.exception)


def test_cli_fulltext_prompt_needs_fulltext_exp(project):
    result = _prisma("--fulltext-prompt", "p1")

    assert isinstance(result.exception, UnexpectedCLIParamsException)
    assert "need --fulltext-exp" in str(result.exception)


def test_cli_writes_decisions_and_applies_manual_ones(project):
    analysis = project / "analysis"
    _prisma("--field", "include")
    decisions_path = analysis / DECISIONS
    text = decisions_path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == ";".join(prisma.DECISION_COLUMNS)
    lines = text.splitlines()
    lines = [
        line.replace(";excluded;;;;excluded", ";excluded;;yes;;excluded") if line.startswith("screening;S03;") else line
        for line in lines
    ]
    decisions_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = _prisma()

    assert result.exit_code == 0, result.exception
    assert _counts_file(project)["derived"]["screen_excluded"]["value"] == 1
    manual = prisma.load_decisions_file(decisions_path)
    assert manual.to_dict("records") == [{"stage": "screening", "data_id": "S03", "manual": "yes", "manual_reason": ""}]
    assert any(path.name.startswith("abstracts__prisma_decisions_backup_") for path in (analysis / ".backup").iterdir())


def test_cli_without_field_writes_no_decisions(project):
    assert _prisma().exit_code == 0
    assert not (project / "analysis" / DECISIONS).exists()


def test_cli_rejects_a_bad_manual_value(project):
    _write(project / "analysis" / DECISIONS, "stage;data_id;manual;manual_reason\nscreening;S01;perhaps;\n")

    result = _prisma("--field", "include")

    assert isinstance(result.exception, UnexpectedCLIParamsException)
    assert "'perhaps'" in str(result.exception)


# ---------------------------------------------------------------------------
# CLI: analysis add-prisma
# ---------------------------------------------------------------------------

NOTEBOOK = "abstracts__analyse_prisma.ipynb"


def _add(*options):
    return runner.invoke(app, ["analysis", "add-prisma", "--pid", PID, *options])


def test_add_prisma_writes_a_valid_notebook(project):
    result = _add("--field", "include", "--llm-vote-rule", "any")

    assert result.exit_code == 0, (result.output, result.exception)
    analysis = project / "analysis"
    notebook = nbformat.read(str(analysis / NOTEBOOK), as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            assert cell.outputs == []
            compile(cell.source, f"cell-{index}", "exec")
    for module in COPIED_MODULES:
        assert (analysis / module).is_file()

    config = next(cell.source for cell in notebook.cells if "SCREENING_DB =" in cell.source)
    assert "FIELD = 'include'" in config
    assert "LLM_VOTE_RULE = 'any'" in config
    assert "SCREENING_DATA = 'experiments/abstracts/data.csv'" in config
    assert "COUNTS_FILE = ANALYSIS_DIR / 'abstracts__prisma_counts.yaml'" in config


def test_add_prisma_keeps_then_rewrites(project):
    _add()
    notebook = project / "analysis" / NOTEBOOK
    notebook.write_text("{}", encoding="utf-8")

    kept = _add()
    assert "already exists" in kept.output
    assert notebook.read_text(encoding="utf-8") == "{}"

    _add("--rewrite")
    assert notebook.read_text(encoding="utf-8") != "{}"
    assert any((project / "analysis" / ".backup").iterdir())


def test_notebook_cells_run_against_the_project(project, monkeypatch):
    """Execute the code cells in order, as Jupyter would, inside analysis/."""

    _add("--field", "include", "--fulltext-exp", "fulltext", "--fulltext-field", "eligible", "--reason-field", "reason")
    analysis = project / "analysis"
    monkeypatch.chdir(analysis)
    monkeypatch.setitem(__import__("sys").modules, "prisma", prisma)

    displayed = []
    fake_display = Mock(side_effect=displayed.append)
    monkeypatch.setattr("IPython.display.display", fake_display)

    namespace = _run_notebook_cells(analysis / NOTEBOOK)

    assert (analysis / "abstracts__prisma.svg").is_file()
    assert namespace["values"]["included"] == 1
    assert (analysis / "abstracts__prisma_counts.yaml").is_file()
    assert not (analysis / ".backup").exists()

    counts_path = analysis / "abstracts__prisma_counts.yaml"
    content = yaml.safe_load(counts_path.read_text(encoding="utf-8"))
    content["overrides"] = {"included": 5}
    counts_path.write_text(yaml.safe_dump(content), encoding="utf-8")
    edited = counts_path.read_text(encoding="utf-8")

    namespace = _run_notebook_cells(analysis / NOTEBOOK)

    assert namespace["values"]["included"] == 5
    backups = {path.name: path for path in (analysis / ".backup").iterdir()}
    assert {name.split("_backup_")[0] for name in backups} == {"abstracts__prisma_counts", "abstracts__prisma"}
    counts_backup = next(path for name, path in backups.items() if name.endswith(".yaml"))
    assert counts_backup.read_text(encoding="utf-8") == edited


def _run_notebook_cells(notebook_path) -> dict:
    """Execute the code cells in order, as Jupyter would."""

    notebook = nbformat.read(str(notebook_path), as_version=4)
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, "cell", "exec"), namespace)  # nosec B102 - our own notebook
    return namespace


def test_add_prisma_injects_the_filters_and_reuses_settings(project):
    _prisma("--field", "include", "--filter-model", "m1", "--title", "My review")

    result = _add("--prompt", "prompt01")

    assert result.exit_code == 0, (result.output, result.exception)
    assert "Settings from" in result.output
    notebook = nbformat.read(str(project / "analysis" / NOTEBOOK), as_version=4)
    config = next(cell.source for cell in notebook.cells if "SCREENING_DB =" in cell.source)
    assert "FIELD = 'include'" in config
    assert "FILTER_MODELS = ['m1']" in config
    assert "PROMPTS = ['prompt01']" in config
    assert "TITLE = 'My review'" in config
    assert "DECISIONS_FILE = ANALYSIS_DIR / 'abstracts__prisma_decisions.csv'" in config


def test_notebook_and_command_write_the_same_files(project, monkeypatch):
    options = ("--field", "include", "--filter-model", "m1,m2", "--fulltext-exp", "fulltext")
    options += ("--fulltext-field", "eligible", "--reason-field", "reason")
    assert _prisma(*options).exit_code == 0
    analysis = project / "analysis"
    decisions_path = analysis / DECISIONS
    table = pd.read_csv(decisions_path, sep=";", dtype=str, keep_default_na=False)
    table.loc[(table["stage"] == "fulltext") & (table["data_id"] == "S04"), "manual"] = "yes"
    table.to_csv(decisions_path, sep=";", index=False)
    assert _prisma().exit_code == 0
    before = {
        name: (analysis / name).read_text(encoding="utf-8") for name in (COUNTS, DECISIONS, "abstracts__prisma.svg")
    }
    shutil.rmtree(analysis / ".backup")
    assert _add().exit_code == 0

    monkeypatch.chdir(analysis)
    monkeypatch.setitem(__import__("sys").modules, "prisma", prisma)
    monkeypatch.setattr("IPython.display.display", Mock())
    namespace = _run_notebook_cells(analysis / NOTEBOOK)

    assert namespace["values"]["included"] == 2  # S04 kept by hand
    assert {name: (analysis / name).read_text(encoding="utf-8") for name in before} == before
    assert not (analysis / ".backup").exists()
