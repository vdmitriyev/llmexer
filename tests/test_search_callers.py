"""Tests for the Europe PMC, PubMed, arXiv and CORE callers of `search run` (no network)."""

import json
from unittest.mock import Mock

import pytest

import llmexer.base.search_arxiv as search_arxiv
import llmexer.base.search_core as search_core
import llmexer.base.search_europepmc as search_europepmc
import llmexer.base.search_pubmed as search_pubmed
import llmexer.base.search_zenodo as search_zenodo
from llmexer.base.search_record import parse_year_range


def _response(json_data=None, text=None):
    response = Mock()
    response.raise_for_status = Mock()
    response.json.return_value = json_data
    response.text = text
    return response


@pytest.fixture()
def session(monkeypatch):
    """One mocked HTTP session for every caller; set ``session.get.side_effect`` per test."""

    mock_session = Mock()
    mock_session.headers = {}
    monkeypatch.setattr("llmexer.base.search_record.make_http_session", Mock(return_value=mock_session))
    monkeypatch.setattr("llmexer.base.search_record.detect_publication_lang", Mock(return_value="en"))
    monkeypatch.setattr("llmexer.base.search_pubmed.time.sleep", Mock())
    monkeypatch.setattr("llmexer.base.search_arxiv.time.sleep", Mock())
    monkeypatch.delenv("SEARCH_MAX_RESPONSES", raising=False)
    return mock_session


@pytest.mark.parametrize(
    "year, expected",
    [
        ("2020-2025", ("2020", "2025")),
        ("2023", ("2023", "2023")),
        ("2020-", ("2020", "2999")),
        ("-2020", ("1800", "2020")),
        ("", None),
        (None, None),
    ],
)
def test_parse_year_range(year, expected):
    assert parse_year_range(year) == expected


# ---------------------------------------------------------------------------
# Europe PMC
# ---------------------------------------------------------------------------


def _epmc_result(index, **extra):
    result = {
        "id": str(index),
        "source": "MED",
        "title": f"Paper {index}",
        "abstractText": "<h4>Background</h4>Air <i>quality</i> sensors.",
        "authorList": {"author": [{"fullName": "Doe J"}, {"firstName": "Ann", "lastName": "Lee"}]},
        "pubYear": "2024",
        "doi": f"10.1/{index}",
        "isOpenAccess": "Y",
        "citedByCount": 3,
    }
    result.update(extra)
    return result


def test_europepmc_query_with_year_and_open_access():
    query = search_europepmc.build_europepmc_query('"air quality" + sensor', "2020-2025", True)

    assert query == 'TITLE_ABS:("air quality" AND sensor) AND PUB_YEAR:[2020 TO 2025] AND OPEN_ACCESS:y'


def test_europepmc_maps_fields_and_pages_with_the_cursor(session, tmp_path):
    session.get.side_effect = [
        _response(
            {"hitCount": 3, "nextCursorMark": "c1", "resultList": {"result": [_epmc_result(1), _epmc_result(2)]}}
        ),
        _response({"hitCount": 3, "nextCursorMark": "c2", "resultList": {"result": [_epmc_result(3)]}}),
    ]
    raw_path = tmp_path / "raw.json"

    records = search_europepmc.run_europepmc_search("sensor", "2024", False, 2, None, str(raw_path))

    assert [record["search_engine_internal_id"] for record in records] == ["MED:1", "MED:2", "MED:3"]
    first = records[0]
    assert first["abstract"] == "Background Air quality sensors."
    assert first["authors"] == "Doe J; Ann Lee"
    assert (first["year"], first["doi"], first["isOpenAccess"], first["citationCount"]) == (2024, "10.1/1", True, 3)
    assert first["entry_source"] == "Europe PMC"
    assert session.get.call_args_list[1].kwargs["params"]["cursorMark"] == "c1"
    assert len(json.loads(raw_path.read_text(encoding="utf-8"))) == 2


def test_europepmc_stops_at_the_limit(session, tmp_path):
    session.get.return_value = _response(
        {"hitCount": 50, "nextCursorMark": "c1", "resultList": {"result": [_epmc_result(i) for i in range(5)]}}
    )

    records = search_europepmc.run_europepmc_search("sensor", None, False, 5, 3, str(tmp_path / "raw.json"))

    assert len(records) == 3
    assert session.get.call_count == 1


# ---------------------------------------------------------------------------
# PubMed
# ---------------------------------------------------------------------------

PUBMED_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
<PubmedArticle>
  <MedlineCitation>
    <PMID Version="1">111</PMID>
    <Article>
      <Journal><JournalIssue><PubDate><MedlineDate>2021 Jan-Feb</MedlineDate></PubDate></JournalIssue></Journal>
      <ArticleTitle>Low-cost <i>sensors</i></ArticleTitle>
      <Abstract>
        <AbstractText Label="BACKGROUND">Air is dirty.</AbstractText>
        <AbstractText Label="RESULTS">Sensors help.</AbstractText>
      </Abstract>
      <AuthorList>
        <Author><LastName>Wang</LastName><ForeName>Xi</ForeName></Author>
        <Author><CollectiveName>Air Group</CollectiveName></Author>
      </AuthorList>
    </Article>
  </MedlineCitation>
  <PubmedData>
    <ArticleIdList><ArticleId IdType="pubmed">111</ArticleId><ArticleId IdType="doi">10.1/own</ArticleId></ArticleIdList>
    <ReferenceList><Reference><ArticleIdList><ArticleId IdType="doi">10.9/cited</ArticleId></ArticleIdList></Reference>
    </ReferenceList>
  </PubmedData>
</PubmedArticle>
<PubmedBookArticle><BookDocument><PMID>222</PMID></BookDocument></PubmedBookArticle>
</PubmedArticleSet>
"""


def test_pubmed_parses_an_article_and_skips_books():
    records, skipped = search_pubmed.pubmed_records(PUBMED_XML)

    assert skipped == 1
    record = records[0]
    assert record["search_engine_internal_id"] == "111"
    assert record["title"] == "Low-cost sensors"
    assert record["abstract"] == "BACKGROUND: Air is dirty. RESULTS: Sensors help."
    assert record["authors"] == "Xi Wang; Air Group"
    assert (record["year"], record["doi"]) == (2021, "10.1/own")


def test_pubmed_term_with_open_access():
    assert search_pubmed.build_pubmed_term("a + b", True) == "(a AND b) AND free full text[sb]"


def test_pubmed_searches_then_fetches_from_the_history(session, tmp_path, monkeypatch):
    monkeypatch.setenv("UNPAYWALL_EMAIL", "me@example.org")
    session.get.side_effect = [
        _response({"esearchresult": {"count": "3", "querykey": "1", "webenv": "W1"}}),
        _response(text=PUBMED_XML),
        _response(text=PUBMED_XML),
    ]
    raw_path = tmp_path / "raw.json"

    records = search_pubmed.run_pubmed_search("sensor", "2020-2025", False, 2, None, str(raw_path))

    search_params = session.get.call_args_list[0].kwargs["params"]
    assert (search_params["field"], search_params["usehistory"]) == ("tiab", "y")
    assert (search_params["mindate"], search_params["maxdate"], search_params["datetype"]) == ("2020", "2025", "pdat")
    assert search_params["email"] == "me@example.org" and search_params["tool"] == "llmexer"
    fetches = [call.kwargs["params"] for call in session.get.call_args_list[1:]]
    assert [(params["WebEnv"], params["retstart"]) for params in fetches] == [("W1", 0), ("W1", 2)]
    assert len(records) == 2  # one article per mocked page
    assert len(json.loads(raw_path.read_text(encoding="utf-8"))) == 3


def test_pubmed_with_no_hits_fetches_nothing(session, tmp_path):
    session.get.side_effect = [_response({"esearchresult": {"count": "0"}})]

    assert search_pubmed.run_pubmed_search("nothing", None, False, 100, None, str(tmp_path / "raw.json")) == []
    assert session.get.call_count == 1


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------


def _atom(entries: str, total: int) -> str:
    return (
        '<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom" '
        'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
        f"<opensearch:totalResults>{total}</opensearch:totalResults>{entries}</feed>"
    )


ARXIV_ENTRY = """<entry>
  <id>http://arxiv.org/abs/2101.00001v2</id>
  <published>2021-01-01T10:00:00Z</published>
  <title>Sensors
    for air</title>
  <summary>We measure air.</summary>
  <author><name>Ada Lovelace</name></author>
  <author><name>Alan Turing</name></author>
</entry>"""

ARXIV_ENTRY_WITH_DOI = """<entry>
  <id>http://arxiv.org/abs/hep-ex/0307015v1</id>
  <published>2003-07-07T00:00:00Z</published>
  <title>Old paper</title>
  <summary>Old.</summary>
  <arxiv:doi>10.1103/journal</arxiv:doi>
</entry>"""


def test_arxiv_maps_entries():
    records, total = search_arxiv.arxiv_records(_atom(ARXIV_ENTRY + ARXIV_ENTRY_WITH_DOI, 2))

    assert total == 2
    first, second = records
    assert first["search_engine_internal_id"] == "2101.00001"
    assert first["doi"] == "10.48550/arXiv.2101.00001"
    assert (first["title"], first["year"], first["authors"]) == ("Sensors for air", 2021, "Ada Lovelace; Alan Turing")
    assert first["isOpenAccess"] is True and first["entry_source"] == "arXiv"
    assert second["search_engine_internal_id"] == "hep-ex/0307015"
    assert second["doi"] == "10.1103/journal"


def test_arxiv_query_with_years():
    query = search_arxiv.build_arxiv_query("sensor", "2020-2021")

    assert query == "(ti:sensor OR abs:sensor) AND submittedDate:[202001010000 TO 202112312359]"


def test_arxiv_pages_and_waits_between_calls(session, tmp_path):
    session.get.side_effect = [
        _response(text=_atom(ARXIV_ENTRY, 2)),
        _response(text=_atom(ARXIV_ENTRY_WITH_DOI, 2)),
    ]

    records = search_arxiv.run_arxiv_search("sensor", None, False, 1, None, str(tmp_path / "raw.json"))

    assert len(records) == 2
    assert [call.kwargs["params"]["start"] for call in session.get.call_args_list] == [0, 1]
    search_arxiv.time.sleep.assert_called_once_with(search_arxiv.ARXIV_DELAY_SECONDS)


# ---------------------------------------------------------------------------
# CORE
# ---------------------------------------------------------------------------


def _core_work(index, **extra):
    work = {
        "id": index,
        "title": f"Work {index}",
        "abstract": "About sensors.",
        "authors": [{"name": "Lee, Ann"}, {"name": "Bo  Chen"}],
        "yearPublished": 2022,
        "doi": f"https://doi.org/10.2/{index}",
        "downloadUrl": f"https://core.ac.uk/download/{index}.pdf",
        "citationCount": 1,
    }
    work.update(extra)
    return work


def test_core_query():
    assert search_core.build_core_query('"air quality" + sensor', "2020-2025") == (
        '((title:"air quality" OR abstract:"air quality") AND (title:sensor OR abstract:sensor)) '
        "AND yearPublished>=2020 AND yearPublished<=2025"
    )


def test_core_sends_the_key_and_pages_by_offset(session, tmp_path, monkeypatch):
    monkeypatch.setenv("CORE_API_KEY", "secret")
    session.get.side_effect = [
        _response({"totalHits": 3, "results": [_core_work(1), _core_work(2)]}),
        _response({"totalHits": 3, "results": [_core_work(3, downloadUrl=None)]}),
    ]

    records = search_core.run_core_search("sensor", None, False, 2, None, str(tmp_path / "raw.json"))

    assert session.headers["Authorization"] == "Bearer secret"
    assert [call.kwargs["params"]["offset"] for call in session.get.call_args_list] == [0, 2]
    assert [record["doi"] for record in records] == ["10.2/1", "10.2/2", "10.2/3"]
    assert records[0]["authors"] == "Lee, Ann; Bo Chen"
    assert (records[0]["isOpenAccess"], records[2]["isOpenAccess"]) == (True, None)
    assert records[0]["entry_source"] == "CORE"


def test_core_only_open_access_drops_works_without_full_text(session, tmp_path):
    session.get.side_effect = [_response({"totalHits": 2, "results": [_core_work(1), _core_work(2, downloadUrl=None)]})]

    records = search_core.run_core_search("sensor", None, True, 10, None, str(tmp_path / "raw.json"))

    assert [record["search_engine_internal_id"] for record in records] == [1]


def test_core_has_its_own_cap(session, tmp_path, monkeypatch):
    monkeypatch.setenv("CORE_MAX_RESPONSES", "2")
    monkeypatch.setenv("SEARCH_MAX_RESPONSES", "100")
    session.get.side_effect = [_response({"totalHits": 50, "results": [_core_work(i) for i in range(5)]})]
    progress = []

    records = search_core.run_core_search("sensor", None, False, 5, None, str(tmp_path / "raw.json"), progress.append)

    assert len(records) == 2
    assert any("raise CORE_MAX_RESPONSES" in message for message in progress)


def test_shared_cap_does_not_limit_core(session, tmp_path, monkeypatch):
    monkeypatch.delenv("CORE_MAX_RESPONSES", raising=False)
    monkeypatch.setenv("SEARCH_MAX_RESPONSES", "1")
    session.get.side_effect = [_response({"totalHits": 3, "results": [_core_work(i) for i in range(3)]})]

    assert len(search_core.run_core_search("sensor", None, False, 5, None, str(tmp_path / "raw.json"))) == 3


# ---------------------------------------------------------------------------
# Own search string per engine (`queries:` in the search file)
# ---------------------------------------------------------------------------

OWN = 'title:"air quality" AND monitoring'


def test_own_string_replaces_only_the_terms_in_every_builder():
    assert search_core.build_core_query("ignored", "2020-2025", OWN) == (
        f"({OWN}) AND yearPublished>=2020 AND yearPublished<=2025"
    )
    assert search_europepmc.build_europepmc_query("ignored", "2020-2025", True, OWN) == (
        f"({OWN}) AND PUB_YEAR:[2020 TO 2025] AND OPEN_ACCESS:y"
    )
    assert search_pubmed.build_pubmed_term("ignored", True, '"air quality"[tiab]') == (
        '("air quality"[tiab]) AND free full text[sb]'
    )
    assert search_arxiv.build_arxiv_query("ignored", "2021", "ti:llm ANDNOT abs:survey") == (
        "(ti:llm ANDNOT abs:survey) AND submittedDate:[202101010000 TO 202112312359]"
    )


def test_pubmed_own_string_drops_the_title_abstract_field(session, tmp_path, caplog):
    session.get.side_effect = [_response({"esearchresult": {"count": "0"}})]

    with caplog.at_level("INFO", logger="llmexer"):
        search_pubmed.run_pubmed_search(
            "ignored", "2020-2025", False, 10, None, str(tmp_path / "raw.json"), custom_query="asthma[mh]"
        )

    params = session.get.call_args.kwargs["params"]
    assert params["term"] == "asthma[mh]" and "field" not in params
    assert params["mindate"] == "2020"
    assert "PubMed query (own string from queries:): asthma[mh] [publication date 2020-2025]" in caplog.text


def test_converted_query_is_logged(session, tmp_path, caplog):
    session.get.side_effect = [_response({"hitCount": 0, "resultList": {"result": []}})]

    with caplog.at_level("INFO", logger="llmexer"):
        search_europepmc.run_europepmc_search("a + b", None, False, 10, None, str(tmp_path / "raw.json"))

    assert "Europe PMC query (converted from query): TITLE_ABS:(a AND b)" in caplog.text


def test_semantic_scholar_and_openalex_use_the_own_string(tmp_path, monkeypatch, caplog):
    import llmexer.base.search_openalex as search_openalex
    import llmexer.base.search_semantic_scholar as search_semantic_scholar

    s2_session, oa_session = Mock(), Mock()
    s2_session.get.return_value = _response({"data": [], "token": None})
    oa_session.get.return_value = _response({"results": [], "meta": {"count": 0, "next_cursor": None}})
    monkeypatch.setattr(search_semantic_scholar, "make_http_session", Mock(return_value=s2_session))
    monkeypatch.setattr(search_openalex, "make_http_session", Mock(return_value=oa_session))
    progress = []

    with caplog.at_level("INFO", logger="llmexer"):
        search_semantic_scholar.run_semantic_scholar_search(
            "ignored",
            "2020-",
            False,
            10,
            None,
            str(tmp_path / "s2.json"),
            str(tmp_path / "s2.csv"),
            progress.append,
            custom_query='"air quality" +sensor',
        )
        search_openalex.run_openalex_search(
            "ignored",
            None,
            False,
            10,
            None,
            str(tmp_path / "oa.json"),
            "key",
            progress.append,
            custom_query="air AND quality",
        )

    assert s2_session.get.call_args.kwargs["params"]["query"] == '"air quality" +sensor'
    assert oa_session.get.call_args.kwargs["params"]["filter"] == "title_and_abstract.search:air AND quality"
    assert 'Semantic Scholar query (own string from queries:): "air quality" +sensor [year=2020-]' in caplog.text
    assert "OpenAlex query (own string from queries:): title_and_abstract.search:air AND quality" in caplog.text
    assert any(message.startswith("Semantic Scholar query (from queries:)") for message in progress)


def test_query_with_square_brackets_survives_the_console():
    """PubMed tags like [tiab] must not be eaten as Rich markup."""
    from rich.console import Console

    from llmexer.base.search_record import announce_query

    console = Console(record=True, width=200)
    announce_query("PubMed", '"air quality"[tiab] AND asthma[mh]', True, console.print)

    assert 'PubMed query (from queries:): "air quality"[tiab] AND asthma[mh]' in console.export_text()


# ---------------------------------------------------------------------------
# Zenodo
# ---------------------------------------------------------------------------


ZENODO_EXCLUSION = (
    "NOT metadata.resource_type.id:(publication-datamanagementplan OR publication-deliverable "
    "OR publication-milestone OR publication-proposal)"
)


def _zenodo_hit(index, **metadata):
    meta = {
        "title": f"Record {index}",
        "publication_date": "2023-05-01",
        "description": "<p>Low-cost <b>sensors</b>.</p>",
        "creators": [{"name": "Lewis, Alastair C"}, {"name": "Air Group"}],
        "access_right": "open",
        "resource_type": {"type": "publication", "subtype": "article"},
    }
    meta.update(metadata)
    return {"id": index, "doi": f"10.5281/zenodo.{index}", "metadata": meta}


def test_zenodo_query_always_leaves_out_the_project_paperwork():
    assert search_zenodo.build_zenodo_query("sensor", None, False) == (
        f"((title:sensor OR description:sensor)) AND {ZENODO_EXCLUSION}"
    )
    assert search_zenodo.build_zenodo_query("ignored", "2020-2025", True, "title:x") == (
        f"(title:x) AND {ZENODO_EXCLUSION} AND publication_date:[2020-01-01 TO 2025-12-31] AND access_right:open"
    )


def test_zenodo_maps_a_record():
    record = search_zenodo.zenodo_record(_zenodo_hit(7, doi="10.1/journal"))

    assert record["search_engine_internal_id"] == 7
    assert (record["title"], record["year"], record["doi"]) == ("Record 7", 2023, "10.1/journal")
    assert record["abstract"] == "Low-cost sensors ."
    assert record["authors"] == "Alastair C Lewis; Air Group"
    assert record["pdf_filename"].startswith("2023_Lewis_")
    assert (record["isOpenAccess"], record["entry_source"]) == (True, "Zenodo")


def test_zenodo_falls_back_to_the_zenodo_doi():
    assert search_zenodo.zenodo_record(_zenodo_hit(8))["doi"] == "10.5281/zenodo.8"


def test_zenodo_pages_anonymously_and_waits(session, tmp_path, monkeypatch):
    monkeypatch.setattr("llmexer.base.search_zenodo.time.sleep", Mock())
    session.get.side_effect = [
        _response({"hits": {"total": 30, "hits": [_zenodo_hit(i) for i in range(25)]}}),
        _response({"hits": {"total": 30, "hits": [_zenodo_hit(i) for i in range(25, 30)]}}),
    ]
    raw_path = tmp_path / "raw.json"

    records = search_zenodo.run_zenodo_search("sensor", None, False, 1000, None, str(raw_path))

    assert len(records) == 30
    calls = [call.kwargs["params"] for call in session.get.call_args_list]
    assert [(params["page"], params["size"], params["type"]) for params in calls] == [
        (1, 25, "publication"),
        (2, 25, "publication"),
    ]
    assert ZENODO_EXCLUSION in calls[0]["q"]
    assert "Authorization" not in session.headers
    search_zenodo.time.sleep.assert_called_once_with(2.1)
    assert len(json.loads(raw_path.read_text(encoding="utf-8"))) == 2


def test_zenodo_key_gives_bigger_pages(session, tmp_path, monkeypatch):
    monkeypatch.setattr("llmexer.base.search_zenodo.time.sleep", Mock())
    monkeypatch.setenv("ZENODO_API_KEY", "token")
    session.get.side_effect = [_response({"hits": {"total": 2, "hits": [_zenodo_hit(1), _zenodo_hit(2)]}})]

    records = search_zenodo.run_zenodo_search("sensor", None, False, 1000, None, str(tmp_path / "raw.json"))

    assert len(records) == 2
    assert session.get.call_args.kwargs["params"]["size"] == 100
    assert session.headers["Authorization"] == "Bearer token"


def test_zenodo_stops_at_the_limit(session, tmp_path, monkeypatch):
    monkeypatch.setattr("llmexer.base.search_zenodo.time.sleep", Mock())
    session.get.side_effect = [_response({"hits": {"total": 500, "hits": [_zenodo_hit(i) for i in range(25)]}})]

    records = search_zenodo.run_zenodo_search("sensor", None, False, 1000, 10, str(tmp_path / "raw.json"))

    assert len(records) == 10
    assert session.get.call_count == 1
