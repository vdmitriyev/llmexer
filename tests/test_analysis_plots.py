"""Tests for the copied `plots` analysis module."""

import matplotlib

matplotlib.use("Agg")  # no display in CI; must precede the pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from llmexer.base.analysis.plots import (  # noqa: E402
    plot_failures_by_model,
    plot_flattened_values,
    plot_response_time_by_model,
    plot_responses_by_model,
    plot_results_by_search_engine,
    plot_tokens_by_model,
    setup_style,
)
from llmexer.base.analysis.stats import value_counts_summary  # noqa: E402

_EXPERIMENT_PLOTS = (
    plot_responses_by_model,
    plot_response_time_by_model,
    plot_tokens_by_model,
    plot_failures_by_model,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture()
def frame():
    return pd.DataFrame(
        [
            dict(
                provider_name="ollama",
                model_name="m1",
                status="success",
                state="finished",
                elapsed_seconds=2.0,
                total_tokens=100,
                usage_tokens=100,
            ),
            dict(
                provider_name="ollama",
                model_name="m1",
                status="success",
                state="maxtokenreached",
                elapsed_seconds=4.0,
                total_tokens=50,
                usage_tokens=50,
            ),
            dict(
                provider_name="litellm",
                model_name="m1",
                status="Error: boom",
                state="error",
                elapsed_seconds=None,
                total_tokens=None,
                usage_tokens=None,
            ),
            dict(
                provider_name="litellm",
                model_name="m2",
                status=None,
                state=None,
                elapsed_seconds=None,
                total_tokens=None,
                usage_tokens=None,
            ),
        ]
    )


@pytest.fixture()
def searches():
    return pd.DataFrame({"entry_source": ["Semantic Scholar", "Semantic Scholar", "OpenAlex"]})


@pytest.mark.parametrize("plot", _EXPERIMENT_PLOTS)
def test_plot_returns_a_titled_axes(plot, frame):
    ax = plot(frame)

    assert isinstance(ax, Axes)
    assert ax.get_title()


@pytest.mark.parametrize("plot", _EXPERIMENT_PLOTS)
def test_plot_draws_into_a_supplied_axes(plot, frame):
    _, ax = plt.subplots()

    assert plot(frame, ax=ax) is ax


@pytest.mark.parametrize("plot", _EXPERIMENT_PLOTS)
def test_plot_on_an_empty_frame_degrades_instead_of_raising(plot):
    """A half-finished project must still run the notebook top to bottom."""

    ax = plot(pd.DataFrame())

    assert isinstance(ax, Axes)
    assert any("No data" in text.get_text() for text in ax.texts)


@pytest.mark.parametrize("plot", _EXPERIMENT_PLOTS)
def test_plot_without_a_model_column_raises_key_error(plot):
    with pytest.raises(KeyError):
        plot(pd.DataFrame({"something": [1]}))


@pytest.mark.parametrize("plot", _EXPERIMENT_PLOTS)
def test_plot_writes_no_file(plot, frame, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plot(frame)

    assert not list(tmp_path.iterdir())


def test_plot_results_by_search_engine(searches):
    ax = plot_results_by_search_engine(searches)

    assert isinstance(ax, Axes)
    assert ax.get_ylabel() == "search engine"


def test_plot_results_by_search_engine_without_the_column():
    with pytest.raises(KeyError):
        plot_results_by_search_engine(pd.DataFrame({"x": [1]}))


def test_plot_results_by_search_engine_on_an_empty_frame():
    ax = plot_results_by_search_engine(pd.DataFrame())

    assert any("No data" in text.get_text() for text in ax.texts)


def test_response_time_plot_without_finished_calls(frame):
    ax = plot_response_time_by_model(frame.assign(status=None))

    assert any("No finished calls" in text.get_text() for text in ax.texts)


def test_importing_plots_does_not_change_rcparams():
    """Style belongs in `setup_style()`, called explicitly by the notebook."""

    before = matplotlib.rcParams["axes.grid"]
    import importlib

    import llmexer.base.analysis.plots as module

    importlib.reload(module)

    assert matplotlib.rcParams["axes.grid"] == before


def test_setup_style_applies_the_theme():
    setup_style()

    assert matplotlib.rcParams["axes.grid"] is True


# ---------------------------------------------------------------------------
# plot_flattened_values — the parsed answers
# ---------------------------------------------------------------------------


@pytest.fixture()
def answers():
    return pd.DataFrame(
        {
            "verdict": ["yes", "no", "yes", None],
            "rating": [1, 2, 3, 3],
            "topic": ["ml", "nlp", "ml", "cv"],
            "reason": ["a long free-text justification that reads as prose " * 2] * 4,
        }
    )


def test_plot_flattened_values_draws_one_panel_per_plottable_column(answers):
    """Free text has no small set of values, so it gets no panel."""

    figure = plot_flattened_values(answers, value_counts_summary(answers))
    titles = [ax.get_title() for ax in figure.axes if ax.get_visible()]

    assert titles == ["verdict", "rating", "topic"]


def test_plot_flattened_values_returns_a_figure(answers):
    """This chart is inherently multi-panel, so it returns the Figure."""

    assert isinstance(plot_flattened_values(answers, value_counts_summary(answers)), Figure)


def test_plot_flattened_values_orders_a_rating_by_its_own_scale(answers):
    figure = plot_flattened_values(answers, value_counts_summary(answers))
    rating = next(ax for ax in figure.axes if ax.get_title() == "rating")

    assert [label.get_text() for label in rating.get_xticklabels()] == ["1", "2", "3"]


def test_plot_flattened_values_caps_the_number_of_panels(answers):
    figure = plot_flattened_values(answers, value_counts_summary(answers), max_columns=2)

    assert len([ax for ax in figure.axes if ax.get_visible()]) == 2


def test_plot_flattened_values_with_nothing_worth_charting():
    df = pd.DataFrame({"reason": ["a long free-text justification " * 5] * 3})
    figure = plot_flattened_values(df, value_counts_summary(df))

    assert any("No yes/no or countable" in text.get_text() for text in figure.axes[0].texts)


def test_plot_flattened_values_with_an_empty_summary(answers):
    figure = plot_flattened_values(answers, pd.DataFrame())

    assert any("No answer columns" in text.get_text() for text in figure.axes[0].texts)


def test_plot_flattened_values_writes_no_file(answers, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plot_flattened_values(answers, value_counts_summary(answers))

    assert not list(tmp_path.iterdir())
