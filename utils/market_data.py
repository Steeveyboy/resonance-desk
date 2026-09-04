"""Read-only access to the Financial Tools market warehouse.

The warehouse is a PostgreSQL database maintained by a separate repo
(``Financial_Tools``) holding news articles, per-article sentiment scores, and
daily OHLCV pricing. That repo is *not* an installable package, so this module
is the app's own, self-contained SQL boundary — nothing from ``findata`` is
imported here.

Connection details come from ``DATABASE_URL`` in the local ``.env``. Mirroring
:mod:`utils.llm`, this module never raises on a missing or unreachable
database: callers get an empty result and :func:`database_status` explains why,
so the UI can show a message instead of a traceback.

All queries are read-only, parameterised, and bounded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()

#: Placeholder shipped in ``.env.example``; treated the same as "not set".
_PLACEHOLDER_URL = "postgresql://user:password@localhost:5432/corporate_db"

#: Upper bound on any single query, in milliseconds.
_STATEMENT_TIMEOUT_MS = 15_000

#: Tickers verified to have *both* scored news and price history — the app is a
#: demo over a partially backfilled warehouse, so these give a populated view.
DEMO_TICKERS = ["ADBE", "NVDA", "AAPL", "AVGO", "ADSK", "ADI"]

#: ``date_trunc`` units this module will accept. Never take this from user input.
_ALLOWED_BUCKETS = ("day", "week", "month")

#: Characters permitted in a ticker symbol after normalisation.
_TICKER_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")


@dataclass
class SentimentSnapshot:
    """Aggregate news sentiment over the most recent *available* scored window.

    The window is anchored on the ticker's newest scored article rather than on
    today, because sentiment scoring lags ingestion by a wide margin. Callers
    should surface :attr:`window_start` / :attr:`window_end` so the reader can
    see how current the figure actually is.

    Attributes:
        mean_score:    Mean ``sentiment_score`` in ``[-1.0, 1.0]``.
        article_count: Number of scored articles in the window.
        window_start:  Publication date of the oldest article in the window.
        window_end:    Publication date of the newest article in the window.
    """

    mean_score: float
    article_count: int
    window_start: date
    window_end: date

    @property
    def label(self) -> str:
        """Return a human-readable description of the sentiment direction."""
        if self.mean_score >= 0.25:
            return "Positive"
        if self.mean_score >= 0.05:
            return "Leaning positive"
        if self.mean_score <= -0.25:
            return "Negative"
        if self.mean_score <= -0.05:
            return "Leaning negative"
        return "Neutral"

    @property
    def emoji(self) -> str:
        """Return an emoji representing the sentiment direction."""
        if self.mean_score >= 0.25:
            return "🟢"
        if self.mean_score >= 0.05:
            return "🌱"
        if self.mean_score <= -0.25:
            return "🔴"
        if self.mean_score <= -0.05:
            return "🍂"
        return "⚪"


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
def normalize_ticker(raw: str) -> str:
    """Normalize user input into a ticker symbol.

    Upper-cases, strips surrounding whitespace and a leading ``$``, drops any
    character that cannot appear in a symbol, and truncates to the 10-character
    width of ``article_tickers.ticker``.

    Args:
        raw: Text typed by the user.

    Returns:
        A normalized ticker, or ``""`` if nothing usable remains.
    """
    cleaned = (raw or "").strip().lstrip("$").upper()
    return "".join(c for c in cleaned if c in _TICKER_CHARS)[:10]


@st.cache_resource(show_spinner=False)
def get_engine() -> Optional[Engine]:
    """Return a cached SQLAlchemy engine, or ``None`` if unconfigured.

    ``None`` is returned when ``DATABASE_URL`` is absent or still holds the
    ``.env.example`` placeholder, which lets the app run without a warehouse.
    """
    url = os.getenv("DATABASE_URL", "").strip()
    if not url or url == _PLACEHOLDER_URL:
        return None
    try:
        return create_engine(
            url,
            pool_pre_ping=True,
            future=True,
            # Bound worst-case latency: a slow ticker becomes a caught
            # error rather than a hung page. The read-only default makes it
            # impossible for this app to write to a warehouse it does not own.
            connect_args={
                "options": (
                    f"-c statement_timeout={_STATEMENT_TIMEOUT_MS} "
                    "-c default_transaction_read_only=on"
                )
            },
        )
    except Exception:  # noqa: BLE001 — never surface a URL-bearing traceback
        return None


def database_status() -> tuple[bool, str]:
    """Report whether the warehouse is reachable.

    Returns:
        ``(ok, message)`` — *message* is safe to render in the UI and never
        contains the connection URL or credentials.
    """
    engine = get_engine()
    if engine is None:
        return False, (
            "No market database configured. Set `DATABASE_URL` in your `.env` "
            "to connect to the Financial Tools warehouse."
        )
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "Connected to the market warehouse."
    except Exception:  # noqa: BLE001
        return False, (
            "Could not reach the market database. Check that PostgreSQL is "
            "running and that `DATABASE_URL` is correct."
        )


def _query(sql: str, params: dict) -> pd.DataFrame:
    """Run a read-only query and return the rows as a DataFrame.

    Returns an empty DataFrame if the warehouse is unconfigured or the query
    fails, so callers never have to guard against exceptions.
    """
    engine = get_engine()
    if engine is None:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            return pd.read_sql_query(text(sql), conn, params=params)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------
_CURRENT_SENTIMENT_SQL = """
    WITH latest AS (
        SELECT max(a.published_at) AS m
        FROM article_tickers t
        JOIN articles a ON a.id = t.article_id
        WHERE t.ticker = :ticker
          AND a.sentiment_score IS NOT NULL
          AND a.published_at IS NOT NULL
    )
    SELECT avg(a.sentiment_score)      AS mean_score,
           count(*)                    AS article_count,
           min(a.published_at)::date   AS window_start,
           max(a.published_at)::date   AS window_end
    FROM article_tickers t
    JOIN articles a ON a.id = t.article_id,
         latest
    WHERE t.ticker = :ticker
      AND a.sentiment_score IS NOT NULL
      AND a.published_at IS NOT NULL
      AND a.published_at >= latest.m - make_interval(days => :window_days)
"""


@st.cache_data(ttl=600, show_spinner=False)
def fetch_current_sentiment(
    ticker: str, window_days: int = 30
) -> Optional[SentimentSnapshot]:
    """Return aggregate sentiment over the ticker's most recent scored window.

    Args:
        ticker: Normalized ticker symbol.
        window_days: Width of the window, counted back from the newest scored
            article rather than from today.

    Returns:
        A :class:`SentimentSnapshot`, or ``None`` when the ticker has no scored
        articles at all.
    """
    df = _query(
        _CURRENT_SENTIMENT_SQL, {"ticker": ticker, "window_days": window_days}
    )
    if df.empty or df.at[0, "article_count"] == 0:
        return None

    row = df.iloc[0]
    return SentimentSnapshot(
        mean_score=float(row["mean_score"]),
        article_count=int(row["article_count"]),
        window_start=row["window_start"],
        window_end=row["window_end"],
    )


_SENTIMENT_SERIES_SQL = """
    SELECT date_trunc(:bucket, a.published_at)::date AS bucket,
           avg(a.sentiment_score)                    AS mean_score,
           count(*)                                  AS article_count
    FROM article_tickers t
    JOIN articles a ON a.id = t.article_id
    WHERE t.ticker = :ticker
      AND a.sentiment_score IS NOT NULL
      AND a.published_at IS NOT NULL
    GROUP BY 1
    ORDER BY 1
"""


@st.cache_data(ttl=600, show_spinner=False)
def fetch_sentiment_series(ticker: str, bucket: str = "month") -> pd.DataFrame:
    """Return mean sentiment per time bucket for *ticker*.

    Args:
        ticker: Normalized ticker symbol.
        bucket: One of ``day``, ``week``, or ``month``. Anything else falls
            back to ``month`` — this value reaches ``date_trunc``, so it is
            whitelisted rather than passed through from user input.

    Returns:
        Columns ``bucket`` (datetime), ``mean_score``, ``article_count``.
        Empty if the ticker has no scored articles.
    """
    if bucket not in _ALLOWED_BUCKETS:
        bucket = "month"

    df = _query(_SENTIMENT_SERIES_SQL, {"ticker": ticker, "bucket": bucket})
    if df.empty:
        return df

    df["bucket"] = pd.to_datetime(df["bucket"])
    df["mean_score"] = df["mean_score"].astype(float)
    df["article_count"] = df["article_count"].astype(int)
    return df


def choose_bucket(df: pd.DataFrame) -> str:
    """Pick a sensible ``date_trunc`` unit for the span covered by *df*.

    Sentiment history can run from a few weeks to more than a decade, so the
    bucket adapts to keep the series readable.

    Args:
        df: A frame with a ``bucket`` datetime column (typically a first pass
            fetched at daily resolution).

    Returns:
        ``"day"``, ``"week"``, or ``"month"``.
    """
    if df.empty:
        return "month"
    span_days = (df["bucket"].max() - df["bucket"].min()).days
    if span_days <= 90:
        return "day"
    if span_days <= 365 * 3:
        return "week"
    return "month"


_PRICE_SERIES_SQL = """
    SELECT date, open, high, low, close, volume
    FROM daily_ohlcv
    WHERE ticker = :ticker
    ORDER BY date
"""


@st.cache_data(ttl=600, show_spinner=False)
def fetch_price_series(ticker: str) -> pd.DataFrame:
    """Return the daily OHLCV history for *ticker*.

    Args:
        ticker: Normalized ticker symbol.

    Returns:
        Columns ``date`` (datetime), ``open``, ``high``, ``low``, ``close``
        (floats) and ``volume``. Empty if the warehouse holds no pricing for
        this ticker — coverage is currently about 100 symbols.
    """
    df = _query(_PRICE_SERIES_SQL, {"ticker": ticker})
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    for column in ("open", "high", "low", "close"):
        # Stored as NUMERIC, which arrives as Decimal and cannot be plotted.
        df[column] = df[column].astype(float)
    return df


_ARTICLES_SQL = """
    SELECT DISTINCT ON (a.title)
           a.title, a.url, a.publisher, a.published_at, a.sentiment_score
    FROM article_tickers t
    JOIN articles a ON a.id = t.article_id
    WHERE t.ticker = :ticker
      AND a.published_at IS NOT NULL
      AND (NOT :scored_only OR a.sentiment_score IS NOT NULL)
    ORDER BY a.title, a.published_at DESC
    LIMIT :limit
"""


@st.cache_data(ttl=600, show_spinner=False)
def fetch_articles(
    ticker: str, limit: int = 25, scored_only: bool = False
) -> pd.DataFrame:
    """Return recent articles mentioning *ticker*.

    The historical dataset stores near-identical headlines under several URLs,
    so rows are de-duplicated by title. PostgreSQL requires the ``DISTINCT ON``
    column to lead ``ORDER BY``, hence the re-sort by date here.

    Args:
        ticker: Normalized ticker symbol.
        limit: Maximum number of articles to return.
        scored_only: Restrict to articles that carry a sentiment score.

    Returns:
        Columns ``title``, ``url``, ``publisher``, ``published_at``,
        ``sentiment_score``, newest first.
    """
    df = _query(
        _ARTICLES_SQL,
        {"ticker": ticker, "limit": limit, "scored_only": scored_only},
    )
    if df.empty:
        return df

    df["published_at"] = pd.to_datetime(df["published_at"])
    # ~80% of historical rows carry no publisher.
    df["publisher"] = df["publisher"].fillna("—")
    return df.sort_values("published_at", ascending=False).reset_index(drop=True)


_HAS_ARTICLES_SQL = """
    SELECT 1 FROM article_tickers WHERE ticker = :ticker LIMIT 1
"""


@st.cache_data(ttl=600, show_spinner=False)
def ticker_exists(ticker: str) -> bool:
    """Return whether the warehouse holds any article linked to *ticker*."""
    return not _query(_HAS_ARTICLES_SQL, {"ticker": ticker}).empty
