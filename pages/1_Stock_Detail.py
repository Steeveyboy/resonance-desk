"""Stock Detail — per-ticker news sentiment and price history.

Reads the Financial Tools warehouse through :mod:`utils.market_data` and
renders four views for a single ticker: current news sentiment, sentiment over
time, price over time, and the underlying articles.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from utils.market_data import (
    DEMO_TICKERS,
    SentimentSnapshot,
    choose_bucket,
    database_status,
    fetch_articles,
    fetch_current_sentiment,
    fetch_price_series,
    fetch_sentiment_series,
    normalize_ticker,
    ticker_exists,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Resonance — Stock Detail",
    page_icon="📈",
    layout="wide",
)

#: Diverging red → grey → green ramp used for every sentiment encoding.
_SENTIMENT_SCALE = alt.Scale(
    domain=[-1, 0, 1], range=["#d62728", "#9aa0a6", "#2ca02c"]
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📈 Stock Detail")
st.caption(
    "News sentiment and price history for a single ticker, straight from the "
    "Financial Tools warehouse."
)
st.divider()

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
db_ok, db_message = database_status()

with st.sidebar:
    st.header("⚙️ Data source")
    if db_ok:
        st.success(db_message, icon="✅")
    else:
        st.error(db_message, icon="🔌")

    st.markdown("**Well-covered tickers**")
    st.caption(
        "The warehouse is partially backfilled. These have both scored news "
        "and price history:"
    )
    st.markdown(" · ".join(f"`{t}`" for t in DEMO_TICKERS))
    st.caption(
        "Sentiment is FinBERT `P(positive) − P(negative)`, ranging from "
        "−1 (negative) to +1 (positive)."
    )

# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------
st.subheader("🔎 Choose a ticker")

col_input, col_example = st.columns([3, 2])
with col_example:
    selected_example = st.selectbox(
        "Or pick a well-covered ticker:", ["Select an example…", *DEMO_TICKERS]
    )
with col_input:
    default_ticker = "" if selected_example.startswith("Select") else selected_example
    ticker_raw = st.text_input(
        "Enter a stock ticker:",
        value=default_ticker,
        placeholder="e.g. NVDA",
    )

ticker = normalize_ticker(ticker_raw)

if not db_ok:
    st.info(
        "Connect a market database to use this page. "
        "See `DATABASE_URL` in `.env.example`.",
        icon="ℹ️",
    )
    st.stop()

if not ticker:
    st.info("Enter a ticker symbol above to load its market intelligence.", icon="👆")
    st.stop()

if not ticker_exists(ticker):
    st.warning(
        f"No articles found for **{ticker}**. The warehouse is still being "
        f"backfilled — try one of: {', '.join(DEMO_TICKERS)}.",
        icon="🔍",
    )
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# 1. Current sentiment
# ---------------------------------------------------------------------------
st.subheader(f"🌡️ Current news sentiment — {ticker}")

snapshot: SentimentSnapshot | None = fetch_current_sentiment(ticker)

if snapshot is None:
    st.warning(
        f"**{ticker}** has articles, but none have been scored for sentiment "
        "yet. Only a small slice of the warehouse has been through the "
        "sentiment transform so far.",
        icon="⏳",
    )
else:
    col_metric, col_gauge = st.columns([1, 3])

    with col_metric:
        st.metric(
            label=f"{snapshot.emoji} Mean sentiment",
            value=f"{snapshot.mean_score:+.3f}",
            help="FinBERT P(positive) − P(negative), from −1 to +1.",
        )
        st.caption(f"**{snapshot.label}** · {snapshot.article_count} articles")

    with col_gauge:
        gauge_df = pd.DataFrame([{"score": snapshot.mean_score, "row": ticker}])

        track = (
            alt.Chart(pd.DataFrame({"lo": [-1.0], "hi": [1.0], "row": [ticker]}))
            .mark_bar(height=34, color="#e9ecef", cornerRadius=4)
            .encode(
                x=alt.X("lo:Q", scale=alt.Scale(domain=[-1, 1]), title=None),
                x2="hi:Q",
                y=alt.Y("row:N", title=None, axis=None),
            )
        )
        bar = (
            alt.Chart(gauge_df)
            .mark_bar(height=34, cornerRadius=4)
            .encode(
                x=alt.X("baseline:Q", scale=alt.Scale(domain=[-1, 1])),
                x2="score:Q",
                y=alt.Y("row:N", title=None, axis=None),
                color=alt.Color("score:Q", scale=_SENTIMENT_SCALE, legend=None),
                tooltip=[alt.Tooltip("score:Q", format="+.3f", title="Sentiment")],
            )
            .transform_calculate(baseline="0")
        )
        midline = (
            alt.Chart(pd.DataFrame({"zero": [0]}))
            .mark_rule(color="#495057", strokeDash=[4, 3])
            .encode(x="zero:Q")
        )

        st.altair_chart(
            (track + bar + midline).properties(height=90).configure_view(stroke=None),
            width="stretch",
        )

    st.caption(
        f"📅 Window: **{snapshot.window_start:%d %b %Y} – "
        f"{snapshot.window_end:%d %b %Y}** "
        f"({snapshot.article_count} scored articles). This is the most recent "
        "period for which scored articles exist — not necessarily today."
    )

st.divider()

# ---------------------------------------------------------------------------
# 2. Sentiment over time
# ---------------------------------------------------------------------------
st.subheader("📉 Sentiment over time")

daily = fetch_sentiment_series(ticker, bucket="day")
sentiment_df = (
    daily
    if daily.empty
    else fetch_sentiment_series(ticker, bucket=choose_bucket(daily))
)

if sentiment_df.empty:
    st.warning(f"No scored articles for **{ticker}**, so there is no sentiment history.", icon="⏳")
else:
    # Smooth the noise without hiding it; window shrinks on short series.
    window = max(2, min(6, len(sentiment_df) // 4))
    sentiment_df = sentiment_df.assign(
        rolling_score=sentiment_df["mean_score"].rolling(window, min_periods=1).mean()
    )

    zero_rule = (
        alt.Chart(pd.DataFrame({"zero": [0]}))
        .mark_rule(color="#adb5bd")
        .encode(y="zero:Q")
    )
    points = (
        alt.Chart(sentiment_df)
        .mark_circle()
        .encode(
            x=alt.X("bucket:T", title=None),
            y=alt.Y(
                "mean_score:Q",
                scale=alt.Scale(domain=[-1, 1]),
                title="Mean sentiment",
            ),
            # Article count reads as confidence — thin buckets stay faint.
            size=alt.Size("article_count:Q", legend=None, scale=alt.Scale(range=[15, 220])),
            color=alt.Color("mean_score:Q", scale=_SENTIMENT_SCALE, legend=None),
            tooltip=[
                alt.Tooltip("bucket:T", title="Period"),
                alt.Tooltip("mean_score:Q", format="+.3f", title="Mean sentiment"),
                alt.Tooltip("article_count:Q", title="Articles"),
            ],
        )
    )
    trend = (
        alt.Chart(sentiment_df)
        .mark_line(color="#1f77b4", strokeWidth=2.5)
        .encode(x="bucket:T", y="rolling_score:Q")
    )

    st.altair_chart(
        (zero_rule + points + trend).properties(height=280).interactive(),
        width="stretch",
    )
    st.caption(
        f"Bubble size is the number of articles in each bucket; the blue line "
        f"is a {window}-period rolling mean. "
        f"Covers {sentiment_df['bucket'].min():%b %Y} – "
        f"{sentiment_df['bucket'].max():%b %Y} "
        f"({int(sentiment_df['article_count'].sum())} scored articles)."
    )

st.divider()

# ---------------------------------------------------------------------------
# 3. Price over time
# ---------------------------------------------------------------------------
st.subheader("💵 Price over time")

price_df = fetch_price_series(ticker)

if price_df.empty:
    st.warning(
        f"No price history for **{ticker}**. Daily pricing currently covers "
        "only about 100 symbols in the warehouse.",
        icon="📭",
    )
else:
    price_chart = (
        alt.Chart(price_df)
        .mark_line(color="#1f77b4", strokeWidth=1.6)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("close:Q", scale=alt.Scale(zero=False), title="Close"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("close:Q", format=".2f", title="Close"),
                alt.Tooltip("volume:Q", format=",", title="Volume"),
            ],
        )
        .properties(height=280)
        .interactive()
    )
    st.altair_chart(price_chart, width="stretch")

    caption = (
        f"Daily close, {price_df['date'].min():%b %Y} – "
        f"{price_df['date'].max():%b %Y}."
    )
    # The two series rarely line up: pricing starts in 2015, while most scored
    # news predates 2020. Say so rather than clipping one to the other.
    if not sentiment_df.empty:
        caption += (
            " Note the sentiment chart above covers a different period — the "
            "two are shown at their full extents rather than clipped to overlap."
        )
    st.caption(caption)

st.divider()

# ---------------------------------------------------------------------------
# 4. Relevant articles
# ---------------------------------------------------------------------------
st.subheader("📰 Relevant articles")

col_count, col_filter = st.columns([1, 2])
with col_count:
    article_limit = st.slider("How many?", min_value=5, max_value=50, value=25, step=5)
with col_filter:
    scored_only = st.checkbox("Only articles with a sentiment score", value=False)

articles_df = fetch_articles(ticker, limit=article_limit, scored_only=scored_only)

if articles_df.empty:
    st.info("No articles matched those filters.", icon="🗒️")
else:
    st.dataframe(
        articles_df,
        width="stretch",
        hide_index=True,
        column_order=("published_at", "title", "publisher", "sentiment_score", "url"),
        column_config={
            "published_at": st.column_config.DatetimeColumn(
                "Published", format="YYYY-MM-DD", width="small"
            ),
            "title": st.column_config.TextColumn("Headline", width="large"),
            "publisher": st.column_config.TextColumn("Publisher", width="small"),
            "sentiment_score": st.column_config.NumberColumn(
                "Sentiment", format="%+.3f", width="small"
            ),
            "url": st.column_config.LinkColumn(
                "Link", display_text="Open ↗", width="small"
            ),
        },
    )
    st.caption(
        "Headlines are de-duplicated — the historical dataset stores "
        "near-identical stories under several URLs."
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption("Resonance v0.1 — skeleton app — not financial advice.")
