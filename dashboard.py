import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import numpy as np
import os

# Page config
st.set_page_config(page_title="Financial Dashboard", page_icon="💹", layout="wide")

# Load data
@st.cache_data
def load_data():
    file_path = "data/combined_trades.csv"
    if not os.path.exists(file_path):
        st.error(f"❌ File not found: {file_path}")
        st.stop()
    return pd.read_csv(file_path)

# Sidebar
st.sidebar.title("⚙️ Controls")

df = load_data()

# Convert Date column if it exists
if hasattr(df, "columns") and "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Rename PnL column to pnl if it exists
if hasattr(df, "columns") and "PnL" in df.columns:
    df = df.rename(columns={"PnL": "pnl"})

# Filters
if hasattr(df, "columns") and "Date" in df.columns and not df["Date"].isna().all():
    min_date = df["Date"].min()
    max_date = df["Date"].max()
    min_date = min_date.date() if isinstance(min_date, pd.Timestamp) else datetime.date.today()
    max_date = max_date.date() if isinstance(max_date, pd.Timestamp) else datetime.date.today()

    date_range = st.sidebar.date_input("Date Range", (min_date, max_date))
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = date_range
        df = df[(df["Date"] >= pd.to_datetime(start)) & (df["Date"] <= pd.to_datetime(end))]

# Ticker filter
if hasattr(df, "columns") and "Ticker" in df.columns:
    tickers = st.sidebar.multiselect(
        "Select Ticker(s)",
        options=sorted(pd.Series(df["Ticker"]).dropna().unique())
    )
    if tickers:
        df = df[pd.Series(df["Ticker"]).isin(tickers)]

# Strategy filter
strategy_col = None
if hasattr(df, "columns"):
    strategy_col = next((col for col in df.columns if col.lower().replace(" ", "_") == "opening_strategy"), None)
if strategy_col:
    strategies = sorted(pd.Series(df[strategy_col]).dropna().unique())
    selected_strategies = st.sidebar.multiselect("Select Opening Strategy", options=strategies)
    if selected_strategies:
        df = df[pd.Series(df[strategy_col]).isin(selected_strategies)]

# Main dashboard
st.title("💹 Financial Dashboard")
st.markdown("A modern, interactive trading performance dashboard.")

# --- Metrics Functions ---
def compute_sharpe_ratio(pnl_series, risk_free_rate=0.0, period_per_year=252):
    if len(pnl_series) < 2:
        return np.nan
    mean_return = pnl_series.mean()
    std_return = pnl_series.std(ddof=1)
    if std_return == 0:
        return np.nan
    return (mean_return - risk_free_rate/period_per_year) / std_return * np.sqrt(period_per_year)

def compute_max_drawdown(equity_curve):
    if len(equity_curve) < 2:
        return 0.0
    roll_max = equity_curve.cummax()
    return (equity_curve - roll_max).min()

def compute_kelly_fraction(pnl_series):
    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series < 0]
    if len(wins) == 0 or len(losses) == 0:
        return np.nan
    p = len(wins) / len(pnl_series)
    q = 1 - p
    avg_win = wins.mean()
    avg_loss = -losses.mean()
    if avg_loss == 0:
        return np.nan
    b = avg_win / avg_loss
    return (p * b - q) / b

# Prepare daily PnL for metrics
if (
    hasattr(df, "columns")
    and "Date" in df.columns
    and "pnl" in df.columns
    and hasattr(df, "empty")
    and not df.empty
):
    daily_pnl = df.groupby("Date")["pnl"].sum().sort_index()
    cumulative_pnl = daily_pnl.cumsum()
    sharpe = compute_sharpe_ratio(daily_pnl)
    max_dd = compute_max_drawdown(cumulative_pnl)
    kelly = compute_kelly_fraction(daily_pnl)
else:
    sharpe = max_dd = kelly = np.nan

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total P/L", f"${df['pnl'].sum():,.2f}" if hasattr(df, "columns") and "pnl" in df.columns else "N/A")
col2.metric("Win Rate", f"{(df['pnl'] > 0).mean()*100:.2f}%" if hasattr(df, "columns") and "pnl" in df.columns else "N/A")
col3.metric("Trades", len(df) if hasattr(df, "__len__") else "N/A")
col4.metric("Avg Trade", f"${df['pnl'].mean():,.2f}" if hasattr(df, "columns") and "pnl" in df.columns else "N/A")

# Advanced Metrics Row
col5, col6, col7 = st.columns(3)
col5.metric("Sharpe Ratio", f"{sharpe:.2f}" if pd.notnull(sharpe) else "N/A")
col6.metric("Max Drawdown", f"${max_dd:,.2f}" if pd.notnull(max_dd) else "N/A")
col7.metric("Optimal Kelly Fraction", f"{kelly:.2%}" if pd.notnull(kelly) else "N/A")

# ROI metric
if (
    hasattr(df, "columns")
    and "pnl" in df.columns
    and "cost_basis" in df.columns
    and pd.Series(df["cost_basis"]).abs().sum() > 0
):
    roi = df["pnl"].sum() / pd.Series(df["cost_basis"]).abs().sum() * 100
    st.metric("ROI", f"{roi:.2f}%")

# Charts
if hasattr(df, "columns") and "Date" in df.columns and hasattr(df, "empty") and not df.empty:
    pnl_time_df = df.groupby("Date", as_index=False)["pnl"].sum().sort_values("Date")
    pnl_time_df["cumulative_pnl"] = pnl_time_df["pnl"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pnl_time_df["Date"], y=pnl_time_df["pnl"],
        name="Daily PnL", marker_color="#636EFA", opacity=0.7,
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Daily PnL: $%{y:,.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=pnl_time_df["Date"], y=pnl_time_df["cumulative_pnl"],
        name="Cumulative PnL", mode="lines+markers",
        line=dict(color="#00CC96", width=3), marker=dict(size=6),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Cumulative PnL: $%{y:,.2f}<extra></extra>",
        yaxis="y2"
    ))
    fig.update_layout(
        title="Interactive PnL Chart Over Time",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis=dict(title="Daily PnL", showgrid=False),
        yaxis2=dict(title="Cumulative PnL", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        bargap=0.2,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

# P/L by Ticker
if hasattr(df, "columns") and "Ticker" in df.columns and hasattr(df, "empty") and not df.empty:
    pnl_ticker_df = df.groupby("Ticker", as_index=False)["pnl"].sum()
    fig_pnl_ticker = px.bar(
        pnl_ticker_df, x="Ticker", y="pnl", title="P/L by Ticker",
        template="plotly_dark", color="pnl", color_continuous_scale="Tealgrn"
    )
    st.plotly_chart(fig_pnl_ticker, use_container_width=True)

# Trades Table
st.subheader("📋 Trades Table")
def pnl_color(val):
    if pd.isnull(val):
        return ""
    color = "green" if val > 0 else "red" if val < 0 else "black"
    return f"color: {color}; font-weight: bold;"

if hasattr(df, "empty") and not df.empty:
    if hasattr(df, "columns") and "pnl" in df.columns:
        # Use Styler.map instead of applymap (applymap is deprecated)
        st.dataframe(df.style.map(pnl_color, subset=["pnl"]))
    else:
        st.dataframe(df)
else:
    st.info("No trades to display for the selected filters.")
