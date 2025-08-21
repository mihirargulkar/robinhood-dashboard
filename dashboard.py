import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import timedelta
from PIL import Image

# --- Configuration & UI Setup ---
im = Image.open("MIMLOGONEW copy.png")
st.set_page_config(
    page_title="Mills Investment Dashboard",
    page_icon=im,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data():
    df = pd.read_csv('data/combined_trades.csv')

    # Data type conversions
    if 'expiration_date' in df.columns:
        df['expiration_date'] = pd.to_datetime(df['expiration_date'], errors='coerce')
    if 'order_created_at' in df.columns:
        df['order_created_at'] = pd.to_datetime(df['order_created_at'], errors='coerce', utc=True)
    if 'strike_price' in df.columns:
        df['strike_price'] = pd.to_numeric(df['strike_price'], errors='coerce')
        df.dropna(subset=['strike_price'], inplace=True)
    if 'expiration_date' in df.columns:
        df['entry_date'] = df['expiration_date'] - pd.to_timedelta(np.random.randint(5, 30, size=len(df)), unit='D')
    if 'order_created_at' in df.columns:
        df.sort_values('order_created_at', inplace=True)
    return df

# --- Benchmark Data Loading ---
@st.cache_data
def load_benchmark_data(benchmark_file, price_col):
    """
    Loads and preprocesses benchmark data from a given CSV file.
    """
    for path in [benchmark_file, os.path.join('data', benchmark_file)]:
        if os.path.exists(path):
            try:
                bench_df = pd.read_csv(path)
                if 'Date' in bench_df.columns and price_col in bench_df.columns:
                    bench_df['Date'] = pd.to_datetime(bench_df['Date']).dt.tz_localize('UTC')
                    bench_df.rename(columns={'Date': 'date', price_col: 'price'}, inplace=True)
                    bench_df.sort_values('date', inplace=True)
                    return bench_df
                else:
                    st.error(f"Error: The '{benchmark_file}' file must contain 'Date' and '{price_col}' columns.")
                    return pd.DataFrame()
            except Exception as e:
                st.error(f"Error reading '{path}': {e}")
                return pd.DataFrame()
    st.error(f"Error: The file '{benchmark_file}' was not found. Please ensure it is in the app directory or in a 'data/' subdirectory.")
    return pd.DataFrame()

# --- Benchmark Options ---
BENCHMARKS = [
    {"label": "S&P 500 (SPY)", "file": "spy_data.csv", "price_col": "SPY", "display": "S&P 500"},
    {"label": "NASDAQ 100 (QQQ)", "file": "qqq_data.csv", "price_col": "QQQ", "display": "NASDAQ 100"},
    {"label": "Russell 2000 (IWM)", "file": "iwm_data.csv", "price_col": "IWM", "display": "Russell 2000"},
    {"label": "Dow Jones (DIA)", "file": "dia_data.csv", "price_col": "DIA", "display": "Dow Jones"},
    # Add more benchmarks here as needed
]

# --- Main App Logic ---

# Set a consistent dark theme for the app
st.markdown(
    """
    <style>
    .stApp {background-color: #0e1117; color: #fafafa;}
    .stMarkdown {color: #fafafa;}
    </style>
    """,
    unsafe_allow_html=True
)
plotly_theme = "plotly_dark"

# Load the data
df = load_data()

# Sidebar for filters
with st.sidebar:
    st.header("Dashboard Settings")
    with st.expander("Filter Options", expanded=True):
        if st.button("Reset Filters"):
            st.rerun()
        initial_balance = st.number_input(
            "Initial Account Balance", min_value=0.0, value=5500.0, step=100.0, format="%.2f"
        )
        all_strategies = ['All'] + list(df['opening_strategy'].unique()) if not df.empty and 'opening_strategy' in df.columns else ['All']
        selected_strategies = st.multiselect(
            "Select Opening Strategy(s)", options=all_strategies, default='All'
        )
        all_symbols = ['All'] + sorted(list(df['symbol'].unique())) if not df.empty and 'symbol' in df.columns else ['All']
        selected_symbols = st.multiselect(
            "Select Trading Symbol(s)", options=all_symbols, default='All'
        )
        # Date filter using 'order_created_at'
        if not df.empty and 'order_created_at' in df.columns and not df['order_created_at'].isnull().all():
            min_date_trade = df['order_created_at'].min().date()
            max_date_trade = df['order_created_at'].max().date()
            trade_date_range = st.date_input(
                "Select Trade Creation Date Range",
                value=(min_date_trade, max_date_trade),
                min_value=min_date_trade,
                max_value=max_date_trade
            )
            if isinstance(trade_date_range, (tuple, list)) and len(trade_date_range) == 2:
                start_date = pd.to_datetime(trade_date_range[0]).tz_localize('UTC')
                end_date = pd.to_datetime(trade_date_range[1]).tz_localize('UTC') + timedelta(days=1) - timedelta(microseconds=1)
            else:
                start_date = end_date = pd.to_datetime(min_date_trade).tz_localize('UTC')
        else:
            start_date = end_date = None

    # --- Benchmark Selection ---
    st.header("Benchmark Settings")
    benchmark_labels = [b["label"] for b in BENCHMARKS]
    default_benchmark_idx = 0
    selected_benchmark_label = st.selectbox(
        "Select Benchmark for Comparison",
        options=benchmark_labels,
        index=default_benchmark_idx
    )
    selected_benchmark = next(b for b in BENCHMARKS if b["label"] == selected_benchmark_label)
    benchmark_file = selected_benchmark["file"]
    benchmark_price_col = selected_benchmark["price_col"]
    benchmark_display_name = selected_benchmark["display"]

# Load selected benchmark data
benchmark_df = load_benchmark_data(benchmark_file, benchmark_price_col)

# Main title and introduction
st.title("Mills Investment Dashboard")
st.markdown("A comprehensive and interactive overview of our portfolio performance.")
st.markdown("---")

# Only proceed if the data was loaded successfully
if not df.empty:
    filtered_df = df.copy()
    if 'All' not in selected_strategies and 'opening_strategy' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['opening_strategy'].isin(selected_strategies)]
    if 'All' not in selected_symbols and 'symbol' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['symbol'].isin(selected_symbols)]
    if start_date is not None and end_date is not None and 'order_created_at' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['order_created_at'] >= start_date) &
            (filtered_df['order_created_at'] <= end_date)
        ]

    # --- Portfolio Statistics Calculations ---
    pnl_col = 'pnl'
    collateral_col = 'collateral'
    total_pnl = filtered_df[pnl_col].sum() if pnl_col in filtered_df.columns else 0
    num_trades = len(filtered_df)
    win_rate = (filtered_df[filtered_df[pnl_col] > 0].shape[0] / num_trades * 100) if num_trades > 0 and pnl_col in filtered_df.columns else 0.0
    avg_pnl_per_trade = filtered_df[pnl_col].mean() if num_trades > 0 and pnl_col in filtered_df.columns else 0
    max_pnl = filtered_df[pnl_col].max() if num_trades > 0 and pnl_col in filtered_df.columns else 0
    min_pnl = filtered_df[pnl_col].min() if num_trades > 0 and pnl_col in filtered_df.columns else 0
    if pnl_col in filtered_df.columns and collateral_col in filtered_df.columns and filtered_df[collateral_col].sum() != 0:
        avg_roi = (filtered_df[pnl_col].sum() / filtered_df[collateral_col].sum()) * 100
    else:
        avg_roi = 0.0
    total_roi = (total_pnl / initial_balance) * 100 if initial_balance > 0 else 0.0
    avg_holding_period = (filtered_df['expiration_date'] - filtered_df['entry_date']).mean().days if 'expiration_date' in filtered_df.columns and 'entry_date' in filtered_df.columns else 0

    risk_free_rate = 0.0
    sharpe_annualization_factor = np.sqrt(252)
    sharpe_ratio = 0.0
    alpha = beta = portfolio_volatility = None

    if (
        pnl_col in filtered_df.columns
        and 'order_created_at' in filtered_df.columns
        and not benchmark_df.empty
        and initial_balance > 0
    ):
        portfolio_df = filtered_df[['order_created_at', pnl_col]].copy()
        portfolio_df['portfolio_value'] = initial_balance + portfolio_df[pnl_col].cumsum()
        portfolio_df = portfolio_df.rename(columns={'order_created_at': 'date'})
        portfolio_daily = portfolio_df.set_index('date')['portfolio_value'].resample('D').ffill().pct_change().dropna()
        benchmark_daily = benchmark_df.set_index('date')['price'].resample('D').ffill().pct_change().dropna()
        combined_returns = pd.concat([portfolio_daily, benchmark_daily], axis=1).dropna()
        combined_returns.columns = ['portfolio_returns', 'benchmark_returns']
        if len(combined_returns) > 1:
            covariance = combined_returns['portfolio_returns'].cov(combined_returns['benchmark_returns'])
            benchmark_variance = combined_returns['benchmark_returns'].var()
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 0.0
            avg_portfolio_return = combined_returns['portfolio_returns'].mean()
            avg_benchmark_return = combined_returns['benchmark_returns'].mean()
            alpha = (avg_portfolio_return - (risk_free_rate + beta * (avg_benchmark_return - risk_free_rate)))
            alpha = alpha * 252 * 100
            portfolio_volatility = combined_returns['portfolio_returns'].std() * sharpe_annualization_factor
            sharpe_ratio = ((avg_portfolio_return - risk_free_rate) / combined_returns['portfolio_returns'].std() * sharpe_annualization_factor) if portfolio_volatility != 0 else 0.0

    if pnl_col in filtered_df.columns:
        cumulative_pnl = initial_balance + filtered_df[pnl_col].cumsum()
        if not cumulative_pnl.empty:
            running_max = cumulative_pnl.expanding(min_periods=1).max()
            drawdown = running_max - cumulative_pnl
            max_drawdown = drawdown.max()
        else:
            max_drawdown = 0.0
    else:
        max_drawdown = 0.0
    calmar_ratio = total_pnl / max_drawdown if max_drawdown > 0 else (np.inf if total_pnl > 0 else -np.inf)

    # --- Displaying Statistics ---
    st.header("Portfolio Statistics")
    stats_tab1, stats_tab2, stats_tab3 = st.tabs(["Key Metrics", "PnL & ROI", "Advanced Metrics"])

    with stats_tab1:
        cols = st.columns(4)
        metrics1 = [
            ("Total PnL", f"${total_pnl:,.2f}"),
            ("Number of Trades", f"{num_trades:,}"),
            ("Win Rate", f"{win_rate:.2f}%"),
            ("Avg PnL per Trade", f"${avg_pnl_per_trade:,.2f}")
        ]
        for col, (label, value) in zip(cols, metrics1):
            col.metric(label=label, value=value)

    with stats_tab2:
        cols = st.columns(5)
        metrics2 = [
            ("Avg ROI", f"{avg_roi:.2f}%"),
            ("Total ROI", f"{total_roi:.2f}%"),
            ("Max PnL", f"${max_pnl:,.2f}"),
            ("Min PnL", f"${min_pnl:,.2f}"),
            ("Avg Holding Period", f"{avg_holding_period:.0f} days")
        ]
        for col, (label, value) in zip(cols, metrics2):
            col.metric(label=label, value=value)

    with stats_tab3:
        cols = st.columns(7)
        metrics3 = [
            ("Sharpe Ratio", f"{sharpe_ratio:.2f}"),
            ("Portfolio Volatility", f"{portfolio_volatility:.2%}" if portfolio_volatility is not None else "N/A"),
            ("Max Drawdown", f"${max_drawdown:,.2f}"),
            ("Calmar Ratio", f"{calmar_ratio:.2f}"),
            (f"Alpha (vs. {benchmark_display_name})", f"{alpha:.2f}%" if alpha is not None and not np.isnan(alpha) else "N/A"),
            (f"Beta (vs. {benchmark_display_name})", f"{beta:.2f}" if beta is not None and not np.isnan(beta) else "N/A"),
        ]
        for col, (label, value) in zip(cols, metrics3):
            col.metric(label=label, value=value)
        # The 7th column is left for spacing or future use

        with st.expander("What are these metrics?"):
            st.markdown(f"""
            **Sharpe Ratio**: A measure of risk-adjusted return. It indicates how much excess return you receive for the extra volatility you endure for holding a riskier asset. A higher number is better.
            
            $Sharpe Ratio = \\frac{{R_p - R_f}}{{\\sigma_p}}$
            
            Where:
            * $R_p$ = Average return of the portfolio
            * $R_f$ = Risk-free rate of return (assumed to be 0 here)
            * $\\sigma_p$ = Standard deviation of the portfolio's returns

            **Portfolio Volatility**: The annualized standard deviation of daily returns of the portfolio. It measures the degree of variation of returns over time. Higher volatility means more risk.

            **Max Drawdown**: The maximum observed loss from a peak to a trough of a portfolio, before a new peak is attained. It is a key measure of downside risk.
            
            **Calmar Ratio**: A risk-adjusted return metric that uses maximum drawdown in the denominator instead of standard deviation. It's often preferred by traders to understand return relative to worst-case losses. A higher number is better.
            
            $Calmar Ratio = \\frac{{Total Return}}{{Maximum Drawdown}}$

            **Alpha (vs. {benchmark_display_name})**: A measure of a portfolio's performance relative to the selected benchmark, adjusted for risk. A positive alpha means you're outperforming the benchmark. It is calculated using the Capital Asset Pricing Model (CAPM).

            $Alpha = R_p - [R_f + \\beta * (R_m - R_f)]$

            Where:
            * $R_p$ = Average daily return of the portfolio
            * $R_f$ = Risk-free rate of return (assumed to be 0)
            * $\\beta$ = Beta of the portfolio
            * $R_m$ = Average daily return of the market ({benchmark_display_name})
            
            **Beta (vs. {benchmark_display_name})**: A measure of a portfolio's volatility in relation to the selected benchmark. A beta of 1 means your portfolio moves with the market, while a beta > 1 means it is more volatile and a beta < 1 means it is less volatile.
            """)

    st.markdown("---")

    # --- Interactive Charts ---
    st.header("Interactive Charts")
    if not filtered_df.empty:
        chart_tabs = st.tabs([
            "Cumulative PnL", "PnL Distribution", "PnL by Strategy", "ROI Distribution", "Collateral by Strategy"
        ])
        # Cumulative PnL vs. Benchmark
        with chart_tabs[0]:
            st.subheader(f"Cumulative PnL vs. {benchmark_display_name} Benchmark")
            chart_df = filtered_df.copy()
            if pnl_col in chart_df.columns and 'order_created_at' in chart_df.columns and not benchmark_df.empty:
                chart_df = chart_df.sort_values('order_created_at')
                portfolio_cumulative_pnl = chart_df[['order_created_at', pnl_col]].copy()
                portfolio_cumulative_pnl['value'] = initial_balance + portfolio_cumulative_pnl[pnl_col].cumsum()
                portfolio_cumulative_pnl = portfolio_cumulative_pnl.rename(columns={'order_created_at': 'date'})
                portfolio_cumulative_pnl['Type'] = 'Portfolio'
                bench_df = benchmark_df.copy()
                start_date_of_trades = portfolio_cumulative_pnl['date'].min()
                bench_mask = bench_df['date'] >= start_date_of_trades
                if bench_mask.any():
                    bench_start_value = bench_df.loc[bench_mask, 'price'].iloc[0]
                    bench_df = bench_df.loc[bench_mask].copy()
                    bench_df['cumulative_return'] = (bench_df['price'] / bench_start_value)
                    bench_df['value'] = initial_balance * bench_df['cumulative_return']
                    bench_df['Type'] = benchmark_display_name
                    end_date_of_trades = portfolio_cumulative_pnl['date'].max()
                    bench_df = bench_df[bench_df['date'] <= end_date_of_trades]
                    combined_df = pd.concat([portfolio_cumulative_pnl, bench_df[['date', 'value', 'Type']]])
                    fig_line = px.line(
                        combined_df,
                        x='date',
                        y='value',
                        color='Type',
                        title=f'Cumulative Performance: Portfolio vs. {benchmark_display_name}',
                        labels={'date': 'Date', 'value': 'Account Value ($)'},
                        template=plotly_theme
                    )
                    fig_line.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_line, use_container_width=True)
                else:
                    st.info(f"No {benchmark_display_name} data available for the selected date range.")
            else:
                st.info(f"PnL data, order_created_at, or {benchmark_display_name} data not available for cumulative chart.")

        # PnL Distribution
        with chart_tabs[1]:
            st.subheader("PnL Distribution")
            if pnl_col in filtered_df.columns:
                fig_hist = px.histogram(
                    filtered_df,
                    x=pnl_col,
                    nbins=30,
                    title='Distribution of Profit and Loss',
                    labels={pnl_col: 'PnL ($)'},
                    template=plotly_theme
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("PnL data not available for distribution chart.")

        # PnL by Strategy
        with chart_tabs[2]:
            st.subheader("PnL by Strategy")
            if 'opening_strategy' in filtered_df.columns and pnl_col in filtered_df.columns:
                pnl_by_strategy = filtered_df.groupby('opening_strategy')[pnl_col].sum().reset_index()
                fig_bar = px.bar(
                    pnl_by_strategy,
                    x='opening_strategy',
                    y=pnl_col,
                    title='Total PnL by Opening Strategy',
                    labels={'opening_strategy': 'Strategy', pnl_col: 'Total PnL ($)'},
                    template=plotly_theme
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Strategy or PnL data not available for bar chart.")

        # ROI Distribution
        with chart_tabs[3]:
            st.subheader("ROI Distribution")
            if 'roi' in filtered_df.columns:
                fig_roi_hist = px.histogram(
                    filtered_df,
                    x='roi',
                    nbins=30,
                    title='Distribution of Return on Investment (ROI)',
                    labels={'roi': 'ROI (%)'},
                    template=plotly_theme
                )
                st.plotly_chart(fig_roi_hist, use_container_width=True)
            else:
                st.info("ROI data not available for distribution chart.")

        # Collateral by Strategy
        with chart_tabs[4]:
            st.subheader("Collateral by Strategy")
            if 'opening_strategy' in filtered_df.columns and collateral_col in filtered_df.columns:
                collateral_by_strategy = filtered_df.groupby('opening_strategy')[collateral_col].sum().reset_index()
                fig_pie = px.pie(
                    collateral_by_strategy,
                    values=collateral_col,
                    names='opening_strategy',
                    title='Total Collateral by Strategy',
                    labels={collateral_col: 'Collateral', 'opening_strategy': 'Strategy'},
                    template=plotly_theme
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Strategy or collateral data not available for pie chart.")

    st.markdown("---")

    # --- Display Filtered Dataframe ---
    st.header("Filtered Trade Data")
    st.dataframe(filtered_df, use_container_width=True)