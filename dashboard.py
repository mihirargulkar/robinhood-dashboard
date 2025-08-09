import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# --- Configuration & UI Setup ---
st.set_page_config(
    page_title="Mills Investment Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data():
    """
    Loads and preprocesses the CSV data from the default file.
    """
    # Try both with and without 'data/' prefix for compatibility
    possible_paths = ['combined_trades.csv', os.path.join('data', 'combined_trades.csv')]
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                break
            except Exception as e:
                st.error(f"Error reading '{path}': {e}")
                return pd.DataFrame()
    if df is None:
        st.error("Error: The file 'combined_trades.csv' was not found. Please ensure it is in the app directory or in a 'data/' subdirectory.")
        return pd.DataFrame()
    
    # Clean and convert data types for proper analysis and visualization.
    if 'expiration_date' in df.columns:
        df['expiration_date'] = pd.to_datetime(df['expiration_date'], errors='coerce')
    if 'strike_price' in df.columns:
        df['strike_price'] = pd.to_numeric(df['strike_price'], errors='coerce')
        df.dropna(subset=['strike_price'], inplace=True)
    # Sort by date for time series analysis.
    if 'expiration_date' in df.columns:
        df.sort_values('expiration_date', inplace=True)
    return df

# Main title and introduction
st.title("Mills Investment Dashboard")
st.markdown("A comprehensive and interactive overview of our portfolio performance.")
st.markdown("---")

# Load the data
df = load_data()

# Only proceed if the data was loaded successfully
if not df.empty:
    st.sidebar.markdown("---")
    st.sidebar.header("Filter Options")

    # A button to reset all filters
    if st.sidebar.button("Reset Filters"):
        st.rerun()

    # Filter by Strategy
    all_strategies = ['All'] + list(df['opening_strategy'].unique()) if 'opening_strategy' in df.columns else ['All']
    selected_strategies = st.sidebar.multiselect(
        "Select Opening Strategy(s)",
        options=all_strategies,
        default='All'
    )
    
    # Filter by Symbol
    all_symbols = ['All'] + sorted(list(df['symbol'].unique())) if 'symbol' in df.columns else ['All']
    selected_symbols = st.sidebar.multiselect(
        "Select Trading Symbol(s)",
        options=all_symbols,
        default='All'
    )

    # Filter by Expiration Date Range
    if 'expiration_date' in df.columns and not df['expiration_date'].isnull().all():
        min_date = df['expiration_date'].min().date()
        max_date = df['expiration_date'].max().date()
        date_range = st.sidebar.date_input(
            "Select Expiration Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        # Handle both single date and range
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        elif isinstance(date_range, list) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = min_date
    else:
        start_date = end_date = None

    # Apply the filters to the dataframe
    filtered_df = df.copy()
    if 'All' not in selected_strategies and 'opening_strategy' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['opening_strategy'].isin(selected_strategies)]
    if 'All' not in selected_symbols and 'symbol' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['symbol'].isin(selected_symbols)]
    if start_date is not None and end_date is not None and 'expiration_date' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['expiration_date'].dt.date >= start_date) &
            (filtered_df['expiration_date'].dt.date <= end_date)
        ]

    # --- Portfolio Statistics Calculations ---
    total_pnl = filtered_df['pnl'].sum() if 'pnl' in filtered_df.columns else 0
    num_trades = len(filtered_df)
    win_rate = (filtered_df[filtered_df['pnl'] > 0].shape[0] / num_trades * 100) if num_trades > 0 and 'pnl' in filtered_df.columns else 0.0
    avg_pnl_per_trade = filtered_df['pnl'].mean() if num_trades > 0 and 'pnl' in filtered_df.columns else 0
    max_pnl = filtered_df['pnl'].max() if num_trades > 0 and 'pnl' in filtered_df.columns else 0
    min_pnl = filtered_df['pnl'].min() if num_trades > 0 and 'pnl' in filtered_df.columns else 0
    avg_roi = filtered_df['roi'].mean() if num_trades > 0 and 'roi' in filtered_df.columns else 0

    # Calculate Sharpe Ratio
    risk_free_rate = 0.0
    if 'pnl' in filtered_df.columns and 'collateral' in filtered_df.columns:
        returns = filtered_df['pnl'] / filtered_df['collateral']
        if len(returns) > 1 and returns.std() != 0:
            sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    # Calculate Max Drawdown
    if 'pnl' in filtered_df.columns:
        cumulative_pnl = filtered_df['pnl'].cumsum()
        if not cumulative_pnl.empty:
            running_max = cumulative_pnl.expanding(min_periods=1).max()
            drawdown = running_max - cumulative_pnl
            max_drawdown = drawdown.max()
        else:
            max_drawdown = 0.0
    else:
        max_drawdown = 0.0

    # Calculate Calmar Ratio
    calmar_ratio = total_pnl / max_drawdown if max_drawdown > 0 else (np.inf if total_pnl > 0 else -np.inf)

    # --- Displaying Statistics ---
    st.header("Portfolio Statistics")
    
    # Use tabs for a cleaner layout
    stats_tab1, stats_tab2 = st.tabs(["Key Metrics", "PnL & ROI"])

    with stats_tab1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Total PnL", value=f"${total_pnl:,.2f}")
        with col2:
            st.metric(label="Number of Trades", value=f"{num_trades:,}")
        with col3:
            st.metric(label="Win Rate", value=f"{win_rate:.2f}%")
        with col4:
            st.metric(label="Avg PnL per Trade", value=f"${avg_pnl_per_trade:,.2f}")

    with stats_tab2:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric(label="Sharpe Ratio", value=f"{sharpe_ratio:.2f}")
        with col2:
            st.metric(label="Max Drawdown", value=f"${max_drawdown:,.2f}")
        with col3:
            st.metric(label="Calmar Ratio", value=f"{calmar_ratio:.2f}")
        with col4:
            st.metric(label="Max PnL", value=f"${max_pnl:,.2f}")
        with col5:
            st.metric(label="Min PnL", value=f"${min_pnl:,.2f}")
        
    st.markdown("---")

    # --- Interactive Charts ---
    st.header("Interactive Charts")
    if not filtered_df.empty:
        chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs(["Cumulative PnL", "PnL Distribution", "PnL by Strategy", "ROI Distribution"])
        
        with chart_tab1:
            st.subheader("Cumulative PnL Over Time")
            chart_df = filtered_df.copy()
            if 'pnl' in chart_df.columns:
                chart_df['cumulative_pnl'] = chart_df['pnl'].cumsum()
                fig_line = px.line(
                    chart_df,
                    x='expiration_date' if 'expiration_date' in chart_df.columns else chart_df.index,
                    y='cumulative_pnl',
                    title='Cumulative Profit and Loss',
                    labels={'expiration_date': 'Expiration Date', 'cumulative_pnl': 'Cumulative PnL ($)'},
                    hover_data={'pnl': ':.2f', 'symbol': True, 'opening_strategy': True, 'expiration_date': '|%Y-%m-%d'} if 'expiration_date' in chart_df.columns else None
                )
                fig_line.update_layout(hovermode="x unified")
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("PnL data not available for cumulative chart.")
        
        with chart_tab2:
            st.subheader("PnL Distribution")
            if 'pnl' in filtered_df.columns:
                fig_hist = px.histogram(
                    filtered_df,
                    x='pnl',
                    nbins=30,
                    title='Distribution of Profit and Loss',
                    labels={'pnl': 'PnL ($)'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("PnL data not available for distribution chart.")

        with chart_tab3:
            st.subheader("PnL by Strategy")
            if 'opening_strategy' in filtered_df.columns and 'pnl' in filtered_df.columns:
                pnl_by_strategy = filtered_df.groupby('opening_strategy')['pnl'].sum().reset_index()
                fig_bar = px.bar(
                    pnl_by_strategy,
                    x='opening_strategy',
                    y='pnl',
                    title='Total PnL by Opening Strategy',
                    labels={'opening_strategy': 'Strategy', 'pnl': 'Total PnL ($)'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Strategy or PnL data not available for bar chart.")

        with chart_tab4:
            st.subheader("ROI Distribution")
            if 'roi' in filtered_df.columns:
                fig_roi_hist = px.histogram(
                    filtered_df,
                    x='roi',
                    nbins=30,
                    title='Distribution of Return on Investment (ROI)',
                    labels={'roi': 'ROI (%)'}
                )
                st.plotly_chart(fig_roi_hist, use_container_width=True)
            else:
                st.info("ROI data not available for distribution chart.")
    else:
        st.warning("No data to display based on the current filter selection.")

    st.markdown("---")

    # --- Display Filtered Dataframe ---
    st.header("Filtered Trade Data")
    st.dataframe(filtered_df, use_container_width=True)
