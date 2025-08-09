import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern, sleek CSS design (Robinhood-inspired)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        background: #fcfcfc;
    }
    
    .stApp {
        background: #fcfcfc;
    }
    
    /* Modern card design */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        border: 1px solid #e5e7eb;
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transform: translateY(-1px);
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 4px;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .positive {
        color: #06b6d4 !important; /* A light teal for a modern look */
    }
    
    .negative {
        color: #ef4444 !important; /* A bright red for clear negative indication */
    }
    
    /* Section headers */
    .section-header {
        margin: 24px 0 16px 0;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #111827;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        border: 1px solid #e5e7eb;
    }
    
    /* Filters - make them blend in better */
    .stSelectbox > div > div, .stDateInput > div > div, .stSlider .stSlider {
        background: white;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #111827;
        font-weight: 600;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: white;
    }
    
    /* Hide Streamlit elements */
    .stDeployButton {
        display: none;
    }
    
    #MainMenu {
        visibility: hidden;
    }
    
    footer {
        visibility: hidden;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f9fafb;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #d1d5db;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #9ca3af;
    }
    
    /* Gradient text - A more subtle, cool-toned gradient */
    .gradient-text {
        background: linear-gradient(135deg, #1d4ed8, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-success {
        background: #10b981;
    }
    
    .status-warning {
        background: #f59e0b;
    }
    
    .status-error {
        background: #ef4444;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and process all CSV data"""
    try:
        # Load options data
        options_df = pd.read_csv('options.csv')
        options_df['expiration_date'] = pd.to_datetime(options_df['expiration_date'])
        options_df['p&l'] = pd.to_numeric(options_df['p&l'], errors='coerce')
        
        # Load stock orders with flexible datetime parsing
        stock_orders_df = pd.read_csv('stock_orders_Jan-20-2023.csv')
        stock_orders_df['date'] = pd.to_datetime(stock_orders_df['date'], format='mixed', utc=True)
        
        # Load option orders with flexible datetime parsing
        option_orders_df = pd.read_csv('option_orders_Jan-20-2023.csv')
        option_orders_df['expiration_date'] = pd.to_datetime(option_orders_df['expiration_date'])
        option_orders_df['order_created_at'] = pd.to_datetime(option_orders_df['order_created_at'], format='mixed', utc=True)
        
        return options_df, stock_orders_df, option_orders_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

def calculate_advanced_metrics(options_df, stock_orders_df, option_orders_df):
    """Calculate advanced portfolio metrics"""
    metrics = {}
    
    if options_df is not None and not options_df.empty:
        # Options metrics
        metrics['total_options_pnl'] = options_df['p&l'].sum()
        metrics['options_count'] = len(options_df)
        metrics['profitable_options'] = len(options_df[options_df['p&l'] > 0])
        metrics['win_rate'] = (metrics['profitable_options'] / metrics['options_count'] * 100) if metrics['options_count'] > 0 else 0
        
        # Advanced options metrics
        metrics['avg_options_pnl'] = options_df['p&l'].mean()
        metrics['options_std'] = options_df['p&l'].std()
        metrics['max_options_gain'] = options_df['p&l'].max()
        metrics['max_options_loss'] = options_df['p&l'].min()
        
        # Options by type
        call_options = options_df[options_df['option_type'] == 'call']
        put_options = options_df[options_df['option_type'] == 'put']
        metrics['call_pnl'] = call_options['p&l'].sum() if not call_options.empty else 0
        metrics['put_pnl'] = put_options['p&l'].sum() if not put_options.empty else 0
    
    if stock_orders_df is not None and not stock_orders_df.empty:
        # Stock metrics
        stock_orders_df['total_value'] = stock_orders_df['quantity'] * stock_orders_df['average_price']
        metrics['total_stock_volume'] = stock_orders_df['total_value'].sum()
        metrics['stock_trades'] = len(stock_orders_df)
        metrics['avg_trade_size'] = stock_orders_df['total_value'].mean()
        
        # Stock performance by side
        buy_trades = stock_orders_df[stock_orders_df['side'] == 'buy']
        sell_trades = stock_orders_df[stock_orders_df['side'] == 'sell']
        metrics['buy_volume'] = buy_trades['total_value'].sum() if not buy_trades.empty else 0
        metrics['sell_volume'] = sell_trades['total_value'].sum() if not sell_trades.empty else 0
    
    return metrics

def create_modern_portfolio_overview(metrics):
    """Create modern portfolio overview section"""
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title">
            <span class="gradient-text">📊</span>
            Portfolio Overview
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Main metrics in a clean grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value {'positive' if metrics.get('total_options_pnl', 0) >= 0 else 'negative'}">
                ${metrics.get('total_options_pnl', 0):,.2f}
            </div>
            <div class="metric-label">Total Options P&L</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">
                {metrics.get('win_rate', 0):.1f}%
            </div>
            <div class="metric-label">Win Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">
                {metrics.get('options_count', 0)}
            </div>
            <div class="metric-label">Total Options</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">
                {metrics.get('stock_trades', 0)}
            </div>
            <div class="metric-label">Stock Trades</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Secondary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">
                ${metrics.get('avg_options_pnl', 0):,.2f}
            </div>
            <div class="metric-label">Avg Options P&L</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">
                ${metrics.get('total_stock_volume', 0):,.0f}
            </div>
            <div class="metric-label">Stock Volume</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">
                ${metrics.get('call_pnl', 0):,.2f}
            </div>
            <div class="metric-label">Call Options P&L</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">
                ${metrics.get('put_pnl', 0):,.2f}
            </div>
            <div class="metric-label">Put Options P&L</div>
        </div>
        """, unsafe_allow_html=True)

def create_modern_options_analysis(options_df):
    """Create modern options analysis section"""
    if options_df is None or options_df.empty:
        return
    
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title">
            <span class="gradient-text">🎯</span>
            Options Analysis
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Modern filters in a clean layout
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        option_types = ['All'] + list(options_df['option_type'].unique())
        selected_type = st.selectbox("Option Type", option_types, key="option_type_filter")
    
    with col2:
        symbols = ['All'] + sorted(list(options_df['chain_symbol'].unique()))
        selected_symbol = st.selectbox("Symbol", symbols, key="symbol_filter")
    
    with col3:
        min_pnl = options_df['p&l'].min()
        max_pnl = options_df['p&l'].max()
        pnl_range = st.slider(
            "P&L Range",
            min_value=float(min_pnl),
            max_value=float(max_pnl),
            value=(float(min_pnl), float(max_pnl)),
            key="pnl_range_filter"
        )
    
    with col4:
        date_range = st.date_input(
            "Expiration Date Range",
            value=(options_df['expiration_date'].min(), options_df['expiration_date'].max()),
            min_value=options_df['expiration_date'].min(),
            max_value=options_df['expiration_date'].max(),
            key="date_range_filter"
        )
    
    # Filter data
    filtered_df = options_df.copy()
    if selected_type != 'All':
        filtered_df = filtered_df[filtered_df['option_type'] == selected_type]
    if selected_symbol != 'All':
        filtered_df = filtered_df[filtered_df['chain_symbol'] == selected_symbol]
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['expiration_date'] >= pd.to_datetime(date_range[0])) &
            (filtered_df['expiration_date'] <= pd.to_datetime(date_range[1]))
        ]
    filtered_df = filtered_df[
        (filtered_df['p&l'] >= pnl_range[0]) &
        (filtered_df['p&l'] <= pnl_range[1])
    ]
    
    # Modern visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Options P&L by symbol
        symbol_pnl = filtered_df.groupby('chain_symbol')['p&l'].sum().reset_index()
        symbol_pnl = symbol_pnl.sort_values('p&l', ascending=False)
        
        fig = px.bar(
            symbol_pnl,
            x='chain_symbol',
            y='p&l',
            title="Options P&L by Symbol",
            color='p&l',
            color_continuous_scale=['#ef4444', '#06b6d4'],
            height=400
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_font_size=16,
            showlegend=False,
            font=dict(family="Inter, sans-serif"),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Options performance over time
        time_pnl = filtered_df.groupby('expiration_date')['p&l'].sum().reset_index()
        fig2 = px.line(
            time_pnl,
            x='expiration_date',
            y='p&l',
            title="Options P&L Over Time",
            height=400
        )
        fig2.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_font_size=16,
            font=dict(family="Inter, sans-serif"),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        fig2.update_xaxes(showgrid=False)
        fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Options by type distribution
        type_dist = filtered_df['option_type'].value_counts()
        fig3 = px.pie(
            values=type_dist.values,
            names=type_dist.index,
            title="Options Distribution by Type",
            color_discrete_sequence=['#1d4ed8', '#06b6d4']
        )
        fig3.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_font_size=16,
            font=dict(family="Inter, sans-serif"),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # P&L distribution histogram
        fig4 = px.histogram(
            filtered_df,
            x='p&l',
            nbins=20,
            title="P&L Distribution",
            color_discrete_sequence=['#1d4ed8']
        )
        fig4.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_font_size=16,
            font=dict(family="Inter, sans-serif"),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        fig4.update_xaxes(showgrid=False)
        fig4.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def create_modern_stock_analysis(stock_orders_df):
    """Create modern stock analysis section"""
    if stock_orders_df is None or stock_orders_df.empty:
        return
    
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title">
            <span class="gradient-text">📈</span>
            Stock Analysis
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Stock trading activity with modern design
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    fig = px.scatter(
        stock_orders_df,
        x='date',
        y='average_price',
        size='quantity',
        color='side',
        hover_data=['symbol', 'order_type', 'fees'],
        title="Stock Trading Activity Over Time",
        color_discrete_map={'buy': '#06b6d4', 'sell': '#ef4444'}
    )
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        title_font_size=16,
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Trading volume analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Trading volume by symbol
        volume_by_symbol = stock_orders_df.groupby('symbol').agg({
            'quantity': 'sum',
            'average_price': 'mean'
        }).reset_index()
        volume_by_symbol['total_value'] = volume_by_symbol['quantity'] * volume_by_symbol['average_price']
        volume_by_symbol = volume_by_symbol.sort_values('total_value', ascending=False)
        
        fig2 = px.bar(
            volume_by_symbol,
            x='symbol',
            y='total_value',
            title="Trading Volume by Symbol",
            color='total_value',
            color_continuous_scale='viridis'
        )
        fig2.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_font_size=16,
            font=dict(family="Inter, sans-serif"),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        fig2.update_xaxes(showgrid=False)
        fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Trading activity by month
        stock_orders_df['month'] = stock_orders_df['date'].dt.to_period('M')
        monthly_activity = stock_orders_df.groupby('month').size().reset_index(name='trades')
        monthly_activity['month'] = monthly_activity['month'].astype(str)
        
        fig3 = px.line(
            monthly_activity,
            x='month',
            y='trades',
            title="Monthly Trading Activity",
            markers=True
        )
        fig3.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_font_size=16,
            font=dict(family="Inter, sans-serif"),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        fig3.update_xaxes(showgrid=False)
        fig3.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def create_modern_risk_analysis(options_df, stock_orders_df):
    """Create modern risk analysis section"""
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title">
            <span class="gradient-text">⚠️</span>
            Risk Analysis
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if options_df is not None and not options_df.empty:
            # Options risk metrics
            max_loss = options_df['p&l'].min()
            max_gain = options_df['p&l'].max()
            avg_pnl = options_df['p&l'].mean()
            std_pnl = options_df['p&l'].std()
            
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #111827; margin-bottom: 16px;">Options Risk Metrics</h4>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="color: #6b7280;">Max Loss:</span>
                    <span class="negative">${max_loss:,.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="color: #6b7280;">Max Gain:</span>
                    <span class="positive">${max_gain:,.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="color: #6b7280;">Average P&L:</span>
                    <span class="{'positive' if avg_pnl >= 0 else 'negative'}">${avg_pnl:,.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="color: #6b7280;">Standard Deviation:</span>
                    <span>${std_pnl:,.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #6b7280;">Sharpe Ratio:</span>
                    <span>{(avg_pnl/std_pnl if std_pnl != 0 else 0):.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if stock_orders_df is not None and not stock_orders_df.empty:
            # Stock risk metrics
            stock_orders_df['total_value'] = stock_orders_df['quantity'] * stock_orders_df['average_price']
            total_volume = stock_orders_df['total_value'].sum()
            avg_trade_size = stock_orders_df['total_value'].mean()
            max_trade = stock_orders_df['total_value'].max()
            
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #111827; margin-bottom: 16px;">Stock Risk Metrics</h4>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="color: #6b7280;">Total Volume:</span>
                    <span>${total_volume:,.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="color: #6b7280;">Average Trade Size:</span>
                    <span>${avg_trade_size:,.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="color: #6b7280;">Max Trade Size:</span>
                    <span>${max_trade:,.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="color: #6b7280;">Number of Trades:</span>
                    <span>{len(stock_orders_df)}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #6b7280;">Total Fees:</span>
                    <span>${stock_orders_df['fees'].sum():,.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main dashboard function"""
    # Modern header
    st.markdown("""
    <div style="text-align: center; padding: 40px 0; background: white; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); border: 1px solid #e5e7eb;">
        <h1 style="font-size: 2.5rem; font-weight: 700; margin: 0; background: linear-gradient(135deg, #1d4ed8, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
            Portfolio Dashboard
        </h1>
        <p style="color: #6b7280; font-size: 1.1rem; margin: 8px 0 0 0;">
            Modern analytics for your trading portfolio
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    options_df, stock_orders_df, option_orders_df = load_data()
    
    if options_df is None:
        st.error("Failed to load data. Please check if the CSV files are in the correct location.")
        return
    
    # Calculate advanced metrics
    metrics = calculate_advanced_metrics(options_df, stock_orders_df, option_orders_df)
    
    # Create dashboard sections
    create_modern_portfolio_overview(metrics)
    st.markdown("---")
    create_modern_options_analysis(options_df)
    st.markdown("---")
    create_modern_stock_analysis(stock_orders_df)
    st.markdown("---")
    create_modern_risk_analysis(options_df, stock_orders_df)

if __name__ == "__main__":
    main()