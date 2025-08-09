# 🚀 Robinhood Portfolio Dashboard

A modern, sleek, and comprehensive portfolio dashboard inspired by Robinhood's UI/UX design. This dashboard provides advanced analytics and visualizations for your trading portfolio, including options and stock analysis.

## ✨ Features

### 📊 Portfolio Overview
- **Real-time metrics**: Total P&L, win rate, trade counts, and more
- **Interactive cards**: Hover effects and modern design
- **Performance indicators**: Color-coded positive/negative values

### 🎯 Advanced Options Analysis
- **Multi-dimensional filtering**: By symbol, option type, date range, and P&L range
- **Interactive visualizations**: Bar charts, line charts, pie charts, and histograms
- **Performance tracking**: P&L over time and by symbol
- **Risk metrics**: Max gain/loss, standard deviation, Sharpe ratio

### 📈 Stock Analysis
- **Trading activity visualization**: Scatter plots with size and color coding
- **Volume analysis**: Trading volume by symbol and over time
- **Monthly trends**: Trading activity patterns

### ⚠️ Risk Analysis
- **Comprehensive risk metrics**: Options and stock risk analysis
- **Performance ratios**: Sharpe ratio and other risk-adjusted metrics
- **Trade size analysis**: Average and maximum trade sizes

## 🎨 Design Features

- **Modern UI/UX**: Inspired by Robinhood's clean and intuitive design
- **Responsive layout**: Works on desktop and mobile devices
- **Interactive elements**: Hover effects, smooth transitions, and animations
- **Color-coded metrics**: Green for positive, red for negative values
- **Glassmorphism design**: Modern card-based layout with backdrop blur effects

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed on your system
2. **CSV data files** in the project directory:
   - `options.csv` - Options trading data
   - `stock_orders_Jan-20-2023.csv` - Stock trading data
   - `option_orders_Jan-20-2023.csv` - Option orders data

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd robinhood-dashboard
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**:
   ```bash
   # For basic dashboard
   streamlit run dashboard.py
   
   # For advanced dashboard (recommended)
   streamlit run advanced_dashboard.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📁 Project Structure

```
robinhood-dashboard/
├── dashboard.py              # Basic dashboard
├── advanced_dashboard.py     # Advanced dashboard (recommended)
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── options.csv              # Options trading data
├── stock_orders_Jan-20-2023.csv    # Stock trading data
└── option_orders_Jan-20-2023.csv   # Option orders data
```

## 🔧 Configuration

### Data Format Requirements

#### Options Data (`options.csv`)
```csv
chain_symbol,expiration_date,strike_price,option_type,buy,sell,p&l
AAPL,2025-08-01,217.5,call,56.99,74.0,17.0
```

#### Stock Orders Data (`stock_orders_Jan-20-2023.csv`)
```csv
symbol,date,order_type,side,fees,quantity,average_price
AAPL,2025-07-31T18:05:45.750000Z,market,sell,0,1.00000000,98.02000000
```

#### Option Orders Data (`option_orders_Jan-20-2023.csv`)
```csv
chain_symbol,expiration_date,strike_price,option_type,side,order_created_at,direction,order_quantity,order_type,opening_strategy,closing_strategy,price,processed_quantity
```

## 🎯 Usage Guide

### Dashboard Navigation

1. **Portfolio Overview**: View key metrics and performance indicators
2. **Options Analysis**: Analyze options performance with advanced filters
3. **Stock Analysis**: Review stock trading activity and volume
4. **Risk Analysis**: Assess portfolio risk and performance metrics

### Filtering and Analysis

- **Options Analysis**: Filter by symbol, option type, date range, and P&L range
- **Interactive Charts**: Hover over data points for detailed information
- **Real-time Updates**: Data refreshes automatically when filters change

## 🎨 Customization

### Styling
The dashboard uses custom CSS for a modern look. You can modify the styling in the `advanced_dashboard.py` file:

```css
.metric-card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}
```

### Adding New Metrics
To add new metrics, modify the `calculate_advanced_metrics()` function in `advanced_dashboard.py`.

## 🔍 Troubleshooting

### Common Issues

1. **Data not loading**: Ensure CSV files are in the correct location and format
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Port already in use**: Change the port with `streamlit run dashboard.py --server.port 8502`

### Error Messages

- **"Failed to load data"**: Check if CSV files exist and are properly formatted
- **"Module not found"**: Install missing dependencies from requirements.txt

## 📊 Data Privacy

- All data is processed locally on your machine
- No data is sent to external servers
- CSV files remain in your local directory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by Robinhood's modern UI/UX design
- Built with Streamlit for rapid web application development
- Uses Plotly for interactive visualizations
- Powered by Pandas for data analysis

---

**Happy Trading! 📈**
