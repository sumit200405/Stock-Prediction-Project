# Multi-Stock Price Prediction Project

*Because predicting the stock market is totally easy, right? Now with MULTIPLE stocks for extra complexity!*

## Overview

Welcome to yet another attempt at predicting stock prices! This project uses various machine learning models to forecast stock prices, including traditional algorithms and deep learning approaches. The twist? We train on data from MULTIPLE stocks to create more robust models that can generalize better. Will it work? Maybe. Should you bet your life savings on it? Absolutely not.

## What This Project Does

- **Fetches multi-stock data** from Yahoo Finance for ~40 diverse stocks across sectors
- **Trains on combined dataset** for better generalization (tech stocks, finance, healthcare, etc.)
- **Performs detailed analysis** on your chosen target stock while training on the full dataset
- **Engineers features** like a boss (moving averages, RSI, volatility, etc.)
- **Trains multiple models** including Linear Regression, Random Forest, Gradient Boosting, and LSTM
- **Compares model performance** because competition is healthy
- **Generates future predictions** (with a giant disclaimer about how wrong they probably are)
- **Creates beautiful visualizations** that make everything look professional

## Stuff thats going to happen

### Multi-Stock Training
- **Diverse Stock Selection**: Trains on ~40 stocks from different sectors (tech, finance, healthcare, etc.)
- **Combined Dataset**: Uses data from multiple stocks for robust model training
- **Target Stock Analysis**: Detailed analysis and predictions for your chosen stock
- **Sector Diversification**: Better generalization across different market conditions

### Data Analysis
- **Multi-Stock Data Fetching**: Downloads historical data for multiple stocks using yfinance
- **Technical Indicators**: RSI, Moving Averages, Volatility calculations
- **Exploratory Analysis**: Price trends, volume analysis, return distributions
- **Feature Engineering**: Lag features, ratios, and derived metrics with proper grouping by stock

### Machine Learning Models
- **Traditional Models**: Linear Regression, Random Forest, Gradient Boosting
- **Deep Learning**: LSTM neural networks for time series prediction
- **Model Comparison**: Performance metrics and visualizations
- **Future Predictions**: Generates forecasts (with appropriate disclaimers)

### Visualizations
- Stock price charts with moving averages
- Trading volume analysis
- RSI and volatility indicators
- Model performance comparisons
- Prediction accuracy visualizations
- Comprehensive summary reports

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd Stock-Price-Prediction
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the analysis:
   ```bash
   python main.py
   ```

## Usage

Simply run the main script and follow the prompts:

```bash
python main.py
```

The script will:
1. Download historical data for ~40 diverse stocks across sectors (tech, finance, healthcare, etc.)
2. Ask you for a target stock symbol for detailed analysis (defaults to AAPL)
3. Perform comprehensive multi-stock analysis
4. Train models on combined dataset from all stocks
5. Generate predictions and detailed visualizations for your target stock
6. Save everything to appropriate directories

**Note:** The first run will take longer as it downloads data for multiple stocks, but this results in much more robust models!

## Project Structure

```
Stock-Price-Prediction/
├── main.py              # Main analysis script
├── requirements.txt     # Package dependencies
├── README.md           # This file
├── data/               # Raw data files
│   ├── multi_stock_data.csv    # Combined data from all stocks
│   └── target_stock_data.csv   # Data for your target stock
├── plots/              # Generated visualizations
│   ├── target_stock_analysis.png
│   ├── multi_stock_overview.png
│   ├── technical_indicators.png
│   ├── model_comparison.png
│   ├── individual_predictions.png
│   ├── accuracy_analysis.png
│   └── summary_report.png
├── stock_price_model.pkl # Best traditional model
├── lstm_stock_model.pth # LSTM model
└── scaler.pkl          # Data scaler
```

## Models Used

### Traditional Machine Learning
- **Linear Regression**: Simple but effective baseline
- **Random Forest**: Ensemble method that handles non-linearity
- **Gradient Boosting**: Another ensemble approach with sequential learning

### Deep Learning
- **LSTM**: Long Short-Term Memory networks for sequence prediction
- **Feature Engineering**: Multiple lag features and technical indicators

**Key Improvement**: All models are trained on the combined dataset from multiple stocks, making them more robust and better at generalizing to different market conditions!

## Performance Metrics

The project evaluates models using:
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **Visual Comparisons**: Actual vs predicted plots

## Multi-Stock Training Benefits

**Why train on multiple stocks?**
- **Better Generalization**: Models learn from diverse market patterns
- **Reduced Overfitting**: Less likely to memorize quirks of a single stock
- **Sector Diversification**: Exposure to different industry dynamics
- **Robustness**: More stable performance across different market conditions
- **Data Augmentation**: More training examples for better learning

## Important Disclaimers (Ive added them to the code too)

**WARNING**: This project is for educational purposes only!

- Past performance does not guarantee future results
- 💸 Don't use this for actual investment decisions
- 🎲 The stock market is inherently unpredictable
- 🤷 We're not responsible if you lose money
- 📊 These are just mathematical models, not financial advice

## What You'll Learn

- How to fetch and analyze financial data
- Feature engineering for time series data
- Training and comparing multiple ML models
- Multi-stock dataset creation and management
- Creating professional data visualizations
- The challenges of financial prediction
- Why you shouldn't trust stock prediction models

## Sample Output

The script generates numerous plots and statistics:
- Multi-stock overview and sector analysis
- Target stock analysis with technical indicators
- Return distributions and volatility analysis
- Model performance comparisons trained on diverse data
- Future price predictions (with appropriate skepticism)
- Comprehensive summary reports

## Technical Details

### Data Sources
- Yahoo Finance API via yfinance library
- Historical stock data with OHLCV information
- Technical indicators calculated from price data

### Feature Engineering
- Moving averages (20-day, 50-day)
- RSI (Relative Strength Index)
- Volatility measures
- Lag features for temporal dependencies
- Volume ratios and price changes

### Model Architecture
- Traditional models use scikit-learn
- LSTM uses TensorFlow/Keras
- Feature scaling and normalization
- Train/test splits with temporal ordering

## Fun Facts

- The best model typically achieves R² scores around 0.8-0.95
- LSTM models often perform similarly to traditional methods
- Random predictions sometimes outperform complex models
- Professional traders have access to much more data and compute power
- Even the best models struggle with market volatility

## Why This Project Exists

Because everyone needs to learn that:
1. Stock prediction is really, really hard
2. Machine learning isn't magic
3. Historical patterns don't always continue
4. The market is influenced by countless unpredictable factors
5. Data science is fun, even when the results are humbling

## Contributing

Feel free to contribute improvements, but remember:
- Keep the casual, educational tone
- Add appropriate disclaimers for financial content
- Focus on learning rather than profit
- Make sure plots are saved correctly

## License

This project is for educational purposes. Use at your own risk (and don't blame me if you lose money)!

---

*Remember: The only thing predictable about the stock market is its unpredictability! (Damn i should write a book or something)*
