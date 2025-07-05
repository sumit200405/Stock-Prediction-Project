# Have good habits, make your mom proud and read the readme file always before you run this code. Thank you.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import pickle
from tqdm import tqdm
import time
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# suppress warnings for cleaner output (because nobody likes spam)
warnings.filterwarnings('ignore')

# create directory for saving plots
print("Setting up environment...")
os.makedirs('plots', exist_ok=True)
os.makedirs('data', exist_ok=True)

# set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
print("Environment setup complete")

# set random seeds because reproducibility is important (unlike stock predictions)
np.random.seed(42)
torch.manual_seed(42)

# configuration
PERIOD = "5y"    # 5 years of data because we need history

# diverse stock list for training (mix of sectors and market caps)
STOCK_LIST = [
    # Tech giants
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA',
    # Finance
    'JPM', 'BAC', 'WFC', 'GS', 'MS',
    # Healthcare
    'JNJ', 'PFE', 'UNH', 'MRK', 'ABBV',
    # Consumer goods
    'KO', 'PG', 'WMT', 'HD', 'MCD',
    # Industrial
    'BA', 'CAT', 'GE', 'MMM', 'UPS',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB',
    # Utilities
    'NEE', 'DUK', 'SO'
]

# stock price prediction system
print("="*60)
print("MULTI-STOCK PRICE PREDICTION SYSTEM")
print("="*60)
print("Don't use this to make actual investment decisions, for godsake, don't!")
print("The stock market is inherently unpredictable so dont bet your freaking house based on this!")
print("="*60)
print(f"Training on {len(STOCK_LIST)} different stocks for better generalization!")
print("This will take longer but results should be much more robust!")

# ask user for target stock symbol for analysis
target_symbol = input("\nEnter target stock symbol for detailed analysis (or press Enter for AAPL): ").upper()
if not target_symbol:
    target_symbol = "AAPL"
print(f"Target stock for analysis: {target_symbol}")
print(f"Training on: {', '.join(STOCK_LIST[:10])}... and {len(STOCK_LIST)-10} more")

# fetch multi-stock data
print(f"\nFetching data for {len(STOCK_LIST)} stocks...")
print("This might take a few minutes but will result in a much better model!")

def fetch_stock_data(symbol, period=PERIOD):
    """Fetch and process data for a single stock"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        if data.empty:
            print(f"No data found for {symbol}")
            return None
            
        # add technical indicators
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        
        # calculate RSI
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        data['RSI'] = calculate_rsi(data['Close'])
        data['Volatility'] = data['Close'].rolling(window=20).std()
        data['Returns'] = data['Close'].pct_change()
        
        # add stock identifier
        data['Symbol'] = symbol
        
        return data
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# fetch data for all stocks
all_stock_data = []
failed_stocks = []

with tqdm(total=len(STOCK_LIST), desc="Fetching multi-stock data", unit="stock") as pbar:
    for symbol in STOCK_LIST:
        pbar.set_description(f"Fetching {symbol}")
        data = fetch_stock_data(symbol)
        
        if data is not None:
            all_stock_data.append(data)
            pbar.set_postfix({"Success": len(all_stock_data), "Failed": len(failed_stocks)})
        else:
            failed_stocks.append(symbol)
            
        pbar.update(1)
        time.sleep(0.1)  # be nice to the API

# combine all stock data
if not all_stock_data:
    print("No stock data could be fetched! Exiting.")
    exit()

print(f"\nSuccessfully fetched data for {len(all_stock_data)} stocks")
if failed_stocks:
    print(f"Failed to fetch: {', '.join(failed_stocks)}")

# combine all data into one dataframe
df_combined = pd.concat(all_stock_data, ignore_index=False)
df_combined = df_combined.sort_index()  # sort by date

print(f"Combined dataset shape: {df_combined.shape}")
print(f"Date range: {df_combined.index[0].date()} to {df_combined.index[-1].date()}")
print(f"Stocks included: {len(df_combined['Symbol'].unique())} unique symbols")

# save combined data
df_combined.to_csv('data/multi_stock_data.csv')

# also get specific data for target stock for detailed analysis
target_data = fetch_stock_data(target_symbol)
if target_data is None:
    print(f"Could not fetch data for target stock {target_symbol}, using AAPL instead")
    target_symbol = "AAPL"
    target_data = fetch_stock_data(target_symbol)

# save target stock data
target_data.to_csv('data/target_stock_data.csv')

print(f"\nTarget stock ({target_symbol}) analysis:")
print(f"   Price range: ${target_data['Close'].min():.2f} - ${target_data['Close'].max():.2f}")
print(f"   Total return: {((target_data['Close'][-1] / target_data['Close'][0]) - 1) * 100:.2f}%")
print(f"   Data points: {len(target_data)}")

print(f"\nMulti-stock dataset overview:")
print(f"   Total data points: {len(df_combined)}")
print(f"   Memory usage: {df_combined.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# check for missing values in combined dataset
print(f"\nMissing Values Check (Combined Dataset):")
missing_values = df_combined.isnull().sum()
total_missing = missing_values.sum()
print(f"Total missing values: {total_missing}")

if total_missing > 0:
    print(f"Found {total_missing} missing values")
    # drop rows with missing values
    df_combined = df_combined.dropna()
    print(f"Trimmed to {len(df_combined)} rows")
else:
    print("No missing values found!")

# also clean target data
target_data = target_data.dropna()
print(f"Target stock data after cleaning: {len(target_data)} rows")

# data analysis and visualizations
print("\n" + "="*50)
print("DATA ANALYSIS AND VISUALIZATIONS")
print("="*50)

print("Generating visualizations...")
visualization_tasks = [
    "Target stock price analysis",
    "Multi-stock overview", 
    "Technical indicators",
    "Correlation analysis",
    "Volume and sector analysis"
]

progress_bar = tqdm(total=len(visualization_tasks), desc="Creating plots", unit="plot")

# target stock price analysis (using target_data)
progress_bar.set_description("Creating target stock analysis plot")
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(target_data.index, target_data['Close'], label='Close Price', color='blue', alpha=0.7)
plt.plot(target_data.index, target_data['MA_20'], label='20-day MA', color='red', alpha=0.7)
plt.plot(target_data.index, target_data['MA_50'], label='50-day MA', color='green', alpha=0.7)
plt.title(f'{target_symbol} Stock Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.xticks(rotation=45)

# volume
plt.subplot(2, 2, 2)
plt.plot(target_data.index, target_data['Volume'], color='orange', alpha=0.7)
plt.title(f'{target_symbol} Trading Volume')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.xticks(rotation=45)

# rsi
plt.subplot(2, 2, 3)
plt.plot(target_data.index, target_data['RSI'], color='purple', alpha=0.7)
plt.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
plt.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
plt.title(f'{target_symbol} RSI')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()
plt.xticks(rotation=45)

# volatility
plt.subplot(2, 2, 4)
plt.plot(target_data.index, target_data['Volatility'], color='brown', alpha=0.7)
plt.title(f'{target_symbol} 20-day Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('plots/target_stock_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
progress_bar.update(1)

# multi-stock overview
progress_bar.set_description("Creating multi-stock overview plot")
plt.figure(figsize=(15, 12))

# stock performance comparison (sample of stocks)
plt.subplot(2, 3, 1)
sample_stocks = df_combined['Symbol'].unique()[:10]  # first 10 stocks
colors = plt.cm.tab10(np.linspace(0, 1, len(sample_stocks)))
for i, symbol in enumerate(sample_stocks):
    stock_data = df_combined[df_combined['Symbol'] == symbol]
    normalized_prices = (stock_data['Close'] / stock_data['Close'].iloc[0]) * 100
    plt.plot(stock_data.index, normalized_prices, alpha=0.7, color=colors[i], label=symbol)
plt.title('Normalized Stock Performance (Sample)')
plt.xlabel('Date')
plt.ylabel('Normalized Price (Base=100)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)

# returns distribution across all stocks
plt.subplot(2, 3, 2)
all_returns = df_combined['Returns'].dropna()
plt.hist(all_returns, bins=100, alpha=0.7, color='blue', density=True)
plt.title('Daily Returns Distribution (All Stocks)')
plt.xlabel('Daily Returns')
plt.ylabel('Density')
plt.axvline(all_returns.mean(), color='red', linestyle='--', label=f'Mean: {all_returns.mean():.4f}')
plt.legend()

# volatility by stock
plt.subplot(2, 3, 3)
volatility_by_stock = df_combined.groupby('Symbol')['Volatility'].mean().sort_values(ascending=False)[:15]
plt.barh(range(len(volatility_by_stock)), volatility_by_stock.values, color='orange', alpha=0.7)
plt.yticks(range(len(volatility_by_stock)), volatility_by_stock.index)
plt.title('Average Volatility by Stock (Top 15)')
plt.xlabel('Average Volatility')

# volume distribution
plt.subplot(2, 3, 4)
avg_volume_by_stock = df_combined.groupby('Symbol')['Volume'].mean().sort_values(ascending=False)[:15]
plt.barh(range(len(avg_volume_by_stock)), avg_volume_by_stock.values, color='green', alpha=0.7)
plt.yticks(range(len(avg_volume_by_stock)), avg_volume_by_stock.index)
plt.title('Average Volume by Stock (Top 15)')
plt.xlabel('Average Volume')

# price range comparison
plt.subplot(2, 3, 5)
price_stats = df_combined.groupby('Symbol')['Close'].agg(['min', 'max', 'mean']).sort_values('mean', ascending=False)[:15]
x = range(len(price_stats))
plt.bar(x, price_stats['max'], alpha=0.7, label='Max', color='red')
plt.bar(x, price_stats['min'], alpha=0.7, label='Min', color='blue')
plt.bar(x, price_stats['mean'], alpha=0.7, label='Mean', color='green')
plt.xticks(x, price_stats.index, rotation=45)
plt.title('Price Ranges by Stock (Top 15 by Avg)')
plt.ylabel('Price ($)')
plt.legend()

# correlation between target stock and others
plt.subplot(2, 3, 6)
pivot_closes = df_combined.pivot_table(values='Close', index=df_combined.index, columns='Symbol')
target_correlations = pivot_closes.corrwith(pivot_closes[target_symbol]).dropna().sort_values(ascending=False)
top_corr = target_correlations[1:11]  # exclude self-correlation
plt.barh(range(len(top_corr)), top_corr.values, color='purple', alpha=0.7)
plt.yticks(range(len(top_corr)), top_corr.index)
plt.title(f'Correlation with {target_symbol} (Top 10)')
plt.xlabel('Correlation Coefficient')

plt.tight_layout()
plt.savefig('plots/multi_stock_overview.png', dpi=300, bbox_inches='tight')
plt.show()
progress_bar.update(1)

# technical indicators
progress_bar.set_description("Creating technical indicators plot")
plt.figure(figsize=(15, 8))

plt.subplot(2, 1, 1)
plt.plot(target_data.index, target_data['Close'], label='Close Price', color='blue', alpha=0.8)
plt.plot(target_data.index, target_data['MA_20'], label='20-day MA', color='red', alpha=0.8)
plt.plot(target_data.index, target_data['MA_50'], label='50-day MA', color='green', alpha=0.8)
plt.title(f'{target_symbol} Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.xticks(rotation=45)

plt.subplot(2, 1, 2)
plt.plot(target_data.index, target_data['RSI'], color='purple', alpha=0.8)
plt.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
plt.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
plt.axhline(y=50, color='gray', linestyle='-', alpha=0.3, label='Neutral (50)')
plt.title(f'{target_symbol} RSI Indicator')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('plots/technical_indicators.png', dpi=300, bbox_inches='tight')
plt.show()
progress_bar.update(1)

# correlation analysis (using sample of stocks to avoid clutter)
progress_bar.set_description("Creating correlation analysis")
sample_for_corr = sample_stocks[:8]  # use 8 stocks for correlation
numeric_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_20', 'MA_50', 'RSI', 'Volatility']

plt.figure(figsize=(12, 10))
# create correlation matrix for target stock
target_corr = target_data[numeric_features].corr()
mask = np.triu(np.ones_like(target_corr, dtype=bool))
sns.heatmap(target_corr, annot=True, cmap='coolwarm', center=0, 
           square=True, mask=mask, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title(f'{target_symbol} Feature Correlation Matrix', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
progress_bar.update(1)

# volume and sector analysis
progress_bar.set_description("Creating volume analysis")
plt.figure(figsize=(15, 8))

plt.subplot(2, 1, 1)
plt.plot(target_data.index, target_data['Volume'], color='orange', alpha=0.7)
plt.title(f'{target_symbol} Trading Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.xticks(rotation=45)

plt.subplot(2, 1, 2)
plt.scatter(target_data['Volume'], target_data['Close'], alpha=0.5, color='blue')
plt.title(f'{target_symbol} Volume vs Close Price')
plt.xlabel('Volume')
plt.ylabel('Close Price ($)')

plt.tight_layout()
plt.savefig('plots/volume_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
progress_bar.update(1)

progress_bar.close()

# statistical summary
print("\nStatistical Summary of Target Stock Features:")
target_stats = target_data[numeric_features].describe()
print(target_stats)

print(f"\nMulti-Stock Dataset Summary:")
multi_stats = df_combined[numeric_features].describe()
print(multi_stats)

# save statistical summaries
target_stats.to_csv('plots/target_stock_statistics.csv')
multi_stats.to_csv('plots/multi_stock_statistics.csv')
print("All visualizations completed and saved!")

# feature engineering for machine learning
print("\n" + "="*50)
print("FEATURE ENGINEERING")
print("="*50)

feature_engineering_steps = [
    "Creating additional features",
    "Generating lag features", 
    "Preparing data for modeling",
    "Scaling features"
]

fe_progress = tqdm(total=len(feature_engineering_steps), desc="Engineering features", unit="step")

# create additional features for combined dataset
fe_progress.set_description("Creating additional features for multi-stock dataset")
df_combined['High_Low_Pct'] = (df_combined['High'] - df_combined['Low']) / df_combined['Close'] * 100
df_combined['Price_Change'] = df_combined['Close'].pct_change()
df_combined['Volume_MA'] = df_combined['Volume'].rolling(window=20).mean()
df_combined['Volume_Ratio'] = df_combined['Volume'] / df_combined['Volume_MA']
fe_progress.update(1)

# lag features (because the past might predict the future, maybe?)
fe_progress.set_description("Creating lag features for multi-stock dataset")
for lag in [1, 2, 3, 5, 10]:
    df_combined[f'Close_Lag_{lag}'] = df_combined.groupby('Symbol')['Close'].shift(lag)
    df_combined[f'Volume_Lag_{lag}'] = df_combined.groupby('Symbol')['Volume'].shift(lag)
    df_combined[f'Returns_Lag_{lag}'] = df_combined.groupby('Symbol')['Returns'].shift(lag)
fe_progress.update(1)

# prepare data for modeling
fe_progress.set_description("Preparing multi-stock data for modeling")
# drop rows with nan values
df_combined = df_combined.dropna()
print(f"Multi-stock feature engineering complete! Dataset now has {df_combined.shape[1]} features")
print(f"Final multi-stock dataset shape: {df_combined.shape}")
fe_progress.update(1)

# identify features for modeling
feature_columns = ['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'RSI', 
                  'Volatility', 'High_Low_Pct', 'Price_Change', 'Volume_Ratio'] + \
                 [f'Close_Lag_{lag}' for lag in [1, 2, 3, 5, 10]] + \
                 [f'Volume_Lag_{lag}' for lag in [1, 2, 3, 5, 10]] + \
                 [f'Returns_Lag_{lag}' for lag in [1, 2, 3, 5, 10]]

# use combined dataset for training
X = df_combined[feature_columns]
y = df_combined['Close']

print(f"Features for modeling: {len(feature_columns)}")
for i, feature in enumerate(feature_columns[:10], 1):  # show first 10
    print(f"   {i:2d}. {feature}")
if len(feature_columns) > 10:
    print(f"   ... and {len(feature_columns) - 10} more")

# feature scaling
fe_progress.set_description("Scaling features")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
fe_progress.update(1)

fe_progress.close()

# save preprocessing objects
scaler_path = 'scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to {scaler_path}")

# split data for training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# traditional machine learning models
print("\n" + "="*50)
print("TRADITIONAL MACHINE LEARNING MODELS")
print("="*50)

print("Training traditional ML models...")
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}

# train models with progress bar
for name, model in tqdm(models.items(), desc="Training models"):
    print(f"Training {name}...")
    
    # train model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # calculate metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    results[name] = {
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'RMSE': np.sqrt(mse),
        'Predictions': predictions,
        'Model': model
    }
    
    print(f"{name} - R²: {r2:.4f}, RMSE: {np.sqrt(mse):.2f}")

# save the best traditional model
best_model_name = max(results.items(), key=lambda x: x[1]['R2'])[0]
best_model = results[best_model_name]['Model']
joblib.dump(best_model, 'stock_price_model.pkl')
print(f"Best traditional model: {best_model_name} with R² = {results[best_model_name]['R2']:.4f}")

# pytorch lstm model
print("\n" + "="*50)
print("PYTORCH LSTM MODEL")
print("="*50)

# define pytorch lstm model
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=3, dropout=0.2):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        
        return out

# dataset class for pytorch
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# prepare data for lstm
print("Preparing LSTM data...")
# use combined dataset for LSTM training
lstm_data = df_combined['Close'].values
lstm_scaler = MinMaxScaler()
scaled_lstm_data = lstm_scaler.fit_transform(lstm_data.reshape(-1, 1))

# create sequences
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_lstm, y_lstm = create_sequences(scaled_lstm_data)

# split data for lstm
split_idx = int(len(X_lstm) * 0.8)
X_lstm_train, X_lstm_test = X_lstm[:split_idx], X_lstm[split_idx:]
y_lstm_train, y_lstm_test = y_lstm[:split_idx], y_lstm[split_idx:]

# reshape for lstm
X_lstm_train = X_lstm_train.reshape((X_lstm_train.shape[0], X_lstm_train.shape[1], 1))
X_lstm_test = X_lstm_test.reshape((X_lstm_test.shape[0], X_lstm_test.shape[1], 1))

# create datasets and dataloaders
train_dataset = StockDataset(X_lstm_train, y_lstm_train)
test_dataset = StockDataset(X_lstm_test, y_lstm_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lstm_model = StockLSTM(input_size=1, hidden_size=50, num_layers=3, dropout=0.2)
lstm_model.to(device)

print(f"Training LSTM on {device}")
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

# training loop
lstm_model.train()
train_losses = []
epochs = 50

print("Training LSTM (this might take a while)...")
for epoch in tqdm(range(epochs), desc="Training LSTM"):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = lstm_model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"   Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

# make predictions
lstm_model.eval()
lstm_predictions = []
y_lstm_actual = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = lstm_model(batch_X)
        lstm_predictions.extend(outputs.squeeze().cpu().numpy())
        y_lstm_actual.extend(batch_y.cpu().numpy())

lstm_predictions = np.array(lstm_predictions)
y_lstm_actual = np.array(y_lstm_actual)

# inverse transform
lstm_predictions = lstm_scaler.inverse_transform(lstm_predictions.reshape(-1, 1)).flatten()
y_lstm_actual = lstm_scaler.inverse_transform(y_lstm_actual.reshape(-1, 1)).flatten()

# calculate lstm metrics
lstm_mse = mean_squared_error(y_lstm_actual, lstm_predictions)
lstm_mae = mean_absolute_error(y_lstm_actual, lstm_predictions)
lstm_r2 = r2_score(y_lstm_actual, lstm_predictions)

results['LSTM'] = {
    'MSE': lstm_mse,
    'MAE': lstm_mae,
    'R2': lstm_r2,
    'RMSE': np.sqrt(lstm_mse),
    'Predictions': lstm_predictions,
    'Model': lstm_model
}

print(f"LSTM - R²: {lstm_r2:.4f}, RMSE: {np.sqrt(lstm_mse):.2f}")

# analyze LSTM performance
if lstm_r2 < 0:
    print("LSTM Performance Analysis:")
    print(f"   • Negative R² ({lstm_r2:.4f}) indicates model performs worse than baseline")
    print(f"   • This is common with multi-stock data due to different price ranges")
    print(f"   • Traditional ML models may perform better on this combined dataset")
    print(f"   • Consider: separate models per stock or advanced normalization techniques")
else:
    print("LSTM shows positive predictive power")

# save lstm model
torch.save(lstm_model.state_dict(), 'lstm_stock_model.pth')
print("LSTM model saved as 'lstm_stock_model.pth'")

# model comparison and visualization
print("\n" + "="*50)
print("MODEL COMPARISON AND VISUALIZATION")
print("="*50)

print("Creating model comparison plots...")
comparison_tasks = [
    "Model performance comparison",
    "Individual model predictions",
    "Prediction accuracy analysis"
]

comp_progress = tqdm(total=len(comparison_tasks), desc="Creating comparison plots", unit="plot")

# model performance comparison
comp_progress.set_description("Creating performance comparison")
plt.figure(figsize=(15, 12))

# r² comparison
plt.subplot(2, 3, 1)
models_list = list(results.keys())
r2_scores = [results[model]['R2'] for model in models_list]
colors = plt.cm.Set3(np.linspace(0, 1, len(models_list)))
bars = plt.bar(models_list, r2_scores, color=colors)
plt.title('Model R² Comparison')
plt.ylabel('R² Score')
plt.xticks(rotation=45)

# add value labels on bars
for bar, score in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{score:.3f}', ha='center', va='bottom')

# rmse comparison
plt.subplot(2, 3, 2)
rmse_scores = [results[model]['RMSE'] for model in models_list]
bars = plt.bar(models_list, rmse_scores, color=colors)
plt.title('Model RMSE Comparison')
plt.ylabel('RMSE')
plt.xticks(rotation=45)

for bar, score in zip(bars, rmse_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{score:.2f}', ha='center', va='bottom')

# mae comparison
plt.subplot(2, 3, 3)
mae_scores = [results[model]['MAE'] for model in models_list]
bars = plt.bar(models_list, mae_scores, color=colors)
plt.title('Model MAE Comparison')
plt.ylabel('MAE')
plt.xticks(rotation=45)

for bar, score in zip(bars, mae_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{score:.2f}', ha='center', va='bottom')

# prediction vs actual for best model
best_overall = max(results.items(), key=lambda x: x[1]['R2'])
best_name = best_overall[0]

plt.subplot(2, 3, 4)
if best_name == 'LSTM':
    actual = y_lstm_actual
    predicted = lstm_predictions
else:
    actual = y_test.values
    predicted = results[best_name]['Predictions']

plt.scatter(actual, predicted, alpha=0.6, color='blue')
plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'{best_name} - Actual vs Predicted')

# time series plot for best model
plt.subplot(2, 3, 5)
plt.plot(actual, label='Actual', color='blue', alpha=0.7)
plt.plot(predicted, label='Predicted', color='red', alpha=0.7)
plt.title(f'{best_name} - Time Series Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()

# error distribution
plt.subplot(2, 3, 6)
errors = actual - predicted
plt.hist(errors, bins=30, alpha=0.7, color='green')
plt.title(f'{best_name} - Prediction Errors')
plt.xlabel('Error')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
comp_progress.update(1)

# individual model predictions
comp_progress.set_description("Creating individual predictions")
plt.figure(figsize=(15, 10))

subplot_idx = 1
for model_name in models_list:
    if subplot_idx > 6:
        break
        
    plt.subplot(2, 3, subplot_idx)
    
    if model_name == 'LSTM':
        actual = y_lstm_actual
        predicted = lstm_predictions
    else:
        actual = y_test.values
        predicted = results[model_name]['Predictions']
    
    plt.plot(actual, label='Actual', color='blue', alpha=0.7)
    plt.plot(predicted, label='Predicted', color='red', alpha=0.7)
    plt.title(f'{model_name} Predictions')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    
    subplot_idx += 1

plt.tight_layout()
plt.savefig('plots/individual_predictions.png', dpi=300, bbox_inches='tight')
plt.show()
comp_progress.update(1)

# prediction accuracy analysis
comp_progress.set_description("Creating accuracy analysis")
plt.figure(figsize=(15, 8))

plt.subplot(1, 2, 1)
model_names = list(results.keys())
accuracies = [results[model]['R2'] for model in model_names]
colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))

# For pie chart, we need to handle negative R² scores
# Use relative performance instead of absolute R² values
min_r2 = min(accuracies)
adjusted_accuracies = [r2 - min_r2 + 0.01 for r2 in accuracies]  # shift to positive values
wedges, texts, autotexts = plt.pie(adjusted_accuracies, labels=model_names, autopct='%1.1f%%', 
                                  colors=colors, startangle=90)
plt.title('Model Performance Distribution (Relative)')

plt.subplot(1, 2, 2)
x_pos = np.arange(len(model_names))
bars = plt.bar(x_pos, accuracies, color=colors)
plt.xlabel('Models')
plt.ylabel('R² Score')
plt.title('Model R² Scores')
plt.xticks(x_pos, model_names, rotation=45)

# add value labels
for i, v in enumerate(accuracies):
    if v >= 0:
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    else:
        plt.text(i, v - 0.02, f'{v:.3f}', ha='center', va='top')

plt.tight_layout()
plt.savefig('plots/accuracy_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
comp_progress.update(1)

comp_progress.close()

# summary report
print("\n" + "="*50)
print("SUMMARY REPORT")
print("="*50)

# calculate summary statistics
total_return = ((target_data['Close'][-1] / target_data['Close'][0]) - 1) * 100
volatility = target_data['Returns'].std() * np.sqrt(252) * 100  # annualized volatility
sharpe_ratio = (target_data['Returns'].mean() * 252) / (target_data['Returns'].std() * np.sqrt(252))

print("Creating summary report...")
plt.figure(figsize=(15, 10))

# stock performance
plt.subplot(2, 2, 1)
plt.plot(target_data.index, target_data['Close'], color='blue', linewidth=2)
plt.title(f'{target_symbol} Price Performance')
plt.xlabel('Date')
plt.ylabel('Price ($)')

# returns distribution
plt.subplot(2, 2, 2)
plt.hist(target_data['Returns'].dropna(), bins=50, alpha=0.7, color='green')
plt.axvline(target_data['Returns'].mean(), color='red', linestyle='--', 
           label=f'Mean: {target_data["Returns"].mean():.4f}')
plt.title('Daily Returns Distribution')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.legend()

# model performance
plt.subplot(2, 2, 3)
model_names = list(results.keys())
r2_scores = [results[model]['R2'] for model in model_names]
colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
bars = plt.bar(model_names, r2_scores, color=colors)
plt.title('Model R² Comparison')
plt.ylabel('R² Score')
plt.xticks(rotation=45)

# best model predictions
plt.subplot(2, 2, 4)
if best_name == 'LSTM':
    actual = y_lstm_actual
    predicted = lstm_predictions
else:
    actual = y_test.values
    predicted = results[best_name]['Predictions']

plt.scatter(actual, predicted, alpha=0.6, color='purple')
plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Best Model: {best_name}')

# add text summary
plt.figtext(0.02, 0.02, 
           f'Summary Statistics:\n'
           f'Target Stock: {target_symbol}\n'
           f'Total Return: {total_return:.2f}%\n'
           f'Annualized Volatility: {volatility:.2f}%\n'
           f'Sharpe Ratio: {sharpe_ratio:.2f}\n'
           f'Best Model: {best_name} (R² = {best_overall[1]["R2"]:.4f})\n'
           f'Training Data: {len(df_combined)} samples from {len(df_combined["Symbol"].unique())} stocks\n'
           f'Target Data Period: {target_data.index[0].date()} to {target_data.index[-1].date()}\n'
           f'Target Data Points: {len(target_data)}',
           fontsize=10, ha='left', va='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('plots/summary_report.png', dpi=300, bbox_inches='tight')
plt.show()

# save results to csv
results_df = pd.DataFrame({
    'Model': model_names,
    'R2_Score': [results[model]['R2'] for model in model_names],
    'RMSE': [results[model]['RMSE'] for model in model_names],
    'MAE': [results[model]['MAE'] for model in model_names],
    'MSE': [results[model]['MSE'] for model in model_names]
})

results_df.to_csv('plots/model_results.csv', index=False)
print("Model results saved to 'plots/model_results.csv'")

# final summary
print("\n" + "="*60)
print("STOCK PRICE PREDICTION ANALYSIS COMPLETE!")
print("="*60)
print(f"Target Stock: {target_symbol}")
print(f"Training Data: {len(df_combined)} samples from {len(df_combined['Symbol'].unique())} stocks")
print(f"Target Data Points: {len(target_data)}")
print(f"Target Date Range: {target_data.index[0].date()} to {target_data.index[-1].date()}")
print(f"Target Price Range: ${target_data['Close'].min():.2f} - ${target_data['Close'].max():.2f}")
print(f"Target Total Return: {total_return:.2f}%")
print(f"Target Volatility: {volatility:.2f}%")
print(f"Target Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Best Model: {best_name} (R² = {best_overall[1]['R2']:.4f})")
print(f"All plots saved in 'plots/' directory")
print(f"Models saved as 'stock_price_model.pkl' and 'lstm_stock_model.pth'")

# multi-stock training insights
print(f"\nMulti-Stock Training Insights:")
print(f"   • Trained on {len(df_combined['Symbol'].unique())} different stocks")
print(f"   • Combined dataset: {len(df_combined):,} samples")
print(f"   • Traditional ML models often perform better on multi-stock data")
print(f"   • LSTM may struggle with diverse price ranges across stocks")
print(f"   • Consider sector-specific models for better performance")

print("\nIMPORTANT DISCLAIMERS:")
print("• This is for educational purposes only!")
print("• Past performance does not guarantee future results!")
print("• Don't use this for actual investment decisions!")
print("• The stock market is inherently unpredictable!")
print("• Seriously, I'm not responsible if you lose money!")
print("="*60)
print("Thanks for using the Stock Price Prediction System!")
print("Remember: Learning > Earning!, wow, so wise right?")
