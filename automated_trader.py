import os
import pandas as pd
import pandas_ta as ta
import requests
from dotenv import load_dotenv
import time
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

# --- CONFIGURATION & SETUP ---
load_dotenv()
twelvedata_api_key = os.getenv('TWELVEDATA_API_KEY')
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMZN', 'META','VOO']
DATASET_FILE = 'stock_features.csv'
LOG_FILE = 'trading_log.txt'

def log_message(message):
    """Writes a message to the log file and prints it."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    with open(LOG_FILE, 'a') as f:
        f.write(full_message + '\n')

# --- PART 1: DATASET CREATION FUNCTIONS ---

def get_historical_data(ticker, days=1000):
    """Fetches a long history of price data for a single ticker."""
    log_message(f"Fetching {days} days of historical data for {ticker}...")
    # ... (Code is the same as in create_dataset.py)
    try:
        url = f"https://api.twelvedata.com/time_series?symbol={ticker}&interval=1day&outputsize={days}&apikey={twelvedata_api_key}"
        response = requests.get(url).json()
        if response.get('status') != 'ok': return None
        df = pd.DataFrame(response['values']).iloc[::-1].reset_index(drop=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
        return df
    except Exception: return None

def create_features_and_target(df, future_days=5):
    """Engineers features and the target variable."""
    log_message("  Engineering features and target...")
    # ... (Code is the same as in create_dataset.py, combined for efficiency)
    df.ta.sma(length=20, append=True); df.ta.sma(length=50, append=True)
    df.ta.rsi(length=14, append=True); df.ta.macd(append=True)
    df.ta.bbands(length=20, append=True); df.ta.atr(length=14, append=True)
    for period in [1, 3, 5, 10, 21]: df[f'price_change_{period}d'] = df['close'].pct_change(periods=period)
    df['volatility_21d'] = df['close'].pct_change().rolling(window=21).std()
    df['future_price'] = df['close'].shift(-future_days)
    df['target'] = (df['future_price'] > df['close']).astype(int)
    return df

def create_master_dataset():
    """Builds and saves the full historical dataset with features."""
    log_message("--- Starting Historical Dataset Creation/Update ---")
    all_tickers_df = pd.DataFrame()
    for ticker in TICKERS:
        time.sleep(1) 
        hist_data = get_historical_data(ticker)
        if hist_data is not None:
            final_df = create_features_and_target(hist_data)
            final_df['ticker'] = ticker
            all_tickers_df = pd.concat([all_tickers_df, final_df], ignore_index=True)
    
    all_tickers_df.dropna(inplace=True)
    all_tickers_df.drop(columns=['future_price'], inplace=True)
    all_tickers_df.to_csv(DATASET_FILE, index=False)
    log_message(f"Dataset update complete! Saved {len(all_tickers_df)} rows to {DATASET_FILE}")

# --- PART 2: MODEL TRAINING & PREDICTION FUNCTIONS ---

def train_and_predict():
    """Loads data, trains model, evaluates, and makes final predictions."""
    log_message("--- Starting Model Training & Prediction ---")
    try:
        df = pd.read_csv(DATASET_FILE)
    except FileNotFoundError:
        log_message(f"Error: Dataset file not found. Run dataset creation first.")
        return

    # --- Train Model ---
    features = [col for col in df.columns if col not in ['ticker', 'datetime', 'target']]
    X = df[features]; y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)

    # --- Evaluate Model ---
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    log_message(f"Model Evaluation Accuracy on unseen data: {accuracy * 100:.2f}%")

    # --- Make Live Predictions ---
    latest_data = df.groupby('ticker').last().reset_index()
    X_latest = latest_data[features]
    probabilities = model.predict_proba(X_latest)[:, 1]
    
    results_df = pd.DataFrame({'ticker': latest_data['ticker'], 'prediction_confidence': probabilities})
    results_df['signal'] = results_df['prediction_confidence'].apply(lambda p: 'UP' if p > 0.5 else 'DOWN')
    results_df = results_df.sort_values(by='prediction_confidence', ascending=False).reset_index(drop=True)
    
    log_message("--- Model Predictions for the Next 5 Trading Days ---")
    log_message("\n" + results_df.to_string())

# --- MAIN ORCHESTRATOR ---
def main():
    """Main function to run the entire automated pipeline."""
    log_message("Starting the autonomous trading agent workflow.")
    create_master_dataset()
    train_and_predict()
    log_message("Autonomous trading agent workflow finished.")

if __name__ == "__main__":
    main()