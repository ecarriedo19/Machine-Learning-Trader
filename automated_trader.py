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
import finnhub

# --- CONFIGURATION & SETUP ---
load_dotenv()
twelvedata_api_key = os.getenv('TWELVEDATA_API_KEY')
finnhub_api_key = os.getenv('FINNHUB_API_KEY')
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMZN', 'META','VOO']
DATASET_FILE = 'stock_features.csv'
LOG_FILE = 'trading_log.txt'
PREDICTIONS_FILE = 'predictions.csv'

finnhub_client = finnhub.Client(api_key=finnhub_api_key)

def log_message(message):
    """Writes a message to the log file and prints it."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(full_message + '\n')

# --- PART 1: DATASET CREATION FUNCTIONS ---

def get_historical_data(ticker, days=1000):
    """Fetches a long history of price data for a single ticker."""
    log_message(f"Fetching {days} days of historical price data for {ticker}...")
    try:
        url = f"https://api.twelvedata.com/time_series?symbol={ticker}&interval=1day&outputsize={days}&apikey={twelvedata_api_key}"
        response = requests.get(url).json()
        if response.get('status') != 'ok': return None
        df = pd.DataFrame(response['values']).iloc[::-1].reset_index(drop=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
        return df
    except Exception: return None

def get_fundamental_data(ticker):
    """Fetches historical fundamental metrics (quarterly)."""
    log_message(f"Fetching historical fundamental data for {ticker}...")
    try:
        res = finnhub_client.financials_reported(symbol=ticker, freq='quarterly')
        # FIX #1: Check if the 'data' key exists and is not empty. This handles ETFs like VOO.
        if not res or not res.get('data'):
            log_message(f"  No fundamental data found for {ticker} (likely an ETF).")
            return pd.DataFrame()

        df = pd.DataFrame(res['data'])
        fundamentals = ['peRatio', 'psRatio', 'pbRatio']
        fund_data = []
        for report in df['report']:
            quarter_data = {f: report.get(f) for f in fundamentals}
            fund_data.append(quarter_data)
        fund_df = pd.DataFrame(fund_data)
        fund_df['datetime'] = pd.to_datetime(df['endDate'])
        return fund_df
    except Exception as e:
        log_message(f"  Could not fetch fundamental data for {ticker}: {e}")
        return pd.DataFrame()

def create_features_and_target(df, future_days=5):
    """Engineers an expanded set of features."""
    log_message("  Engineering technical features and target...")
    df.ta.sma(length=20, append=True); df.ta.sma(length=50, append=True)
    df.ta.rsi(length=14, append=True); df.ta.macd(append=True)
    df.ta.bbands(length=20, append=True); df.ta.atr(length=14, append=True)
    df.ta.adx(length=14, append=True); df.ta.obv(append=True)
    df.ta.stoch(k=14, d=3, append=True)
    for period in [1, 3, 5, 10, 21]: df[f'price_change_{period}d'] = df['close'].pct_change(periods=period)
    df['volatility_21d'] = df['close'].pct_change().rolling(window=21).std()
    df['future_price'] = df['close'].shift(-future_days)
    df['target'] = (df['future_price'] > df['close']).astype(int)
    return df

def create_master_dataset():
    """Builds and saves the full historical dataset with all features."""
    log_message("--- Starting Historical Dataset Creation/Update ---")
    all_tickers_df = pd.DataFrame()
    for ticker in TICKERS:
        time.sleep(1.5)
        price_data = get_historical_data(ticker)
        fund_data = get_fundamental_data(ticker)
        if price_data is not None:
            if not fund_data.empty:
                price_data = pd.merge_asof(price_data.sort_values('datetime'), fund_data.sort_values('datetime'), on='datetime', direction='backward')
            final_df = create_features_and_target(price_data)
            final_df['ticker'] = ticker
            all_tickers_df = pd.concat([all_tickers_df, final_df], ignore_index=True)

    # FIX #2: Use a smarter dropna. Only drop rows where the 'target' is unknown.
    # This preserves all other data, even if some features are missing.
    all_tickers_df.dropna(subset=['target'], inplace=True)
    all_tickers_df.drop(columns=['future_price'], inplace=True)
    
    all_tickers_df.to_csv(DATASET_FILE, index=False)
    log_message(f"Dataset update complete! Saved {len(all_tickers_df)} rows to {DATASET_FILE}")

# --- PART 2: MODEL TRAINING & PREDICTION ---
def train_and_predict():
    """Loads data, trains model, evaluates, and makes final predictions."""
    log_message("--- Starting Model Training & Prediction ---")
    try:
        df = pd.read_csv(DATASET_FILE)
        # Fill any remaining missing values in features with 0, which is a neutral value.
        df.fillna(0, inplace=True)
    except FileNotFoundError:
        log_message(f"Error: Dataset file not found. Run dataset creation first.")
        return

    features = [col for col in df.columns if col not in ['ticker', 'datetime', 'target']]
    X = df[features]; y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    log_message(f"Model Evaluation Accuracy on unseen data: {accuracy * 100:.2f}%")

    latest_data = df.groupby('ticker').last().reset_index()
    X_latest = latest_data[features]
    probabilities = model.predict_proba(X_latest)[:, 1]
    
    results_df = pd.DataFrame({'Ticker': latest_data['ticker'], 'Confidence': probabilities})
    results_df['Forecast'] = results_df['Confidence'].apply(lambda p: 'UP ⬆️' if p > 0.5 else 'DOWN ⬇️')
    results_df['Confidence'] = (results_df['Confidence'] * 100).map('{:.2f}%'.format)
    results_df['prob_sort'] = probabilities
    results_df = results_df.sort_values(by='prob_sort', ascending=False).drop(columns=['prob_sort'])
    
    log_message("--- Model Predictions for the Next 5 Trading Days ---")
    log_message("\n" + results_df.to_string(index=False))

    results_df.to_csv(PREDICTIONS_FILE, index=False)
    log_message(f"Predictions saved to {PREDICTIONS_FILE}")

# --- MAIN ORCHESTRATOR ---
def main():
    """Main function to run the entire automated pipeline."""
    log_message("Starting the autonomous trading agent workflow.")
    create_master_dataset()
    train_and_predict()
    log_message("Autonomous trading agent workflow finished.")

if __name__ == "__main__":
    main()

