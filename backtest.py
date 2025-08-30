import pandas as pd
from backtesting import Backtest, Strategy
import xgboost as xgb

# --- CONFIGURATION ---
DATASET_FILE = 'stock_features.csv'
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMZN', 'META']

# --- 1. Define the More Realistic "Walk-Forward" Trading Strategy ---
class MLWalkForwardStrategy(Strategy):
    # --- Strategy Parameters ---
    training_window = 500
    retrain_every = 20
    buy_threshold = 0.55
    sell_threshold = 0.45

    def init(self):
        self.model = None
        self.last_retrain_day = -self.retrain_every

    def next(self):
        i = len(self.data) - 1

        if i >= self.last_retrain_day + self.retrain_every and i >= self.training_window:
            self.last_retrain_day = i
            past_data = self.data.df.iloc[i - self.training_window : i]
            
            features = [col for col in past_data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'ticker', 'datetime', 'target']]
            X_train = past_data[features]
            y_train = past_data['target']

            print(f"  Retraining model on day {i}...")
            # FIX: Removed the unnecessary 'use_label_encoder' parameter to clean up warnings.
            self.model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_estimators=100)
            self.model.fit(X_train, y_train)

        if self.model:
            features = [col for col in self.data.df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'ticker', 'datetime', 'target']]
            current_features = self.data.df[features].iloc[[i]]
            
            todays_confidence = self.model.predict_proba(current_features)[:, 1][0]
            
            if todays_confidence > self.buy_threshold and not self.position:
                self.buy()
            elif todays_confidence < self.sell_threshold and self.position:
                self.position.close()

# --- 2. Main Backtesting Orchestrator ---
def run_backtest():
    print("--- Starting REALISTIC Walk-Forward Backtesting Simulation ---")
    try:
        full_df = pd.read_csv(DATASET_FILE, parse_dates=['datetime'])
        full_df.fillna(0, inplace=True)
    except FileNotFoundError:
        print(f"Error: Dataset file '{DATASET_FILE}' not found.")
        print("Please run 'python automated_trader.py' first to generate the data.")
        return

    for ticker in TICKERS:
        print(f"\n--- Backtesting for {ticker} ---")
        ticker_data = full_df[full_df['ticker'] == ticker].copy()
        ticker_data.set_index('datetime', inplace=True)
        ticker_data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)

        bt = Backtest(ticker_data, MLWalkForwardStrategy, cash=10000, commission=.002)
        stats = bt.run()
        print(stats)

if __name__ == "__main__":
    run_backtest()

