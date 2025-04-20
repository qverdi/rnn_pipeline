import pandas as pd
import yfinance as yf
from datetime import datetime
from src.config.file_constants import DATA_DIR

def main():
    # Load existing data
    file_path = DATA_DIR / "data.csv"
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])

    # Today's date (no time)
    today = pd.to_datetime(datetime.today().date())

    # Check if already exists
    if today in df['Date'].values:
        print(f"Data for {today.date()} already exists.")
    else:
        # Fetch actual S&P 500 Index value from Yahoo Finance
        ticker = yf.Ticker("^GSPC")
        hist = ticker.history(period="1d", interval="1d")

        if not hist.empty:
            close_price = hist['Close'].iloc[0]
            new_row = pd.DataFrame([{
                'Date': today,
                'Price': round(close_price, 2)
            }])

            df = pd.concat([df, new_row], ignore_index=True)
            df.sort_values(by='Date', inplace=True)
            df.to_csv(file_path, index=False)
            print(f"Appended {today.date()} with actual S&P 500 close: {round(close_price, 2)}")
        else:
            print("Failed to fetch S&P 500 data.")


if __name__ == "__main__":
    main()
