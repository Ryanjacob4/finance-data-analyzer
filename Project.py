import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def get_stock_data(ticker, period="1y"):
    stock = yf.download(ticker, period=period, auto_adjust=True)
    return stock

def calculate_metrics(df, ticker):
    close = df["Close"][ticker]
    df["MA20"] = close.rolling(window=20).mean()
    df["MA50"] = close.rolling(window=50).mean()
    df["Daily Return"] = close.pct_change()
    df["Volatility"] = df["Daily Return"].rolling(window=20).std() * (252 ** 0.5)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def print_summary(df, ticker):
    close = df["Close"][ticker]
    print(f"\n--- {ticker} Summary ---")
    print(f"Current Price:   ${close.iloc[-1]:.2f}")
    print(f"1Y High:         ${close.max():.2f}")
    print(f"1Y Low:          ${close.min():.2f}")
    total_return = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100
    print(f"1Y Return:       {total_return:.2f}%")
    avg_volatility = df["Volatility"].mean()
    print(f"Avg Volatility:  {avg_volatility:.2%}")
    current_rsi = df["RSI"].iloc[-1]
    print(f"Current RSI:     {current_rsi:.2f}")
    if current_rsi > 70:
        print("RSI Signal:      Overbought")
    elif current_rsi < 30:
        print("RSI Signal:      Oversold")
    else:
        print("RSI Signal:      Neutral")

def plot_stock(df, ticker):
    close = df["Close"][ticker]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(f"{ticker} Stock Analysis", fontsize=16, fontweight="bold")

    # Price + Moving Averages
    ax1.plot(close.index, close, label="Close Price", color="blue", linewidth=1.5)
    ax1.plot(df.index, df["MA20"], label="20-Day MA", color="orange", linewidth=1)
    ax1.plot(df.index, df["MA50"], label="50-Day MA", color="red", linewidth=1)
    ax1.set_ylabel("Price (USD)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Volatility
    ax2.plot(df.index, df["Volatility"], label="Annualized Volatility", color="purple", linewidth=1)
    ax2.set_ylabel("Volatility")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # RSI
    ax3.plot(df.index, df["RSI"], label="RSI", color="green", linewidth=1)
    ax3.axhline(70, color="red", linestyle="--", alpha=0.5, label="Overbought (70)")
    ax3.axhline(30, color="blue", linestyle="--", alpha=0.5, label="Oversold (30)")
    ax3.set_ylabel("RSI")
    ax3.set_xlabel("Date")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout()
    plt.savefig(f"{ticker}_analysis.png", dpi=150)
    plt.show()
    print(f"\nChart saved as {ticker}_analysis.png")

    df.to_csv(f"{ticker}_data.csv")
    print(f"Data exported to {ticker}_data.csv")

def compare_stocks(tickers, period="1y"):
    plt.figure(figsize=(14, 6))
    plt.title("Stock Price Comparison (Normalized)", fontsize=14, fontweight="bold")

    for ticker in tickers:
        df = yf.download(ticker, period=period, auto_adjust=True)
        close = df["Close"][ticker]
        normalized = (close / close.iloc[0]) * 100
        plt.plot(close.index, normalized, label=ticker, linewidth=1.5)

    plt.ylabel("Normalized Price (Base 100)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150)
    plt.show()
    print("\nComparison chart saved as comparison.png")

# --- Main ---
print("=== Finance Data Analyzer ===")
print("1. Analyze a single stock")
print("2. Compare multiple stocks")
choice = input("\nEnter choice (1 or 2): ")

if choice == "1":
    ticker = input("Enter ticker (e.g. AAPL): ").upper()
    print(f"\nFetching {ticker} data...")
    df = get_stock_data(ticker)
    df = calculate_metrics(df, ticker)
    print_summary(df, ticker)
    plot_stock(df, ticker)

elif choice == "2":
    raw = input("Enter tickers separated by spaces (e.g. AAPL MSFT GOOGL): ").upper()
    tickers = raw.split()
    print(f"\nFetching data for {tickers}...")
    compare_stocks(tickers)