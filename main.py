from src.portfolio_pipeline import full_portfolio_pipeline

def main():
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
    start = "2020-01-01"
    end = "2025-01-01"

    results = full_portfolio_pipeline(tickers, start, end)
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
