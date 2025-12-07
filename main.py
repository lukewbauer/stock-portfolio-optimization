from src.portfolio_pipeline import full_portfolio_pipeline, bonmin_portfolio_pipeline


def main():
    # ----------------------------------------
    # Common configuration
    # ----------------------------------------
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

    # Example sector mapping (user can change this freely)
    sector_map = {
        "AAPL": "Tech",
        "MSFT": "Tech",
        "NVDA": "Tech",
        "AMZN": "Consumer",
        "GOOGL": "Tech",
    }

    # ----------------------------------------
    # PART 1 – Classical Markowitz pipeline
    # ----------------------------------------
    print("=== Running Part 1: Markowitz / SciPy pipeline ===")
    part1_start = "2020-01-01"
    part1_end = "2025-01-01"

    results_part1 = full_portfolio_pipeline(
        tickers=tickers,
        start=part1_start,
        end=part1_end,
        output_dir="outputs/part1",  # keep all Part 1 artifacts here
    )
    print("Part 1 completed successfully!\n")

    # ----------------------------------------
    # PART 2 – Bonmin MINLP + logic + paper trading
    # ----------------------------------------
    print("=== Running Part 2: Bonmin MINLP pipeline ===")

    # Training period (fit the model)
    train_start = "2020-01-01"
    train_end = "2024-12-31"

    # Test period (paper trading / backtest)
    test_start = "2025-01-01"
    test_end = "2025-07-01"

    results_part2 = bonmin_portfolio_pipeline(
        tickers=tickers,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        return_floor_low=0.01,      # low-risk scenario return floor
        output_dir="outputs/part2", # keep all Part 2 artifacts here
        min_weight=0.02,            # linking: if chosen, weight >= 2%
        max_weight=0.20,            # linking: if chosen, weight <= 20%
        min_assets=3,               # choose at least 3 stocks overall
        sector_map=sector_map,      # at least sector_min per sector
        sector_min=1,               # choose at least 1 stock per sector
        scenarios=("low", "medium", "high"),
        solver_name="bonmin",       # requires Bonmin installed & on PATH
        solver_options=None,
    )
    print("Part 2 completed successfully!\n")

    print("All pipelines completed. Check the 'outputs/part1' and 'outputs/part2' folders for results.")


if __name__ == "__main__":
    main()
