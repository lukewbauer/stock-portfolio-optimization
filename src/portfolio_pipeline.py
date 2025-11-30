def full_portfolio_pipeline(
    tickers,
    start,
    end,
    return_floor=0.015,
    output_dir="my_output"
):
    # ============================
    # IMPORTS
    # ============================
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.optimize import minimize
    import yfinance as yf

    # --- NEW (minimal): safe display support ---
    try:
        from IPython.display import display as _ip_display
        def _display_df(df, n=5, title=None):
            if title:
                print(title)
            _ip_display(df.head(n))
    except Exception:
        def _display_df(df, n=5, title=None):
            if title:
                print(title)
            # nice plain-text preview if not in a notebook
            print(df.head(n).to_string())
    # ------------------------------------------

    # ============================
    # CREATE OUTPUT FOLDER
    # ============================
    os.makedirs(output_dir, exist_ok=True)

    # ============================
    # 1. DOWNLOAD PRICE DATA
    # ============================
    print("Downloading price data...\n")
    prices = {}
    for t in tickers:
        try:
            df = yf.download(
                t, start=start, end=end,
                interval="1d", auto_adjust=False, progress=False
            )
            if not df.empty:
                prices[t] = df
            else:
                print(f"⚠️ No data for {t}")
        except Exception as e:
            print(f"❌ Failed {t}: {e}")

    if not prices:
        raise ValueError("No data downloaded.")

    print("✓ Price data downloaded.\n")

    # ============================
    # 2. BUILD ADJ CLOSE DATAFRAME
    # ============================
    first = tickers[0]
    prep_data = pd.DataFrame(
        prices[first]['Adj Close']).rename(columns={'Adj Close': first})

    for t in tickers[1:]:
        prep_data[t] = prices[t]['Adj Close']

    _display_df(prep_data, title="=== Adjusted Close Prices ===")

    # Save price data to CSV
    prep_data.to_csv(
        os.path.join(output_dir, "price_data.csv"),
        index=True
    )

    # ============================
    # 3. DAILY RETURNS
    # ============================
    return_data = prep_data.pct_change().dropna()

    # Save daily returns to CSV
    return_data.to_csv(
        os.path.join(output_dir, "daily_returns.csv"),
        index=True
    )

    # ============================
    # 4. CUMULATIVE RETURNS PLOT
    # ============================
    cumulative_returns = (1 + return_data).cumprod() - 1
    ax = cumulative_returns.plot(figsize=(12, 6))
    ax.set_title("Cumulative Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True)

    fig = ax.get_figure()
    fig.savefig(
        os.path.join(output_dir, "cumulative_returns.jpg"),
        format="jpg",
        dpi=300
    )
    plt.close(fig)

    # ============================
    # 5. MONTHLY RETURNS
    # ============================
    # CHANGED: 'ME' -> 'M' (standard month-end alias; avoids errors on some pandas versions)
    monthly_returns = prep_data.resample('M').ffill().pct_change().dropna()
    _display_df(monthly_returns, title="=== Monthly Returns ===")

    # Save monthly returns to CSV
    monthly_returns.to_csv(
        os.path.join(output_dir, "monthly_returns.csv"),
        index=True
    )

    # ============================
    # 6. HEATMAPS: COV & CORR
    # ============================
    plt.figure(figsize=(10, 8))
    sns.heatmap(monthly_returns.cov(), annot=True, cmap='coolwarm')
    plt.title("Covariance Matrix (Monthly Returns)")
    cov_path = os.path.join(output_dir, "covariance_matrix.jpg")
    plt.savefig(cov_path, format="jpg", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(monthly_returns.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix (Monthly Returns)")
    corr_path = os.path.join(output_dir, "correlation_matrix.jpg")
    plt.savefig(corr_path, format="jpg", dpi=300)
    plt.close()

    # ============================
    # 7. PORTFOLIO OPTIMIZATION
    # ============================
    df = monthly_returns.copy()
    df_return = df.mean()
    df_cov = df.cov()

    # math functions
    def port_return(w):
        return np.dot(w, df_return.values)

    def port_risk(w):
        return np.dot(w, df_cov.values @ w)

    # optimize for each risk level
    def optimize_for_risk_limit(risk_limit):
        n = len(df.columns)
        x0 = np.ones(n) / n
        bounds = [(0, 1)] * n

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "ineq", "fun": lambda w: risk_limit - port_risk(w)},
            {"type": "ineq", "fun": lambda w: port_return(w) - return_floor},
        ]

        res = minimize(
            fun=lambda w: -port_return(w),
            x0=x0,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP'
        )

        w = res.x
        return w, port_return(w), port_risk(w)

    # sweep risk levels
    risk_levels = np.arange(0.001, 0.01, 0.0005)

    allocations, rets, risks = [], [], []
    for r in risk_levels:
        w, ret, risk_val = optimize_for_risk_limit(r)
        allocations.append(w)
        rets.append(ret)
        risks.append(risk_val)

    alloc_df = pd.DataFrame(allocations, index=risk_levels, columns=df.columns)
    reward_df = pd.DataFrame({
        "risk_limit": risk_levels,
        "risk": risks,
        "return": rets
    })
    reward_df["Δ return"] = reward_df["return"].diff()
    reward_df["% Δ return"] = reward_df["return"].pct_change()

    # Save optimization tables to CSV
    alloc_df.to_csv(
        os.path.join(output_dir, "allocations_by_risk_limit.csv"),
        index=True
    )
    reward_df.to_csv(
        os.path.join(output_dir, "reward_by_risk_limit.csv"),
        index=False
    )

    # ============================
    # PLOTS (OPTIMIZATION)
    # ============================
    # 1) Weights across risk limits
    ax = alloc_df.plot(figsize=(12, 6))
    ax.set_title("Portfolio Weights Across Risk Limits")
    ax.set_xlabel("Risk Limit")
    ax.set_ylabel("Weight")
    fig = ax.get_figure()
    fig.savefig(
        os.path.join(output_dir, "weights_across_risk_limits.jpg"),
        format="jpg",
        dpi=300
    )
    plt.close(fig)

    # 2) Efficient Frontier
    plt.figure(figsize=(8, 6))
    plt.scatter(reward_df["risk"], reward_df["return"])
    plt.title("Efficient Frontier")
    plt.xlabel("Risk")
    plt.ylabel("Return")
    plt.savefig(
        os.path.join(output_dir, "efficient_frontier.jpg"),
        format="jpg",
        dpi=300
    )
    plt.close()

    # 3) Δ Return vs Risk
    plt.figure(figsize=(8, 6))
    plt.scatter(reward_df["risk"], reward_df["Δ return"])
    plt.title("Change in Return as Risk Increases")
    plt.xlabel("Risk")
    plt.ylabel("Δ Return")
    plt.savefig(
        os.path.join(output_dir, "delta_return_vs_risk.jpg"),
        format="jpg",
        dpi=300
    )
    plt.close()

    # 4) % Δ Return vs Risk
    plt.figure(figsize=(8, 6))
    plt.scatter(reward_df["risk"], reward_df["% Δ return"])
    plt.title("Percent Change in Return as Risk Increases")
    plt.xlabel("Risk")
    plt.ylabel("% Δ Return")
    plt.savefig(
        os.path.join(output_dir, "pct_delta_return_vs_risk.jpg"),
        format="jpg",
        dpi=300
    )
    plt.close()

    # ============================
    # FINAL OUTPUT
    # ============================
    _display_df(alloc_df, title="\n=== First 5 Allocation Rows ===")
    _display_df(reward_df, title="\n=== First 5 Reward Rows ===")

    print(f"\nAll plots (JPG) and CSV files saved in: {output_dir}/")

    return {
        "price_data": prep_data,
        "daily_returns": return_data,
        "monthly_returns": monthly_returns,
        "alloc_df": alloc_df,
        "reward_df": reward_df,
    }


# ============================
# RUN PIPELINE
# ============================
results = full_portfolio_pipeline(
    tickers=['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL'],
    start='2020-01-01',
    end='2025-01-01',
    output_dir="my_output"  # <- folder where everything is written
)
