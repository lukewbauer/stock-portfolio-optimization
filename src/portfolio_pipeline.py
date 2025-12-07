"""
portfolio_optimization.py

Part 1:
    - full_portfolio_pipeline(...)  # your original Markowitz + scipy version

Part 2:
    - bonmin_portfolio_pipeline(...)  # MINLP with Bonmin + binary variables
"""



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


# Example Part 1 run (you can comment this out in the library file)
if __name__ == "__main__":
    results_part1 = full_portfolio_pipeline(
        tickers=['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL'],
        start='2020-01-01',
        end='2025-01-01',
        output_dir="my_output_part1"
    )

# --------------------------------------------------
# PART 2 – BONMIN + BINARY VARIABLES + PAPER TRADING
# --------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    Constraint,
    NonNegativeReals,
    Binary,
    RangeSet,
    Expression,
    SolverFactory,
    value,
    minimize
)

# Safe-ish display helper for both console & notebooks
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
        print(df.head(n).to_string())


def _download_price_data(tickers, start, end):
    """
    Helper: download Adj Close data for a list of tickers.
    """
    print(f"Downloading price data from {start} to {end} ...\n")
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

    first = tickers[0]
    price_df = pd.DataFrame(prices[first]['Adj Close']).rename(columns={'Adj Close': first})
    for t in tickers[1:]:
        price_df[t] = prices[t]['Adj Close']

    _display_df(price_df, title="=== Adjusted Close Prices (All Periods) ===")
    return price_df


def _solve_bonmin_portfolio(
    mu,
    cov,
    scenario="medium",
    min_assets=3,
    min_weight=0.02,
    max_weight=0.20,
    sector_map=None,
    sector_min=1,
    return_floor=0.0,
    solver_name="bonmin",
    solver_options=None
):
    """
    Core MINLP model using Bonmin.

    mu: pd.Series of expected returns per asset (training period, e.g. monthly mean)
    cov: pd.DataFrame of covariance matrix
    scenario: "low", "medium", "high"
    min_assets: choose at least this many stocks
    min_weight, max_weight: linking constraints:
        if z_i = 1 -> min_weight <= w_i <= max_weight
        if z_i = 0 -> w_i = 0
    sector_map: dict {ticker: group/sector}
    sector_min: at least sector_min stocks in each sector
    return_floor: minimum expected return for low-risk scenario
    """
    tickers = list(mu.index)
    n = len(tickers)

    model = ConcreteModel()
    model.A = RangeSet(0, n - 1)

    # Map indices to tickers
    idx_to_ticker = {i: t for i, t in enumerate(tickers)}

    mu_dict = mu.to_dict()
    cov_df = cov

    # Decision variables
    model.w = Var(model.A, domain=NonNegativeReals)  # weights
    model.z = Var(model.A, domain=Binary)            # activation (invest or not)

    # Sum of weights = 1
    def total_weight_rule(m):
        return sum(m.w[i] for i in m.A) == 1.0
    model.total_weight = Constraint(rule=total_weight_rule)

    # Linking constraints: if invest in stock (z_i=1), must invest
    # between min_weight and max_weight; if z_i=0, weight is forced to 0.
    def linking_min_rule(m, i):
        return m.w[i] >= min_weight * m.z[i]
    model.linking_min = Constraint(model.A, rule=linking_min_rule)

    def linking_max_rule(m, i):
        return m.w[i] <= max_weight * m.z[i]
    model.linking_max = Constraint(model.A, rule=linking_max_rule)

    # Choose at least min_assets stocks
    if min_assets is not None and min_assets > 0:
        def min_assets_rule(m):
            return sum(m.z[i] for i in m.A) >= min_assets
        model.min_assets = Constraint(rule=min_assets_rule)

    # Sector/group constraints: at least `sector_min` stocks per sector/group
    if sector_map is not None:
        sectors = sorted(set(sector_map[t] for t in tickers))
        for s in sectors:
            idxs = [i for i, t in idx_to_ticker.items()
                    if sector_map.get(t) == s]
            if idxs:
                def sector_rule(m, idxs=idxs):
                    return sum(m.z[i] for i in idxs) >= sector_min
                model.add_component(f"sector_min_{s}", Constraint(rule=sector_rule))

    # Portfolio return expression
    def ret_expr_rule(m):
        return sum(mu_dict[idx_to_ticker[i]] * m.w[i] for i in m.A)
    model.port_return = Expression(rule=ret_expr_rule)

    # Portfolio risk (variance) expression
    def risk_expr_rule(m):
        return sum(
            cov_df.loc[idx_to_ticker[i], idx_to_ticker[j]] * m.w[i] * m.w[j]
            for i in m.A for j in m.A
        )
    model.port_risk = Expression(rule=risk_expr_rule)

    # Scenario-specific objective / extra constraints
    if scenario == "low":
        # Low risk: minimize variance subject to a minimum expected return
        model.return_floor_con = Constraint(expr=model.port_return >= return_floor)
        model.obj = Objective(expr=model.port_risk, sense=minimize)

    elif scenario == "medium":
        # Medium risk: trade off return and risk (top-left of frontier)
        # Higher lambda => more risk-averse
        risk_aversion = 1.0
        # Minimize: risk_aversion * risk - return  (equiv. maximize return - risk_aversion * risk)
        model.obj = Objective(
            expr=risk_aversion * model.port_risk - model.port_return,
            sense=minimize
        )

    elif scenario == "high":
        # High risk: "dump all in one stock" – only one active stock,
        # maximize expected return.
        model.one_stock = Constraint(expr=sum(model.z[i] for i in model.A) == 1)
        # Minimize negative return => maximize return
        model.obj = Objective(expr=-model.port_return, sense=minimize)

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # Solve with Bonmin (or whatever MINLP solver name is passed in)
    solver = SolverFactory(solver_name)
    if solver_options:
        for k, v in solver_options.items():
            solver.options[k] = v

    print(f"\nSolving {scenario} risk scenario with {solver_name} ...")
    result = solver.solve(model, tee=False)
    print(result.solver.status, result.solver.termination_condition)

    # Extract solution
    w_opt = np.array([value(model.w[i]) for i in model.A])
    z_opt = np.array([value(model.z[i]) for i in model.A])

    weights = pd.Series(w_opt, index=tickers, name=f"weights_{scenario}")
    active = pd.Series(z_opt, index=tickers, name=f"active_{scenario}")

    port_ret = value(model.port_return)
    port_risk = value(model.port_risk)

    return {
        "scenario": scenario,
        "weights": weights,
        "active": active,
        "port_return": port_ret,
        "port_risk": port_risk,
        "solver_result": result
    }


def bonmin_portfolio_pipeline(
    tickers,
    train_start,
    train_end,
    test_start=None,
    test_end=None,
    return_floor_low=0.01,
    output_dir="my_output_bonmin",
    min_weight=0.02,
    max_weight=0.20,
    min_assets=3,
    sector_map=None,
    sector_min=1,
    scenarios=("low", "medium", "high"),
    solver_name="bonmin",
    solver_options=None
):
    """
    Full Bonmin-based pipeline (Part 2).

    Features:
      - Uses Bonmin MINLP with binary activation variables.
      - Linking constraints: if you invest in a stock, you must invest
        at least min_weight and at most max_weight.
      - Logical constraints: at least min_assets stocks, and optionally
        at least sector_min stocks per user-defined sector/group.
      - Low/medium/high risk scenarios.
      - Optional paper trading / backtesting on an out-of-sample period.

    Args:
        tickers: list of tickers.
        train_start, train_end: str, training period used to fit model
                                (e.g. '2020-01-01', '2024-12-31').
        test_start, test_end: optional out-of-sample period for
                              paper trading / backtesting.
        return_floor_low: minimum expected return for low-risk scenario.
        output_dir: folder where CSVs and plots will be written.
        min_weight, max_weight: linking constraints.
        min_assets: minimum number of stocks selected.
        sector_map: dict {ticker: sector/group}, optional.
        sector_min: minimum stocks per sector/group (if sector_map provided).
        scenarios: iterable of scenarios to solve: subset of {"low", "medium", "high"}.
        solver_name: name of MINLP solver (default "bonmin").
        solver_options: dict of additional solver options.

    Returns:
        dict with:
          - "train_monthly_returns"
          - "mu", "cov"
          - "scenario_results": dict scenario -> info
          - "backtest": dict (if test period provided) with
                        "equity_curves" and "metrics"
    """
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------
    # 1. Download price data for training + optional test
    # ----------------------------------------------------
    full_start = train_start
    full_end = test_end if test_end is not None else train_end
    price_df = _download_price_data(tickers, full_start, full_end)

    # Save all prices
    price_df.to_csv(os.path.join(output_dir, "all_price_data.csv"))

    # Split into train / test
    train_prices = price_df.loc[train_start:train_end]
    if test_start is not None and test_end is not None:
        test_prices = price_df.loc[test_start:test_end]
    else:
        test_prices = None

    # ----------------------------------------------------
    # 2. Compute training returns (monthly for mu/cov)
    # ----------------------------------------------------
    train_monthly = train_prices.resample('M').ffill().pct_change().dropna()
    _display_df(train_monthly, title="=== Training Monthly Returns ===")

    mu = train_monthly.mean()
    cov = train_monthly.cov()

    mu.to_csv(os.path.join(output_dir, "train_expected_returns.csv"))
    cov.to_csv(os.path.join(output_dir, "train_covariance_matrix.csv"))

    # ----------------------------------------------------
    # 3. Solve Bonmin model for each scenario
    # ----------------------------------------------------
    scenario_results = {}
    weight_table = {}

    for sc in scenarios:
        if sc == "low":
            res = _solve_bonmin_portfolio(
                mu, cov,
                scenario="low",
                min_assets=min_assets,
                min_weight=min_weight,
                max_weight=max_weight,
                sector_map=sector_map,
                sector_min=sector_min,
                return_floor=return_floor_low,
                solver_name=solver_name,
                solver_options=solver_options
            )
        elif sc == "medium":
            res = _solve_bonmin_portfolio(
                mu, cov,
                scenario="medium",
                min_assets=min_assets,
                min_weight=min_weight,
                max_weight=max_weight,
                sector_map=sector_map,
                sector_min=sector_min,
                solver_name=solver_name,
                solver_options=solver_options
            )
        elif sc == "high":
            res = _solve_bonmin_portfolio(
                mu, cov,
                scenario="high",
                min_assets=min_assets,   # still there, but one_stock overrides
                min_weight=min_weight,
                max_weight=max_weight,
                sector_map=sector_map,
                sector_min=sector_min,
                solver_name=solver_name,
                solver_options=solver_options
            )
        else:
            raise ValueError(f"Unknown scenario {sc}")

        scenario_results[sc] = res
        weight_table[sc] = res["weights"]

    weights_df = pd.DataFrame(weight_table)
    _display_df(weights_df, title="=== Scenario Weights (Training) ===")
    weights_df.to_csv(os.path.join(output_dir, "bonmin_portfolio_weights.csv"))

    # Simple summary of risk/return per scenario
    summary_rows = []
    for sc, res in scenario_results.items():
        summary_rows.append({
            "scenario": sc,
            "expected_return": res["port_return"],
            "expected_variance": res["port_risk"]
        })
    summary_df = pd.DataFrame(summary_rows)
    _display_df(summary_df, title="=== Scenario Summary (Expected Return & Risk) ===")
    summary_df.to_csv(os.path.join(output_dir, "bonmin_scenario_summary.csv"), index=False)

    # ----------------------------------------------------
    # 4. Paper trading / backtesting (optional)
    # ----------------------------------------------------
    backtest_result = None

    if test_prices is not None and not test_prices.empty:
        print("\nRunning paper trading / backtest on test period "
              f"{test_start} to {test_end} ...")

        test_returns_daily = test_prices.pct_change().dropna()
        equity_curves = pd.DataFrame(index=test_returns_daily.index)
        metrics_rows = []

        for sc, res in scenario_results.items():
            w = res["weights"]
            # Align weights with test_returns columns
            w = w.reindex(test_returns_daily.columns).fillna(0.0)

            port_daily_ret = (test_returns_daily * w).sum(axis=1)
            equity = (1 + port_daily_ret).cumprod()

            equity_curves[sc] = equity

            total_return = equity.iloc[-1] - 1.0
            days = len(port_daily_ret)
            if days > 0:
                ann_return = (1 + total_return)**(252.0 / days) - 1
                ann_vol = port_daily_ret.std() * np.sqrt(252.0)
                sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
            else:
                ann_return = np.nan
                ann_vol = np.nan
                sharpe = np.nan

            metrics_rows.append({
                "scenario": sc,
                "total_return": total_return,
                "annualized_return": ann_return,
                "annualized_volatility": ann_vol,
                "sharpe_ratio": sharpe
            })

        # Save backtest results
        equity_curves.to_csv(os.path.join(output_dir, "backtest_equity_curves.csv"))
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(os.path.join(output_dir, "backtest_metrics.csv"), index=False)

        _display_df(equity_curves.tail(), title="=== Backtest Equity Curves (Last 5 Days) ===")
        _display_df(metrics_df, title="=== Backtest Metrics by Scenario ===")

        # Plot equity curves
        ax = equity_curves.plot(figsize=(12, 6))
        ax.set_title("Paper Trading: Portfolio Equity Curves")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity (Starting at 1.0)")
        ax.grid(True)

        fig = ax.get_figure()
        fig.savefig(
            os.path.join(output_dir, "backtest_equity_curves.jpg"),
            format="jpg",
            dpi=300
        )
        plt.close(fig)

        backtest_result = {
            "equity_curves": equity_curves,
            "metrics": metrics_df
        }

    print(f"\nPart 2 (Bonmin) outputs saved in: {output_dir}/")

    return {
        "train_monthly_returns": train_monthly,
        "mu": mu,
        "cov": cov,
        "scenario_results": scenario_results,
        "backtest": backtest_result
    }


# Example Part 2 run (you can tweak or comment this out)
if __name__ == "__main__":
    # Example sector mapping (user can change this freely)
    sector_map_example = {
        'AAPL': 'Tech',
        'MSFT': 'Tech',
        'NVDA': 'Tech',
        'AMZN': 'Consumer',
        'GOOGL': 'Tech'
    }

    results_part2 = bonmin_portfolio_pipeline(
        tickers=['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL'],
        train_start='2020-01-01',
        train_end='2024-12-31',
        test_start='2025-01-01',
        test_end='2025-07-01',
        return_floor_low=0.01,
        output_dir="my_output_part2",
        min_weight=0.02,
        max_weight=0.20,
        min_assets=3,
        sector_map=sector_map_example,
        sector_min=1,
        scenarios=("low", "medium", "high"),
        solver_name="bonmin",       # assumes Bonmin is installed and on PATH
        solver_options=None
    )
