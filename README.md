# Stock Portfolio Optimization 

This repository contains a two-part project for stock portfolio optimization:

1. **Part 1 – Classical Markowitz Optimization (Approved)**
2. **Part 2 – Extended Mixed-Integer Model with Bonmin, Logical Constraints, and Paper Trading**

---

## 1. Part 1: Markowitz Efficient Frontier

**File:** `portfolio_optimization.py`  
**Function:** `full_portfolio_pipeline(...)`

Part 1 implements a **mean–variance optimization** using `scipy.optimize.minimize`:

- Downloads daily price data from **Yahoo! Finance** via `yfinance`
- Builds:
  - Adjusted Close prices (`price_data.csv`)
  - Daily returns (`daily_returns.csv`)
  - Monthly returns (`monthly_returns.csv`)
- Generates plots:
  - Cumulative returns
  - Covariance & correlation heatmaps (monthly returns)
  - Efficient frontier and several diagnostic plots
- Optimizes portfolio weights across a grid of **risk limits** with a **minimum return floor**
- Saves all tables and plots into a specified output folder

Example usage (from Python):

```python
from portfolio_optimization import full_portfolio_pipeline

results_part1 = full_portfolio_pipeline(
    tickers=['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL'],
    start='2020-01-01',
    end='2025-01-01',
    output_dir='my_output_part1'
)
```

Outputs are written into my_output_part1/.
**## 2. Part 2: Bonmin + Binary Variables + Paper Trading**

File: portfolio_optimization.py
Function: bonmin_portfolio_pipeline(...)

Part 2 upgrades the pipeline by:

Using Bonmin (a COIN-OR MINLP solver) via Pyomo

Introducing binary / activation variables for each stock

Implementing logical constraints, e.g.:

Choose at least N stocks

Choose at least one stock from each sector / group

Adding linking constraints:

If you invest in a stock:

𝑧
𝑖
=
1
⇒
min_weight
≤
𝑤
𝑖
≤
max_weight
z
i
	​

=1⇒min_weight≤w
i
	​

≤max_weight

If you do not invest:

𝑧
𝑖
=
0
⇒
𝑤
𝑖
=
0
z
i
	​

=0⇒w
i
	​

=0

Defining low, medium, and high risk scenarios:

Low risk: minimize variance subject to a minimum expected return floor (return_floor_low)

Medium risk: trade off expected return and variance (approximates the top-left of the efficient frontier)

High risk: “dump all in one stock” – the model picks exactly one stock and maximizes expected return

Supporting paper trading / backtesting:

Train the model on a training period (e.g., 2020-01-01 to 2024-12-31)

Evaluate resulting portfolios on a test period (e.g., 2025-01-01 to 2025-07-01)

Compute equity curves, total returns, annualized returns, volatility, and Sharpe ratios

3. Installation
3.1 Python Packages

Create a virtual environment (optional but recommended), then:

pip install numpy pandas matplotlib seaborn yfinance pyomo

3.2 Bonmin

You also need a working installation of Bonmin accessible to Pyomo as a solver named bonmin. Installation steps depend on your OS (COIN-OR, conda, or system packages). Once installed, you should be able to run:

bonmin -v


from a terminal, and Pyomo's SolverFactory("bonmin") should succeed.

4. Function: bonmin_portfolio_pipeline(...)
from portfolio_optimization import bonmin_portfolio_pipeline

Arguments

tickers: list of stock tickers, e.g. ['AAPL','MSFT','NVDA','AMZN','GOOGL']

train_start, train_end (str): training period (e.g. '2020-01-01', '2024-12-31')

test_start, test_end (str, optional): test period for paper trading

If omitted, the pipeline skips backtesting

return_floor_low (float): minimum expected return for the low-risk scenario

output_dir (str): folder where CSVs and plots are written

min_weight, max_weight (floats): linking constraints

If a stock is selected (z_i = 1), then:

w_i >= min_weight

w_i <= max_weight

min_assets (int): minimum number of stocks that must be selected

sector_map (dict, optional): mapping ticker → sector/group, e.g.:

sector_map = {
    'AAPL': 'Tech',
    'MSFT': 'Tech',
    'NVDA': 'Tech',
    'AMZN': 'Consumer',
    'GOOGL':'Tech'
}


sector_min (int): minimum number of stocks per sector (if sector_map is provided)

scenarios: iterable of scenario names ("low", "medium", "high")

solver_name (str): solver name for Pyomo (default "bonmin")

solver_options (dict, optional): extra solver options, e.g. tolerances

Returns

A dictionary with:

"train_monthly_returns" – monthly returns used for fitting

"mu" – expected returns (vector)

"cov" – covariance matrix

"scenario_results" – dict: scenario → info

weights, active (binary vector), port_return, port_risk, solver result

"backtest" – None if no test period, otherwise:

"equity_curves" – DataFrame of cumulative equity for each scenario

"metrics" – DataFrame with total return, annualized return, volatility, Sharpe ratio

5. How the Logic & Constraints Work
5.1 Binary / Activation Variables

For each stock 
𝑖
i:

𝑤
𝑖
w
i
	​

: continuous portfolio weight (0 to 1)

𝑧
𝑖
∈
{
0
,
1
}
z
i
	​

∈{0,1}: binary activation variable

Linking constraints:

If we invest in a stock:

𝑤
𝑖
≥
min_weight
⋅
𝑧
𝑖
,
𝑤
𝑖
≤
max_weight
⋅
𝑧
𝑖
w
i
	​

≥min_weight⋅z
i
	​

,w
i
	​

≤max_weight⋅z
i
	​


If 
𝑧
𝑖
=
0
z
i
	​

=0, then 
𝑤
𝑖
=
0
w
i
	​

=0.

The model also enforces:

∑
𝑖
𝑤
𝑖
=
1
i
∑
	​

w
i
	​

=1

Min number of stocks:

∑
𝑖
𝑧
𝑖
≥
min_assets
i
∑
	​

z
i
	​

≥min_assets
5.2 Sector / Group Constraints

If sector_map is provided, each ticker is assigned to a sector or group.
For each sector 
𝑠
s, the model enforces:

∑
𝑖
∈
sector 
𝑠
𝑧
𝑖
≥
sector_min
i∈sector s
∑
	​

z
i
	​

≥sector_min

This ensures you invest in at least one (or more) stock from each sector.

6. Risk Scenarios

The model supports three named scenarios:

Low risk ("low")

Objective: minimize portfolio variance

Constraint: portfolio expected return must be at least return_floor_low

Still respects linking, min assets, and sector constraints.

Medium risk ("medium")

Objective: trade off risk and return:

min
⁡
 
𝜆
⋅
variance
−
return
min λ⋅variance−return

where 
𝜆
=
1.0
λ=1.0 (risk aversion).

Intuitively, this approximates a point near the top-left of the efficient frontier.

High risk ("high")

Logic: forces the model to choose exactly one stock:

∑
𝑖
𝑧
𝑖
=
1
i
∑
	​

z
i
	​

=1

Objective: maximize expected return (implemented as minimize -return)

This corresponds to the “dump all in one stock” high-risk scenario.

You can select any subset of scenarios:

scenarios=("low","medium","high")  # all three
scenarios=("medium","high")        # only medium + high

7. Paper Trading / Backtesting

To evaluate how your optimization would have performed out-of-sample:

Choose a training period for fitting:

Example: train_start='2020-01-01', train_end='2024-12-31'

Choose a test period for paper trading:

Example: test_start='2025-01-01', test_end='2025-07-01'

The pipeline:

Fits the Bonmin model on the training period to get optimized weights.

Applies these weights to daily returns in the test period.

Computes:

Daily portfolio returns

Cumulative equity curve (starting at 1.0)

Total return, annualized return, annualized volatility, Sharpe ratio

Saves:

backtest_equity_curves.csv

backtest_metrics.csv

backtest_equity_curves.jpg

8. Example Usage (Part 2)
from portfolio_optimization import bonmin_portfolio_pipeline

sector_map = {
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
    output_dir='my_output_part2',
    min_weight=0.02,
    max_weight=0.20,
    min_assets=3,
    sector_map=sector_map,
    sector_min=1,
    scenarios=("low", "medium", "high"),
    solver_name='bonmin',
    solver_options=None
)

Key Outputs in my_output_part2/

all_price_data.csv – raw adjusted close prices

train_expected_returns.csv – expected returns (training period)

train_covariance_matrix.csv – covariance matrix (training period)

bonmin_portfolio_weights.csv – weights per scenario (training)

bonmin_scenario_summary.csv – expected return & variance per scenario

backtest_equity_curves.csv – cumulative equity for each scenario (if test period provided)

backtest_metrics.csv – total return, annualized return/volatility, Sharpe (if test period provided)

backtest_equity_curves.jpg – equity curves plot for all scenarios

9. Customizing Constraints & Logic

The code is generalized so users can change the logic directly via function parameters:

Change min/max weights per asset:

min_weight=0.05, max_weight=0.25


Change min number of stocks:

min_assets=5


Change sector/group constraints:

sector_map = {...}
sector_min = 2  # at least 2 stocks per sector


Change which scenarios to run:

scenarios=("low","high")


Change low-risk return floor:

return_floor_low=0.015


For more advanced customization (e.g., “at most 2 stocks from sector X”, sector caps, etc.), you can extend the Pyomo model in _solve_bonmin_portfolio(...) by adding new constraints.

10. Reproducibility & Notes

All data is fetched from Yahoo! Finance via yfinance; results can change over time if new data arrives or historical data is revised.

The model assumes long-only positions (w_i >= 0).

The paper trading backtest uses fixed weights from the training period; it does not rebalance or re-estimate dynamically.

The Sharpe ratio uses 252 trading days per year.
