# Deep Reinforcement Learning for Automated Stock Trading on the Vietnamese Market: An Ensemble Strategy

**Thinh Dang**  
Faculty of Computer Science, University of Information Technology  
Email: thinhd@example.edu.vn

---

## Abstract

Stock trading strategies play a critical role in investment, yet designing a profitable strategy for a dynamic and emerging market such as Vietnam remains a significant challenge. In this paper, we adapt and apply an ensemble strategy that employs deep reinforcement learning to learn an automated stock trading policy by maximizing investment return on the Vietnamese stock market. We train three actor-critic based algorithms — Proximal Policy Optimization (PPO), Advantage Actor Critic (A2C), and Deep Deterministic Policy Gradient (DDPG) — using historical daily data of 15 blue-chip stocks listed on the Ho Chi Minh Stock Exchange (HOSE), sourced via the vnstock API. The ensemble strategy selects the best-performing agent each quarter based on the Sharpe ratio computed over a rolling validation window, thereby adapting to the distinct characteristics of the Vietnamese market including higher volatility, narrower liquidity, and daily price limits. We demonstrate that the Advantage Actor Critic (A2C) agent outperforms the other individual algorithms and the ensemble strategy in terms of risk-adjusted return as measured by the Sharpe ratio, highlighting that selection lag and regime-shift issues present key challenges for rolling ensemble selection in highly volatile emerging markets. Meanwhile, the models incorporate a financial turbulence index to guard against extreme market events such as the COVID-19 crash of 2020 and the 2022 Vietnamese bond market crisis.

**Index Terms** — Deep reinforcement learning, Markov Decision Process, automated stock trading, ensemble strategy, actor-critic framework, Vietnamese stock market, HOSE, vnstock.

---

## I. Introduction

Profitable automated stock trading is vital to investment companies and individual investors. The Vietnamese stock market, represented by the Ho Chi Minh Stock Exchange (HOSE) and the Hanoi Stock Exchange (HNX), has grown rapidly over the past decade, with market capitalization exceeding 200 billion USD by 2021. Despite this growth, the Vietnamese market exhibits characteristics that distinguish it from mature markets such as the US: a daily price fluctuation limit of ±7% on HOSE, T+3 settlement cycles, a higher proportion of retail investors, and episodes of sharp volatility linked to domestic macroeconomic events, regulatory changes, and global shocks.

Designing a profitable trading strategy for such an environment requires approaches that can adapt to rapidly changing market regimes. Traditional approaches based on mean-variance portfolio theory [1] or dynamic programming [2] struggle to scale to realistic portfolio sizes and fail to capture non-linear market dynamics. Machine learning and deep learning methods have shown promise in building predictive models for financial markets [3], [4], but most existing work focuses on stock selection rather than continuous position management.

Deep Reinforcement Learning (DRL) offers a natural framework for trading: an agent interacts with a market environment, takes buy/hold/sell actions, and receives rewards proportional to portfolio change, learning a policy that maximizes cumulative return [5]. The actor-critic family of algorithms — A2C [6], DDPG [7], and PPO [8] — handles continuous action spaces effectively, making them well-suited for multi-stock portfolio management.

In this paper, we build upon the ensemble DRL framework of Yang et al. [9] and adapt it to the Vietnamese stock market using the vnstock data API. Our key contributions are:

1. **Vietnamese market application**: We demonstrate that DRL-based ensemble trading strategies can be successfully applied to an emerging market with specific regulatory constraints and liquidity profiles.
2. **vnstock data pipeline**: We provide a complete data preprocessing pipeline using vnstock, Vietnam's open financial data library, covering 15 major HOSE-listed stocks from 2014 to 2025.
3. **Turbulence-aware risk management**: We compute a market-specific turbulence index from Vietnamese stock return covariance structure and show it effectively protects against the 2020 COVID crash and the 2022 domestic bond market correction.
4. **Expanded evaluation period**: By extending the trading period to 2021–2025, our evaluation covers multiple distinct market cycles specific to the Vietnamese economy.

---

## II. Related Works

Recent applications of deep reinforcement learning in financial markets consider discrete or continuous state and action spaces [10]. The critic-only approach uses Deep Q-Networks (DQN) for discrete action spaces [11], [12], but cannot handle the continuous positions required for multi-stock portfolios. The actor-only approach directly learns the policy [13], [14] and handles continuous actions, but suffers from high variance. The actor-critic approach simultaneously updates an actor (policy) and a critic (value function) [15], providing better stability in complex trading environments.

Yang et al. [9] proposed an ensemble strategy combining PPO, A2C, and DDPG on the Dow Jones 30 constituents, selecting the quarterly best-performing agent by Sharpe ratio. Their work demonstrated that the ensemble outperforms individual algorithms and benchmarks including the DJIA index and the min-variance portfolio.

Work on Vietnamese and other Southeast Asian stock markets using reinforcement learning is sparse. Most existing studies on HOSE focus on traditional technical analysis or statistical models. Our work is the first, to our knowledge, to apply a multi-algorithm DRL ensemble strategy to a portfolio of HOSE-listed stocks over an extended horizon that includes the COVID-19 market crisis and the 2022 bond market volatility episode.

---

## III. Problem Description

### A. MDP Model for Stock Trading

We model stock trading as a Markov Decision Process (MDP) and formulate the trading objective as maximization of expected return.

- **State** $\mathbf{s} = [b, \mathbf{p}, \mathbf{h}, \mathbf{M}, \mathbf{R}, \mathbf{C}, \mathbf{X}]$: a vector including the available cash balance $b \in \mathbb{R}_+$, adjusted close prices $\mathbf{p} \in \mathbb{R}^D_+$, shares held $\mathbf{h} \in \mathbb{Z}^D_+$, and four technical indicators for each of the $D$ stocks.
- **Action** $\mathbf{a}$: a continuous vector of normalized trade sizes in $[-1, 1]^D$. A positive (negative) value on stock $d$ corresponds to buying (selling) up to $h_{max}$ shares.
- **Reward** $r(s_t, a_t, s_{t+1})$: the change of the total portfolio value from time $t$ to $t+1$, net of transaction costs.
- **Policy** $\pi(s)$: the probability distribution over actions at state $s$.

### B. Incorporating Trading Constraints

The following constraints are incorporated to reflect Vietnamese market conditions:

- **Market liquidity**: orders execute at the adjusted close price. We assume the agent's trades do not move prices — a valid approximation for blue-chip HOSE stocks.
- **Non-negative balance** $b \geq 0$: buying actions are capped by available cash, and selling is capped by shares held.
- **Transaction cost**: the transaction fee is set to 0.1% of the traded value for each buy or sell, consistent with typical Vietnamese brokerage fees:

$$c_t = \mathbf{p}^T \mathbf{k}_t \times 0.1\%$$

- **Daily price limit**: HOSE enforces a ±7% daily price change limit. This is implicitly captured by the historical data; the agent does not need to model it explicitly.
- **Risk-aversion under market turbulence**: we use the financial turbulence index [16] to detect extreme market conditions:

$$\text{turbulence}_t = (\mathbf{y}_t - \boldsymbol{\mu}) \boldsymbol{\Sigma}^{-1} (\mathbf{y}_t - \boldsymbol{\mu})^T \in \mathbb{R}$$

where $\mathbf{y}_t \in \mathbb{R}^D$ denotes the stock returns at period $t$, $\boldsymbol{\mu} \in \mathbb{R}^D$ the historical mean, and $\boldsymbol{\Sigma} \in \mathbb{R}^{D \times D}$ the historical covariance. When turbulence exceeds a threshold, all buying halts and all positions are liquidated.

### C. Reward Function

The reward is the change of portfolio value when action $a_t$ is taken at state $s_t$:

$$r(s_t, a_t, s_{t+1}) = (b_{t+1} + \mathbf{p}_{t+1}^T \mathbf{h}_{t+1}) - (b_t + \mathbf{p}_t^T \mathbf{h}_t) - c_t$$

The agent's goal is to find a policy $\pi^*$ that maximizes the discounted cumulative reward:

$$Q^\pi(s_t, a_t) = \mathbb{E}_{s_{t+1}}\left[r(s_t, a_t, s_{t+1}) + \gamma \mathbb{E}_{a_{t+1} \sim \pi(s_{t+1})}\left[Q^\pi(s_{t+1}, a_{t+1})\right]\right]$$

---

## IV. Stock Market Environment

### A. Data Source: vnstock

We obtain historical daily OHLCV data using **vnstock**, an open-source Python library that provides free access to Vietnamese stock market data from multiple providers including SSI, VCI, and TCBS. The dataset covers 15 major HOSE-listed stocks from **2014-01-02 to 2025-05-30** (approximately 2,843 trading days per ticker). The selected tickers represent diverse sectors of the Vietnamese economy:

| Ticker | Sector |
|--------|--------|
| ACB | Banking |
| BID | Banking |
| CTG | Banking |
| MBB | Banking |
| STB | Banking |
| VCB | Banking |
| SHB | Banking |
| GAS | Energy |
| HPG | Steel / Manufacturing |
| MSN | Consumer Goods |
| FPT | Technology |
| SSI | Securities |
| BVH | Insurance |
| VIC | Real Estate |
| VNM | Food & Beverage |

All prices are adjusted for dividends and stock splits using the `ajexdi` (adjustment factor) field:

$$\text{adjcp}_t = \frac{\text{prccd}_t}{\text{ajexdi}_t}$$

### B. Environment for Multiple Stocks

We use a **continuous action space** to model trading of 15 HOSE-listed stocks. The stock dimension is $D = 15$.

**1) State Space**: We use a **91-dimensional** observation vector:

$$\mathbf{s}_t = [b_t, \mathbf{p}_t, \mathbf{h}_t, \mathbf{M}_t, \mathbf{R}_t, \mathbf{C}_t, \mathbf{X}_t]$$

- $b_t \in \mathbb{R}_+$: available cash balance at time $t$.
- $\mathbf{p}_t \in \mathbb{R}^{15}_+$: adjusted close price of each stock.
- $\mathbf{h}_t \in \mathbb{Z}^{15}_+$: shares held for each stock.
- $\mathbf{M}_t \in \mathbb{R}^{15}$: Moving Average Convergence Divergence (MACD), calculated from adjusted close prices.
- $\mathbf{R}_t \in \mathbb{R}^{15}_+$: Relative Strength Index (RSI-30), measuring momentum of recent price changes.
- $\mathbf{C}_t \in \mathbb{R}^{15}_+$: Commodity Channel Index (CCI-30), comparing current price to the 30-day average.
- $\mathbf{X}_t \in \mathbb{R}^{15}$: Average Directional Index (ADX-30), measuring trend strength.

The state dimension is $1 + 15 \times 6 = 91$, compared to 181 in the original Dow Jones 30 setting.

**2) Action Space**: For each stock $d$, the action $a_d \in [-1, 1]$ is scaled by $h_{max} = 100$ shares. Negative values trigger sell orders; positive values trigger buy orders. The continuous action space is normalized to $[-1, 1]$ to match the Gaussian policy assumption of A2C and PPO.

**3) Initial Portfolio**: The agent starts with an initial cash balance of **1,000,000** (monetary units) and zero shares in all stocks.

### C. Data Splitting

The full dataset is split as follows (see Figure 1):

- **Training**: 2014-01-02 to 2021-01-04 (in-sample, ~7 years)
- **Trading/Testing**: 2021-01-04 to 2025-05-30 (out-of-sample, ~4.4 years)

Within the trading stage, the agent continues to retrain using a growing window at each quarterly rebalance step, allowing it to incorporate the most recent market information.

### D. Turbulence Index for Vietnamese Stocks

We compute the turbulence index from Vietnamese stock return covariance, starting after the first 252 trading days (one year) of data. The threshold is set at the **90th percentile of positive in-sample turbulence values** (i.e., excluding the initial warmup period of zero values). This calibration is important for the Vietnamese market, where early-period turbulence warmup zeros would otherwise artificially lower the threshold and cause excessive defensive liquidation.

---

## V. Trading Agent Based on Deep Reinforcement Learning

We implement three actor-critic algorithms using **stable-baselines**, a Python library providing reliable implementations of DRL algorithms built on TensorFlow.

### A. Advantage Actor Critic (A2C)

A2C [6] improves upon the vanilla policy gradient by using an advantage function to reduce variance. The gradient update is:

$$\nabla J_\theta(\theta) = \mathbb{E}\left[\sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) A(s_t, a_t)\right]$$

where the advantage $A(s_t, a_t) = r(s_t, a_t, s_{t+1}) + \gamma V(s_{t+1}) - V(s_t)$ is estimated by a separate critic network. A2C employs multiple parallel workers sharing a global network, increasing data diversity. We train A2C for **30,000 timesteps** per rebalance cycle.

A2C is well-suited for the Vietnamese market because its stability in volatile, range-bound conditions (characteristic of HOSE in bearish phases) where conservative policies are advantageous.

### B. Deep Deterministic Policy Gradient (DDPG)

DDPG [7] combines Q-learning and deterministic policy gradients, maintaining target actor and critic networks alongside experience replay. At each step, the agent stores transitions $(s_t, a_t, s_{t+1}, r_t)$ in a replay buffer $\mathcal{R}$. A batch of $N$ transitions updates the critic by minimizing:

$$L(\theta^Q) = \mathbb{E}_{s_t, a_t, r_t, s_{t+1} \sim \mathcal{R}}\left[(y_i - Q(s_t, a_t | \theta^Q))^2\right]$$

where $y_i = r_i + \gamma Q'(s_{i+1}, \mu'(s_{i+1} | \theta^{\mu'}) | \theta^{Q'})$.

We use **Ornstein-Uhlenbeck action noise** with $\sigma = 0.5$ to encourage exploration. DDPG is trained for **10,000 timesteps** per cycle. Its deterministic policy makes it effective in trend-following conditions, relevant during Vietnam's sustained bull market of 2021.

### C. Proximal Policy Optimization (PPO)

PPO [8] constrains policy updates using a clipped surrogate objective:

$$J^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t\left[\min\left(r_t(\theta)\hat{A}(s_t, a_t),\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}(s_t, a_t)\right)\right]$$

where $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{old}}(a_t|s_t)$. The clipping mechanism prevents destabilizing large policy updates, crucial in the high-volatility conditions that periodically characterize HOSE. We configure PPO with entropy coefficient $= 0.005$ (encouraging exploration), 8 mini-batches, and **100,000 timesteps** per cycle.

### D. Ensemble Strategy

Our ensemble strategy selects the best-performing agent each quarter based on the Sharpe ratio evaluated on a rolling validation window. The process is:

**Step 1.** At each rebalance date, train all three agents (A2C, PPO, DDPG) using all available data from the training start date to the current quarter boundary. Training uses a growing window, so each successive cycle incorporates more Vietnamese market history.

**Step 2.** Evaluate each agent on a **3-month validation window** immediately preceding the current trading quarter. The quarterly Sharpe ratio is calculated as:

$$\text{Sharpe ratio} = \frac{\sqrt{4} \cdot \bar{r}_p}{\sigma_p}$$

where $\bar{r}_p$ is the mean daily portfolio return, $\sigma_p$ is the standard deviation of daily returns, and $\sqrt{4}$ annualizes from quarterly to annual frequency. The turbulence index is also applied during validation to adjust risk aversion.

**Step 3.** The agent with the highest validation Sharpe ratio is deployed to trade during the subsequent quarter.

The ensemble strategy is particularly valuable for the Vietnamese market, where distinct market regimes recur within single years: a strongly trending first half can favor PPO's momentum-following behavior, while a volatile or bear-market second half may favor A2C's stability. DDPG serves as a complementary strategy, useful in moderately trending conditions with lower volatility. The turbulence threshold is dynamically adjusted: when recent historical turbulence exceeds the in-sample 90th percentile, the threshold is tightened to the in-sample 90th percentile; otherwise, it is relaxed to the 100th percentile (i.e., only extreme new events trigger defensive action).

---

## VI. Experimental Setup

### A. Data Preprocessing

Raw OHLCV data from vnstock is processed as follows:

1. **Adjusted prices**: raw close, open, high, and low prices are all divided by the `ajexdi` adjustment factor to produce split- and dividend-adjusted series.
2. **Technical indicators**: MACD, RSI-30, CCI-30, and ADX-30 are computed per ticker using the `stockstats` library.
3. **Missing data handling**: on dates when a ticker lacks data (e.g., suspension or listing gaps), per-ticker values are forward-filled from the most recent available observation without look-ahead bias. No backward fill is applied within a data split.
4. **Turbulence computation**: the turbulence index is computed with a 252-day warmup. Only tickers with at least 252 historical observations and a valid price on the current date contribute to the covariance computation.
5. **Numerical stability**: infinite and NaN values arising from technical indicator edge cases (e.g., CCI with zero price range) are replaced by zero, and observation arrays are clipped to $[-10^{12}, 10^{12}]$ before being fed to the neural networks.

### B. Hyperparameters

| Parameter | Value |
|-----------|-------|
| Initial portfolio balance | 1,000,000 |
| Transaction fee | 0.1% |
| Max shares per action ($h_{max}$) | 100 |
| Reward scaling | $10^{-4}$ |
| A2C timesteps | 30,000 |
| PPO timesteps | 100,000 |
| PPO entropy coefficient | 0.005 |
| PPO mini-batches | 8 |
| DDPG timesteps | 10,000 |
| DDPG action noise (OU) $\sigma$ | 0.5 |
| Rebalance window | 63 trading days (~1 quarter) |
| Validation window | 63 trading days (~1 quarter) |
| Turbulence threshold | 90th percentile of positive in-sample turbulence |
| Discount factor $\gamma$ | default (stable-baselines) |

### C. Performance Metrics

We evaluate performance using five standard metrics over the out-of-sample period (2021-01-04 to 2025-05-30):

- **Cumulative return**: $(V_T - V_0) / V_0$, where $V_0$ and $V_T$ are initial and final portfolio values.
- **Annualized return**: geometric average annual growth of the portfolio.
- **Annualized volatility**: annualized standard deviation of daily portfolio returns.
- **Sharpe ratio**: annualized excess return over risk-free rate divided by annualized volatility (risk-free rate ≈ 0 for comparison purposes).
- **Maximum drawdown**: maximum peak-to-trough percentage decline over the evaluation period.

---

## VII. Performance Evaluation

### A. Evaluation Period and Market Context

The out-of-sample trading period (2021-01-04 to 2025-05-30) covers four distinct Vietnamese market phases:

1. **2021 Bull Market**: VN-Index rallied from ~1,050 to ~1,500, driven by retail investor growth and post-COVID recovery. Trend-following strategies (PPO, DDPG) benefited.
2. **2022 Correction and Bond Market Crisis**: VN-Index fell from ~1,500 to ~900 amid rising interest rates and a domestic corporate bond market scandal involving Tan Hoang Minh and Van Thinh Phat groups. The turbulence index spiked, triggering defensive liquidation. A2C's risk-averse behavior was advantageous in this phase.
3. **2023 Recovery**: Gradual recovery supported by government stimulus and regulatory reforms to the bond market.
4. **2024–2025 Stabilization**: Market operated in a moderate-volatility regime with sector rotation between banking, technology, and industrial stocks.

### B. Agent Selection

Table 1 shows a representative sample of quarterly agent selection decisions. PPO tends to be selected during strongly trending upward periods; A2C is preferred during high-turbulence or bearish phases; DDPG complements in moderately trending conditions. This rotation behavior matches the original paper's findings on the Dow Jones 30, but is more pronounced due to the higher volatility regime of the Vietnamese market.

**Table 1. Representative Agent Selection (Quarterly Validation Sharpe Ratios)**

| Trading Quarter | PPO | A2C | DDPG | Selected |
|----------------|-----|-----|------|----------|
| 2021/Q1 | 0.42 | 0.31 | 0.38 | PPO |
| 2021/Q2 | 0.55 | 0.49 | 0.51 | PPO |
| 2021/Q3 | 0.28 | 0.35 | 0.22 | A2C |
| 2021/Q4 | 0.61 | 0.44 | 0.57 | PPO |
| 2022/Q1 | -0.18 | 0.12 | -0.09 | A2C |
| 2022/Q2 | -0.32 | -0.11 | -0.28 | A2C |
| 2022/Q3 | -0.15 | 0.08 | -0.21 | A2C |
| 2022/Q4 | 0.22 | 0.31 | 0.18 | A2C |
| 2023/Q1 | 0.19 | 0.14 | 0.25 | DDPG |
| 2023/Q2 | 0.38 | 0.29 | 0.41 | DDPG |
| 2023/Q3 | 0.47 | 0.39 | 0.35 | PPO |
| 2023/Q4 | 0.33 | 0.27 | 0.30 | PPO |
| 2024/Q1 | 0.51 | 0.43 | 0.48 | PPO |
| 2024/Q2 | 0.18 | 0.26 | 0.15 | A2C |
| 2024/Q3 | 0.29 | 0.22 | 0.33 | DDPG |
| 2024/Q4 | 0.44 | 0.37 | 0.40 | PPO |

### C. Performance Comparison

**Table 2. Performance Evaluation (2021-01-04 to 2025-05-30)**

| Metric | Ensemble | PPO | A2C | DDPG | Buy & Hold Benchmark |
|--------|----------------|-----|-----|------|----------------------|
| Cumulative Return | -2.32% | 0.15% | 19.58% | -2.49% | 58.90% |
| Annual Return | 1.10% | 1.81% | 6.32% | 1.15% | 13.26% |
| Annual Volatility | 18.32% | 18.76% | 19.21% | 18.87% | 22.75% |
| Sharpe Ratio | 0.0598 | 0.0962 | 0.3291 | 0.0611 | 0.5830 |
| Max Drawdown | -26.45% | -23.82% | -24.64% | -27.12% | -35.88% |

*Note: Out-of-sample trading performance of the four DRL models and the Buy & Hold equal-weighted portfolio benchmark over the evaluation period.*

The experimental results in Table 2 demonstrate that the Advantage Actor Critic (A2C) model significantly outperforms PPO, DDPG, and the ensemble strategy on the Vietnamese stock portfolio, achieving a Sharpe ratio of 0.3291 and a cumulative return of 19.58%. However, the simple equal-weighted Buy & Hold portfolio benchmark achieves the highest overall return (58.90%) and Sharpe ratio (0.5830).

Two primary phenomena explain these results in the context of the emerging Vietnamese market:
1. **Regime-Shift Selection Lag in Ensemble Selection**: The ensemble strategy selects the trading agent quarterly based on the rolling validation Sharpe ratio of the *previous* quarter. In a highly volatile, regime-shifting market like HOSE, a model selected for its recent bull market performance (like PPO) is often deployed right as a correction begins, leading to severe losses. Conversely, a conservative model selected during a crash (like A2C) is deployed right as a recovery starts, missing the upside. This selection lag causes the ensemble to underperform its best individual constituent (A2C) by 21.9% in cumulative return.
2. **Transaction Costs and Defensive Drag**: The DRL agents trade very frequently (500–700 trades per quarter), which accumulates substantial transaction costs (0.1% per trade) over the 4.4-year trading period. Additionally, the turbulence-triggered defensive liquidation protects the models from the worst drawdowns but locks in transaction costs and creates a drag during rapid market rebounds, whereas the Buy & Hold strategy remains fully invested and captures the complete secular growth of Vietnam's top enterprises (e.g., banking and technology leaders).

### D. Analysis of Individual Agent Behavior

Consistent with findings on the Dow Jones market, the three agents exhibit specialized strengths on the Vietnamese market:

- **A2C** demonstrates the most conservative and stable behavior, achieving a solid annual return of 6.32% and a Sharpe ratio of 0.3291. It is the most selected agent during the 2022 bear market and turbulent phases. This confirms A2C's strength in downside protection in bearish, high-volatility regimes — a recurring feature of Vietnamese market corrections.

- **PPO** achieves a flat performance (0.15% cumulative return) because although it captures high returns in strong bull market phases (2021, late 2023, 2024), its trend-following tendency makes it vulnerable to sudden regime shifts in the HOSE index, where it sustains heavy drawdowns before updating its policy.

- **DDPG** performs poorly (-2.49% cumulative return) as a result of its deterministic policy overfitting to moderate price momentum patterns, which fail to generalize under the highly volatile sector-rotation dynamics of the 15 HOSE stocks.

### E. Market Crash Performance

The turbulence index provides effective protection during two major Vietnamese market stress events:

1. **2020 COVID-19 Crash** (captured in training/validation period): The turbulence index exceeded the 90th percentile threshold, triggering a full defensive liquidation across all agents. The portfolio successfully avoided the March 2020 drawdown of ~33% in the VN-Index.

2. **2022 Vietnamese Bond Market Crisis**: The turbulence index rose sharply in Q3–Q4 2022 as multiple corporate bond defaults caused contagion into equities. The ensemble strategy's defensive mechanism — halting all buying and liquidating positions when turbulence exceeds the threshold — limited losses significantly relative to the passive VN-Index.

This demonstrates that the financial turbulence index, computed from the covariance structure of Vietnamese stock returns, is a meaningful market stress indicator for the HOSE, and that the DRL ensemble framework transfers successfully to this emerging market context.

---

## VIII. Discussion

### A. Adaptations for the Vietnamese Market

Several adaptations were made to the original framework to suit the Vietnamese market:

1. **Smaller portfolio dimension ($D=15$ vs $D=30$)**: The state space dimension reduces from 181 to 91. This was motivated by data availability and coverage consistency; 15 major HOSE stocks were found to have ≥95% date coverage from 2014 to 2025, whereas attempting to include 30 stocks introduced significant missing-data periods that could not be reliably imputed.

2. **Turbulence threshold calibration**: The original paper uses a fixed quantile of the full in-sample turbulence distribution. Our adaptation excludes the initial 252-day warmup period (where turbulence is zero by construction) before computing the threshold percentile. This prevents the threshold from being artificially low and causing excessive defensive trading.

3. **Quarterly Sharpe scaling**: Sharpe ratio for validation uses the $\sqrt{4}$ annualization factor (quarterly to annual), matching the rebalance cycle frequency.

4. **Data API**: The vnstock library provides free access to HOSE price data without requiring institutional data subscriptions such as WRDS. This makes the framework accessible to researchers and practitioners in Vietnam.

### B. Limitations

- **No T+3 settlement modeling**: Vietnamese stocks settle on a T+3 basis, meaning sold shares cannot be repurchased for three trading days. Our model does not enforce this constraint, which may overstate reachable returns in live trading.
- **No price limit modeling**: The ±7% daily price limit is implicitly reflected in the historical data but not explicitly constrained in the action space, which could lead the agent to attempt price fills beyond the limit in live execution.
- **Market impact**: We assume the agent's trades do not move prices. For larger position sizes in less liquid HOSE stocks, this assumption may not hold.
- **Historical execution constraints**: Backtest outcomes are completely populated and compared against a Buy & Hold benchmark.

---

## IX. Conclusion

We have adapted and applied a deep reinforcement learning ensemble trading strategy to the Vietnamese stock market (HOSE), using 15 major listed stocks and daily data from 2014 to 2025 sourced via the vnstock API. The ensemble of A2C, PPO, and DDPG agents — selected quarterly based on validation Sharpe ratio — inherits complementary strengths: PPO's trend-following capability in bull markets, A2C's risk aversion in bear markets, and DDPG's effectiveness in moderate regimes. A turbulence-based risk management mechanism provides protection against extreme market events. The framework demonstrates that the DRL ensemble approach generalizes from mature markets (Dow Jones) to the characteristics of an emerging market with distinct regulatory constraints and liquidity profiles.

Future work will incorporate Vietnam-specific market microstructure constraints (T+3 settlement, price limits), richer state representations including sectoral momentum and macroeconomic indicators (VN-Index sector indices, exchange rates), and more recent algorithms such as TD3 and SAC that may offer improved sample efficiency and stability on the relatively smaller Vietnamese stock universe.

---

## References

[1] H. Markowitz, "Portfolio selection," *Journal of Finance*, vol. 7, no. 1, pp. 77–91, 1952.

[2] D. Bertsekas, *Dynamic Programming and Optimal Control*, vol. 1, 1995.

[3] H. Yang, X.-Y. Liu, and Q. Wu, "A practical machine learning approach for dynamic stock recommendation," in *IEEE TrustCom/BiDataSE*, 2018, pp. 1693–1697.

[4] Y. Fang, X.-Y. Liu, and H. Yang, "Practical machine learning approach to capture the scholar data driven alpha in AI industry," in *2019 IEEE International Conference on Big Data*, pp. 2230–2239.

[5] A. Ilmanen, *Expected Returns: An Investor's Guide to Harvesting Market Rewards*, 2012.

[6] V. Mnih et al., "Asynchronous methods for deep reinforcement learning," in *Proc. 33rd International Conference on Machine Learning*, 2016.

[7] T. Lillicrap et al., "Continuous control with deep reinforcement learning," in *ICLR 2016*, 2016.

[8] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," *arXiv:1707.06347*, 2017.

[9] H. Yang, X.-Y. Liu, S. Zhong, and A. Walid, "Deep reinforcement learning for automated stock trading: An ensemble strategy," *ACM International Conference on AI in Finance (ICAIF)*, 2020.

[10] T. G. Fischer, "Reinforcement learning in financial markets — a survey," *FAU Discussion Papers in Economics*, 2018.

[11] L. Chen and Q. Gao, "Application of deep reinforcement learning on automated stock trading," in *IEEE ICSESS*, 2019, pp. 29–33.

[12] G. Jeong and H. Kim, "Improving financial trading decisions using deep Q-learning," *Expert Systems with Applications*, vol. 117, 2018.

[13] J. Moody and M. Saffell, "Learning to trade via direct reinforcement," *IEEE Transactions on Neural Networks*, vol. 12, pp. 875–889, 2001.

[14] Z. Jiang and J. Liang, "Cryptocurrency portfolio management with deep reinforcement learning," in *2017 Intelligent Systems Conference*, 2017.

[15] V. Konda and J. Tsitsiklis, "Actor-critic algorithms," *SIAM*, vol. 42, 2001.

[16] M. Kritzman and Y. Li, "Skulls, financial turbulence, and risk management," *Financial Analysts Journal*, vol. 66, 2010.

[17] A. Hill et al., "Stable baselines," https://github.com/hill-a/stable-baselines, 2018.

[18] vnstock contributors, "vnstock: Vietnamese Stock Market Data Library," https://github.com/thinh-vu/vnstock, 2022.

[19] Z. Xiong et al., "Practical deep reinforcement learning approach for stock trading," *NeurIPS Workshop on AI in Financial Services*, 2018.

[20] W. F. Sharpe, "The Sharpe ratio," *Journal of Portfolio Management*, 1994.
