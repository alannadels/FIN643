# FIN 643 - Quantitative Finance and Market Microstructure

This repository contains all exercises and assignments for FIN 643 - Quantitative Finance and Market Microstructure.

## Repository Structure

- `Exercise1_Technical_Analysis/` - Technical analysis trading strategies using candlestick patterns and moving averages.
- `Exercise2_Option_Trading_Strategies/` - Option pricing, Greeks analysis, and volatility trading strategies.
- `Exercise3_Structured_Notes/` - Pricing and analysis of structured financial products.
- `Exercise4_Rates/` - Interest rate modeling and fixed income derivatives.
- `Exercise5_Currency_Markets/` - Foreign exchange markets and currency derivatives analysis.

## Exercise Summaries

### Exercise 1: Technical Analysis
Developed a trading strategy for Netflix (NFLX) combining green hammer candlestick patterns with moving average crossovers. The strategy achieved a 71% win rate with a Sharpe ratio of 1.01, outperforming buy-and-hold on a risk-adjusted basis.

### Exercise 2: Option Trading Strategies
Built a reinforcement learning agent using PPO to dynamically select between buy-and-hold, covered calls, and protective puts for NCLH options. The agent learned to use covered calls in bearish conditions, turning a -9.45% loss into a +2.47% gain during the 2025 test period.

### Exercise 3: Structured Notes
Designed and backtested an autocallable structured note on a basket of NVDA (75%) and URA (25%) with downside buffer protection and a kicker multiplier. The optimized structure achieved a Sharpe ratio of 1.32 across rolling 3-year periods.

### Exercise 4: Rates
Implemented a TLT carry strategy with multi-signal risk management that achieved a 27% total return vs -3% for buy-and-hold. The strategy reduced max drawdown from -48% to -13% by only being in the market 28% of the time.

### Exercise 5: Currency Markets
Created a yen carry trade strategy borrowing JPY to invest in US tech stocks (GOOGL, NVDA, IBM), using VIX>20 as a risk-off signal. The strategy achieved an outstanding Sharpe ratio of 3.85 with 1238% total returns over 5 years (2020-2024), all while avoiding major drawdowns.
