# Forex_Trading_System

# Objective
The objective was to build a Forex trading strategy using real-time currency pair data to predict market movements and execute profitable trades

# Scope
This strategy involves retrieving real-time data, performing detailed data analysis, deploying predictive models, and applying a systematic trading strategy

# Tools and Technologies Used
1. Polygon API: For fetching real-time Forex data
2. MongoDB: For storing and retrieving currency data
3. PyCaret: For building and deploying predictive models to forecast currency movements.

# Data Collection
Utilized the Polygon API to gather real-time data on key currency pairs such as EURUSD, GBPCHF, USDCAD, EURCHF, EURCAD, GBPEUR, GBPUSD, GBPCAD, USDCHF and USDJPY

# Feature Engineering
Calculated average mean prices, volume, and other relevant statistics across selected base currency pairs
Used correlations with EURUSD and USDJPY to enhance feature sets for model accuracy

# Predictive Models
1. Regression Model: Deployed a regression model using PyCaret to predict future price movements based on aggregated statistics
2. Classification Model: Categorized currency pairs into Forecastable, Non-Forecastable and Undefined based on volatility metrics

# L/S Trading Strategy
1. Determined Long and Short positions for GBPUSD and USDJPY using a 20-point linear regression analysis of the latest price data
2. Initiated the trading strategy at hour #5, with subsequent adjustments and re-executions at hours #6 and #7, and closing the position at hour #8
3. Adjusted trading ratios to account for the scale differences between USDJPY and GBPUSD, using a factor of 100
4. Computed the profit and loss for each trade cycle, assuming a standard investment of $100 per trade step
5. Evaluated the success of the trading strategy based on accumulated profits or losses and adjust strategies as needed for optimizing returns.



