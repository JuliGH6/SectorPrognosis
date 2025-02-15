## SectorPrognosis

This project aims to predict the sector performance of multiple S&P 500 sectors using historical data collected from Bloomberg.

# Data

I have compiled a dataset covering six sectors—HealthCare, Materials, Industrials, ConsumerStaples, ConsumerDiscretionary, and Utilities—from 1997 to 2025. For each sector, I collected 14 key metrics, which include:
	•	Sector Index Price
	•	Performance (YoY percentage change)
	•	Dividends
	•	Price-to-Earnings (PE) Ratio
	•	Price-to-Book (P/B) Ratio
	•	Return on Equity (ROE)
	•	Price-to-Sales (P/S) Ratio
	•	Best PE (alternative valuation metric)
	•	Debt-to-Equity Ratio
	•	Debt-to-EBITDA Ratio
	•	Gross Margin
	•	EBITDA Margin
	•	Profit Margin
	•	Price-to-Free Cash Flow (P/FCF) Ratio

These sector-specific metrics are scaled within a 10-year window to account for significant market dynamics over time.

Additionally, I incorporate 13 unscaled macroeconomic indicators, including:
	•	Jobless Claims YoY (%)
	•	Total Jobless Claims
	•	GDP (Chained Dollars)
	•	Unemployment Rate (%)
	•	Consumer Price Index (CPI)
	•	Core CPI (excluding food & energy)
	•	Core Personal Consumption Expenditures (Core PCE)
	•	Producer Price Index (PPI)
	•	ISM Manufacturing Index
	•	ISM Services Index
	•	Existing Home Sales
	•	New Home Sales
	•	Employment Ratio

I also add one additional metric for the current month, which is scaled separately in the notebook.

After combining these metrics, I apply column-wise scaling before performing Principal Component Analysis (PCA) to reduce dimensionality.

Preprocessing & Target Variable

Each input to the model represents a time series of these metrics, covering data from 2 years to 1 year prior to the prediction date.
	•	Due to the high dimensionality (6 × 14 + 13 = 97 features per time point), I apply PCA to 21 components, preserving 95% of the variance.
	•	The final input representation consists of the concatenated PCA-reduced time points for a given prediction.
	•	The target variable is the percentage price movement of the sector index one year into the future.

 (Let me know if you are interested in the dataset!)

# Model

The neural network is optimized using Adam, with:
	•	Mean Squared Error (MSE) loss
	•	Cosine annealing for learning rate scheduling
	•	Dropout layers, weight decay, and Kaiming initialization for regularization
	•	Batch normalization to mitigate gradient explosion/vanishing
	•	Residual connections to improve gradient flow

I perform a grid search to optimize the model’s hyperparameters:
	•	Initial learning rate
	•	Hidden layer size
	•	Number of hidden layers

# Results

The model’s R² score suggests that it captures some indicators for future price movement.
	•	Trend prediction: The model effectively predicts price movement tendencies but struggles with exact values.
	•	Impact of major events: External factors like the pandemic influence validation and test performance.
	•	Rolling window testing: While I implemented a script for forward testing, I conducted my hyperparameter search on the full dataset, using the most recent data for validation and testing.

# Rolling Window Testing & Future Work

My initial forward testing script used a different preprocessing approach (without PCA and time series concatenation). The scripts incrementally trains the model on new data windows while reducing the learning rate over time.

Moving forward, I plan to:
	•	Refine the rolling window approach to match my latest preprocessing pipeline.
	•	Improve model generalization by fine-tuning hyperparameters on different timeframes.
	•	Analyze feature importance to better understand which indicators drive sector movements.
