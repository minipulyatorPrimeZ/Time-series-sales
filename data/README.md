# Retail Store Sales Forecasting Dataset

## Forecasting Weekly Retail Sales Across Stores, Departments & Holiday Events

Retail forecasting is one of the most important challenges in modern commerce.

Retailers must continuously predict demand, optimize inventory, plan promotions, and understand how external factors such as holidays, markdowns, fuel prices, and economic conditions influence sales performance.

This dataset provides a multi-table retail forecasting environment designed for:

- time series forecasting
- sales prediction
- markdown impact analysis
- retail business intelligence
- machine learning experimentation
- demand forecasting

---

## Dataset Structure

The dataset contains 3 related files:

| File | Description |
|---|---|
| sales.csv | Historical weekly sales across stores and departments |
| stores.csv | Store metadata including type, size, and region |
| features.csv | Economic indicators, markdowns, holidays, and seasonal variables |

---

## sales.csv

- store_id: Unique store identifier
- department: Store department identifier
- date: Weekly sales date
- weekly_sales: Weekly sales revenue
- is_holiday: Indicates holiday week

---

## stores.csv

- store_id: Unique store identifier
- store_type: Store category/type
- store_size: Physical store size
- region: Geographic region

---

## features.csv

- store_id: Unique store identifier
- date: Weekly date
- temperature: Regional temperature
- fuel_price: Fuel price indicator
- markdown_1 to markdown_5: Promotional markdown values
- cpi: Consumer Price Index
- unemployment: Unemployment rate
- is_holiday: Holiday indicator
- holiday_name: Holiday event label
- season: Seasonal grouping

---

## Use Cases

- Time series forecasting
- Machine learning sales prediction
- Retail demand forecasting
- Markdown effectiveness analysis
- Holiday impact analysis
- Store-level sales analytics
- Inventory planning
- Business intelligence dashboards

---

## Ideal For

- XGBoost forecasting
- Prophet forecasting
- LSTM forecasting
- Random Forest regression
- Feature engineering projects
- Exploratory data analysis
- Kaggle notebooks and competitions

---

## Author

Noopur Bhatt
