# 📦 Smart Demand Prediction Dashboard

A Streamlit app for retail demand forecasting powered by XGBoost.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
cd dashboard
streamlit run app.py
```
The app opens at **http://localhost:8501**

---

## Features

| Tab | What you get |
|-----|-------------|
| 🔮 Prediction & Optimiser | Live High/Low demand prediction + confidence gauge + price sweep to find the revenue-optimal price |
| 📊 Data Insights (EDA) | Price distribution, category breakdown, scatter plots, competitor price boxplot, correlation heatmap |
| 🏆 Model Metrics | Accuracy / ROC-AUC / Precision / Recall, feature importance bar chart, confusion matrix, ROC curve |

## Sidebar Inputs

- **Your Unit Price** – the price you're considering
- **Competitor Prices** (1–3) – current market prices
- **Freight Price** – shipping cost
- **Product Score** – rating (3–5 stars)
- **Customers** – number of customers in the period
- **Category** – product category
- **Month / Day of Week** – seasonality inputs

## Project Structure
```
capstone_geak/
├── dashboard/
│   └── app.py          ← Streamlit app (edit this)
├── data/
│   ├── retail_price.csv
│   ├── cleaned_data.csv
│   └── feature_engineered_data.csv
├── model/
│   ├── trained_model.pkl
│   └── columns.pkl
├── notebooks/
│   └── *.ipynb
├── requirements.txt
└── README.md
```
