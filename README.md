# NTPC Power Plant Coal Stock & Power Generation Forecasting Dashboard

A professional Streamlit web application for forecasting coal power plant stock levels and power generation using ETS (Exponential Smoothing) models.

## Project Overview

This application replicates the complete forecasting pipeline from the Jupyter Notebook into a production-ready web interface. It guides users through four sequential pages:

1. **Home** - Data upload and preview
2. **EDA** - Exploratory Data Analysis with comprehensive visualizations
3. **Forecast** - ETS model execution and forecasting
4. **Results** - Event prediction and early warning system

## Technology Stack

- **Frontend**: Streamlit ≥ 1.35
- **Language**: Python 3.10+
- **Forecasting**: statsmodels (SARIMA, ExponentialSmoothing), XGBoost, LightGBM, CatBoost, TensorFlow (LSTM)
- **Visualizations**: Plotly
- **Data Processing**: pandas, numpy, scikit-learn
- **Peak Detection**: scipy.signal.find_peaks

## Project Structure

```
ntpc_dashboard/
├── app.py                   # Main entry point
├── requirements.txt         # Dependencies
├── assets/                  # Logos (NTPC + AIonOS)
├── styles/
│   └── theme.css           # IndiGo-style CSS theme
├── pages/
│   ├── __init__.py
│   ├── home.py             # Page 1: Upload & Preview
│   ├── eda.py              # Page 2: EDA Analysis
│   ├── forecast.py         # Page 3: Forecast (Multiple Models)
│   └── results.py          # Page 4: Events & Warnings + Model Comparison
└── core/
    ├── __init__.py
    ├── preprocessing.py    # Data prep & periodogram
    ├── ets_model.py        # Base ETS model
    ├── ets_ts_model.py     # ETS Trend+Seasonal
    ├── ets_r_model.py      # ETS Residual
    ├── ets_tsr_model.py    # Final composite (TSR)
    ├── sarima_model.py     # SARIMA model
    ├── xgboost_model.py    # XGBoost model
    ├── lightgbm_model.py   # LightGBM model
    ├── catboost_model.py   # CatBoost model
    ├── lstm_model.py       # LSTM model
    └── events.py           # Event detection logic
```

## Installation & Setup

### Step 1: Install Dependencies

Navigate to the project directory and install required packages:

```bash
cd ntpc_dashboard
pip install -r requirements.txt
```

### Step 2: Add Logo Assets (Optional)

Place logo PNG files in the `assets/` folder:
- `assets/ntpc_logo.png` - NTPC logo
- `assets/aionos_logo.png` - AIonOS logo

If logos are not provided, the navbar will display without them.

### Step 3: Launch the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage Guide

### Page 1: Home (Upload Data)
- Upload an Excel file (.xlsx or .xls) with NTPC coal plant data
- Expected columns: `Date`, `Barh(Power GW)`, `Dadri Thermal(Power GW)`, `Kudgi(Power GW)`, coal stock/consumption columns
- Preview the data with optional filtering
- Download full cleaned data if needed
- Click "Go to EDA" to proceed to analytical page

### Page 2: EDA (Exploratory Data Analysis)
Comprehensive analysis including:
- **Dataset Overview**: Row/column counts, date ranges, data types
- **Data Quality**: Missing values, duplicates, type checks
- **Univariate Analysis**: Power generation statistics and distributions
- **Plant Summary**: Coal stock/consumption by plant
- **Time Series**: Full historical trends (stock vs consumption vs power)
- **Monthly Trends**: Seasonality and operational patterns
- **Peaks & Troughs**: Historical demand cycles with peak/trough detection

### Page 3: Forecast (Multiple Models)
Configure and run forecasting with your choice of model:
- **Model Selection**: Choose from 6 advanced models:
  - **ETS (Exponential Smoothing)**: Decomposed into Trend+Seasonal + Residual components
  - **SARIMA**: Seasonal AutoRegressive Integrated Moving Average with automatic order selection
  - **XGBoost**: Gradient boosting with 14-day lag features
  - **LightGBM**: Fast gradient boosting with lag features
  - **CatBoost**: Categorical boosting (all features numeric here)
  - **LSTM**: Long Short-Term Memory neural network (requires TensorFlow)
- **Plant Selection**: Choose BARH, DADRI, or KUDGI
- **Forecast Horizon**: Set 30-730 days ahead
- Displays performance metrics (RMSE, MAE, MAPE, coverage)
- Download forecast predictions as Excel file
- Navigate to Results to compare all model runs

### Page 4: Results (Event Prediction & Model Comparison)
Comprehensive analysis and comparison:
- **Trust Horizon**: Adjust the planning window (30-365 days)
- **Event Detection**: Identifies PEAK (high demand) and LOW (low supply risk) events
- **Lead Time Analysis**: Shows how far in advance events can be forecasted
- **Visualizations**:
  - Bar chart: Average lead times by plant
  - Scatter plot: Event distribution over time
- **Model Comparison Table**: Compare RMSE, MAPE, and Coverage across all models you've run
- Download actionable intelligence for coal procurement planning
- Download model comparison Excel for reporting

## Key Features

### IndiGo-Style Design
- Professional color scheme with NTPC branding
- Responsive navbar with page navigation
- Progress bar showing completion status
- Pill-style section headers

### Sequential Navigation
- Users cannot skip steps
- Each page unlocks only when prerequisites are complete
- Visual indicators (green/blue/gray) show status
- Toast notifications guide navigation

### Staged Processing
- Spinner-based UX feedback during computation
- Multi-step forecasting with status updates
- Session state caching to prevent recomputation

### Comprehensive Outputs
- Interactive Plotly charts for all visualizations
- Downloadable Excel exports for all forecasts
- Professional print-style summary tables
- Code block outputs for transparency

## Model Details

### ETS (Exponential Smoothing)
The base ETS model uses:
- **Trend**: Additive with damping
- **Seasonality**: Additive with period detection
- **Smoothing Parameters**: Optimized per plant
  - BARH: (0.002, 0.02, 0.2)
  - DADRI: (0.006, 0.06, 0.6)
  - KUDGI: (0.008, 0.08, 0.8)

### Composite Model (TSR)
Final forecast = Trend+Seasonal predictions + Residual predictions

This approach:
- Captures long-term trends and seasonality (TS)
- Models irregular variations (R)
- Provides accurate 96% confidence intervals

### Event Detection
Events identified as:
- **PEAK**: Top 2% forecasted values (high demand)
- **LOW**: Bottom 2% forecasted values (low supply risk)
- **Lead Time**: Days in advance event can be predicted

## File Format Requirements

Expected Excel file structure:

| Date | Barh(Power GW) | Dadri Thermal(Power GW) | Kudgi(Power GW) | Barh_coal_stock | ... |
|------|---------------|------------------------|-----------------|-----------------|-----|
| YYYY-MM-DD | 2.5 | 3.2 | 1.8 | 50000 | ... |

Minimum required columns: `Date`, and at least one power column per plant

## Troubleshooting

**Issue**: Logo images not displayed
- **Solution**: Place PNG files in `assets/` with exact names or ignore (optional)

**Issue**: "Please upload data first" warning
- **Solution**: Go to Home page and upload Excel file before accessing other pages

**Issue**: Slow forecast computation
- **Solution**: Use shorter forecast horizons (30-90 days) for faster runs

**Issue**: Missing column errors
- **Solution**: Ensure Excel file has correct column naming matching NTPC dataset

## Performance Tips

- Use 365-day forecasts for balanced speed/accuracy
- Run forecasts on clean, regularly-sampled data
- For large datasets, consider filtering to recent 2-3 years
- Browser caching speeds up repeated page visits

## Support & Documentation

For detailed technical implementation, refer to the Product Requirements Document (PRD) included with this package.

## Version

**Version 1.0** - NTPC × AIonOS - 2025

---

**Built with Streamlit, Plotly, and statsmodels for professional forecasting workflows.**
