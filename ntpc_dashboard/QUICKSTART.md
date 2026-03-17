# NTPC Forecasting Dashboard - Quick Start Guide

## 🚀 Quick Start (5 Minutes)

### Step 1: Navigate to Project Directory
```bash
cd ntpc_dashboard
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Launch Application
```bash
streamlit run app.py
```

The application will automatically open at `http://localhost:8501`

---

## 📋 Complete Setup Instructions

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- An Excel file with NTPC coal plant data

### Detailed Installation Steps

#### 1. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Verify Setup
```bash
python verify_setup.py
```

You should see checkmarks (✅) for all items. If any are missing, reinstall dependencies.

#### 4. Add Optional Logos
Place these files in the `assets/` directory:
- `assets/ntpc_logo.png` (NTPC company logo)
- `assets/aionos_logo.png` (AIonOS company logo)

*Note: If logos are not provided, the app will still work perfectly without them.*

---

## 🏃 Running the Application

### From Command Line
```bash
streamlit run app.py
```

### Typical Output
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Click on the local URL or manually navigate to `http://localhost:8501`

---

## 📊 Using the Application

### Workflow Overview

1. **Home Page (Upload)**
   - Upload your NTPC Excel file
   - Preview data with optional filtering
   - Once uploaded, "EDA" link unlocks

2. **EDA Page (Analyze)**
   - Explore 7 sections of analysis
   - Understand data patterns and seasonality
   - Once completed, "Forecast" link unlocks

3. **Forecast Page (Model)**
   - Select plant (BARH, DADRI, KUDGI)
   - Configure forecast horizon (30-730 days)
   - Run ETS models (staged with spinners)
   - Download forecast data as Excel
   - Once complete, "Results" link unlocks

4. **Results Page (Events)**
   - Set trust horizon for planning
   - Detect PEAK and LOW demand events
   - Visualize lead times and event distribution
   - Export actionable insights

### Expected Data Format

Your Excel file should have columns like:

```
Date | Barh(Power GW) | Dadri Thermal(Power GW) | Kudgi(Power GW) | Barh_coal_stock | Barh_coal_consumption | ...
```

**Date** column should be in YYYY-MM-DD or standard date format.

---

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'streamlit'`
**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: `FileNotFoundError: [Errno 2] No such file or directory: 'styles/theme.css'`
**Solution:** Make sure you're running the command from inside the `ntpc_dashboard/` directory.
```bash
cd ntpc_dashboard
streamlit run app.py
```

### Issue: "Please upload data first" on EDA page
**Solution:** Go back to Home page and upload the Excel file. The file must contain columns with exact names like `Barh(Power GW)`.

### Issue: Forecast takes too long
**Solution:** Try using 90-day forecast instead of 365-day. The model computation time increases with the horizon.

### Issue: Logos not showing in navbar
**Solution:** This is OK! The app works fine without logos. Optionally, add PNG files to `assets/` folder.

### Issue: Port 8501 already in use
**Solution:** 
```bash
streamlit run app.py --server.port 8502
```

---

## 🎯 Key Features by Page

### Home Page
- ✅ Drag-and-drop file upload
- ✅ Top-10 data preview
- ✅ Column filtering
- ✅ Download full cleaned data

### EDA Page
- ✅ Dataset overview (rows, columns, date range)
- ✅ Data quality assessment (missing, duplicates)
- ✅ Univariate statistics (mean, std, distributions)
- ✅ Plant-wise coal stock/consumption summary
- ✅ Full time series visualization (3 plants)
- ✅ Monthly aggregated trends
- ✅ Historical peaks and troughs detection

### Forecast Page
- ✅ Model selection (ETS)
- ✅ Plant selection (BARH, DADRI, KUDGI)
- ✅ Configurable forecast horizon (30-730 days)
- ✅ Staged ETS modeling (TS + R → TSR)
- ✅ Performance metrics (RMSE, MAE, coverage)
- ✅ Interactive forecast chart
- ✅ Downloadable forecast data

### Results Page
- ✅ Event detection (PEAK/LOW)
- ✅ Lead time analysis
- ✅ Trust horizon configuration
- ✅ Summary tables
- ✅ Bar chart (average lead times)
- ✅ Scatter plot (event distribution)

---

## 📁 Project Files Reference

```
ntpc_dashboard/
├── app.py ........................... Main app router
├── requirements.txt ................. Python dependencies
├── README.md ........................ Full documentation
├── verify_setup.py .................. Setup checker script
├── .gitignore ....................... Git configuration
│
├── core/
│   ├── preprocessing.py ............ Data preparation
│   ├── ets_model.py ................ Base ETS model
│   ├── ets_ts_model.py ............ ETS Trend+Seasonal
│   ├── ets_r_model.py ............ ETS Residual
│   ├── ets_tsr_model.py .......... Final TSR composite
│   └── events.py ................... Event detection
│
├── pages/
│   ├── home.py .................... Upload page
│   ├── eda.py ..................... Analysis page
│   ├── forecast.py ................ Forecasting page
│   └── results.py ................. Results page
│
├── styles/
│   └── theme.css .................. IndiGo theme
│
└── assets/
    ├── ntpc_logo.png .............. (Optional) NTPC logo
    └── aionos_logo.png ............ (Optional) AIonOS logo
```

---

## 🔧 Environment Variables (Optional)

You can set these for advanced configurations:

```bash
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_MAXUPLOADSIZE=200
export STREAMLIT_CLIENT_SHOWSTDERR=true
```

---

## 📈 Performance Optimization Tips

1. **Use 90-365 day forecasts** - Balances accuracy with speed
2. **Data preprocessing** - Ensure your Excel file has no gaps
3. **Browser caching** - Repeated visits are faster
4. **Smaller datasets** - Consider filtering to recent 2-3 years of data
5. **Close other applications** - Free up system memory

---

## 🎓 Learning Resources

For deeper understanding:
- **ETS Models**: https://en.wikipedia.org/wiki/Exponential_smoothing
- **Seasonal Decomposition**: https://otexts.com/fpp2/stl.html
- **Streamlit Docs**: https://docs.streamlit.io
- **Plotly Charts**: https://plotly.com/python/

---

## 📞 Support

If you encounter issues:
1. Check the **Troubleshooting** section above
2. Verify setup with: `python verify_setup.py`
3. Check logfile output in terminal for error messages
4. Ensure Python 3.10+ is installed: `python --version`

---

## ✅ Verification Checklist

Before running the app, verify:

- [ ] All files are present (run `python verify_setup.py`)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] You have a valid Excel file with plant data
- [ ] You're in the `ntpc_dashboard/` directory
- [ ] Port 8501 is available (or use different `--server.port`)

---

## 🚀 Ready to Go!

```bash
cd ntpc_dashboard
streamlit run app.py
```

**Enjoy your forecasting dashboard!** 📊⚡

---

**NTPC Forecasting Dashboard v1.0**  
Built with Streamlit, Plotly & statsmodels  
NTPC × AIonOS - 2025
