# NTPC Forecasting Dashboard - Testing Instructions

## 🧪 Pre-Deployment Testing Guide

This document outlines how to verify that the NTPC Forecasting Dashboard is working correctly before deploying to production.

---

## ✅ Step 1: Setup Verification

### Run Automated Verification
```bash
cd ntpc_dashboard
python verify_setup.py
```

### Expected Output
```
✅ Checking directory structure...
  ✅ pages/
  ✅ core/
  ✅ styles/
  ✅ assets/

✅ Checking Python/config files...
  ✅ app.py (2150 bytes)
  ✅ requirements.txt (185 bytes)
  ... [all files should have ✅]

✅ Checking Python dependencies...
  ✅ streamlit
  ✅ pandas
  ✅ numpy
  ✅ plotly
  ✅ statsmodels
  ✅ sklearn
  ✅ scipy
  ✅ openpyxl

✅ SETUP VERIFICATION PASSED!
```

**If you see ❌ marks**, run:
```bash
pip install -r requirements.txt
```

---

## ✅ Step 2: Application Launch

### Start Streamlit
```bash
streamlit run app.py
```

### Expected Output
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501

  Folder: C:\Users\...\ntpc_dashboard
```

✅ Browser automatically opens to `http://localhost:8501`

---

## ✅ Step 3: Visual Verification

### Home Page Appearance
- [ ] NTPC Forecasting Dashboard title visible
- [ ] Four navigation links in navbar (Home, EDA, Forecast, Results)
- [ ] EDA, Forecast, Results links appear **locked** (faded)
- [ ] Progress bar shows: 1 Active (blue), 2-4 Pending (gray)
- [ ] Upload area shows with folder icon
- [ ] "Accepted formats: .xlsx / .xls" message visible

### IndiGo Theme Check
- [ ] Primary blue color (#001B94) visible in navbar and buttons
- [ ] Typography is clean and professional
- [ ] Buttons have hover effects (darken on hover)
- [ ] Cards have subtle shadows
- [ ] Overall layout is responsive and centered

---

## ✅ Step 4: Functional Testing

### Test 1: File Upload & Preview

1. **Prepare Test Data**
   - Create a sample Excel file OR
   - Use existing Coal_Plant__Forecasting.ipynb output
   
   Required columns:
   ```
   Date | Barh(Power GW) | Dadri Thermal(Power GW) | Kudgi(Power GW)
   | Barh_coal_stock | Dadri_coal_stock | Kudgi_coal_stock
   ```

2. **Upload Test**
   - [ ] Click upload area or drag-drop file
   - [ ] Success message appears: "✅ File loaded — X rows × X columns"
   - [ ] Data preview table shows top 10 rows
   - [ ] Column headers appear in table
   - [ ] Download button is clickable
   - [ ] Filter works (select column, enter value)

3. **Navigation Lock Test**
   - [ ] Click on "📊 EDA" link → should unlock (turn bright)
   - [ ] Progress bar updates: Step 1 = Green (done)
   - [ ] "Go to EDA" button is clickable

**Expected Result**: ✅ All checks pass

---

### Test 2: EDA Page

1. **Navigate to EDA**
   - [ ] Click "Go to EDA" button
   - [ ] Page switches to EDA
   - [ ] Progress bar shows: 1 Green (done), 2 Blue (active), 3-4 Gray (locked)

2. **Section 1: Dataset Overview**
   - [ ] Shows Total Rows, Columns, Date From, Date To
   - [ ] Feature table appears with columns, dtypes, samples
   - [ ] Example: "Date | dtype=datetime64[ns] | 2023-01-01"

3. **Section 2: Data Quality**
   - [ ] Missing values table appears
   - [ ] Duplicate rows count shown (blue info box)
   - [ ] Date type check passes (green success message)

4. **Section 3: Univariate Analysis**
   - [ ] Statistics table appears (mean, std, min, max, etc.)
   - [ ] Histogram chart loads (should see bars)
   - [ ] Shows stats like "Mean=2.450 GW | Std=0.823 GW"

5. **Section 4: Plant Summary**
   - [ ] Code block output appears with plant statistics
   - [ ] Table shows Stock Mean, Max, Min, Cons Mean per plant
   - [ ] Numerical values are reasonable (not NaN)

6. **Section 5: Full Time Series**
   - [ ] 3-subplot Plotly chart loads
   - [ ] Each subplot shows Stock (blue), Consumption (orange), Power (green)
   - [ ] X-axis shows dates, Y-axis shows values
   - [ ] Legend shows all three series

7. **Section 6: Monthly Aggregated**
   - [ ] 3 separate Plotly charts (one per plant)
   - [ ] Lines plot with markers
   - [ ] Dual Y-axes (Stock on left, Consumption/Power on right)
   - [ ] Title shows plant name and "Monthly Averages"

8. **Section 7: Peaks & Troughs**
   - [ ] 3-subplot chart with scatter overlays
   - [ ] Green dots = peaks, Red dots = troughs
   - [ ] Code block shows statistics (count, top-5, seasonal months)
   - [ ] Example: "Peak-heavy months: ['Jun', 'Jul', 'Aug']"

9. **Navigation**
   - [ ] "Go to Forecast" button appears at bottom
   - [ ] "Forecast" link in navbar is clickable (unlocks)

**Expected Result**: ✅ All 7 sections load without errors

---

### Test 3: Forecast Page

1. **Navigate to Forecast**
   - [ ] Click "Go to Forecast" button
   - [ ] Progress bar: 1-2 Green, 3 Blue (active), 4 Gray (locked)
   - [ ] Configuration panel appears with 3 columns

2. **Configuration Panel**
   - [ ] Model selector shows "ETS (Exponential Smoothing)"
   - [ ] Plant selector dropdown has: BARH, DADRI, KUDGI
   - [ ] Forecast horizon slider ranges 30-730, default 365
   - [ ] "Run Forecast" button is visible (type='primary')

3. **Run Forecast (BARH, 365 days)**
   - [ ] Select plant: BARH
   - [ ] Set horizon: 365
   - [ ] Click "Run Forecast"
   
   **Watch for spinners:**
   - [ ] "⚙️  Preparing data..." appears
   - [ ] "📡  Dominant period detected: X days" (info box)
   - [ ] "🔢  Fitting ETS on Trend+Seasonal..." appears
   - [ ] "🔢  Fitting ETS on Residual..." appears
   - [ ] "🔗  Combining TSR composite..." appears
   - [ ] "✅ Forecast complete!" success message

4. **Results Display**
   - [ ] Metrics section shows:
     - RMSE (Validation): ~0.XXXX
     - MAE (Validation): ~0.XXXX
     - Confidence Coverage: ~98.XX%
   - [ ] TSR Composite plot renders
     - Black line = train data
     - Gray line = test data
     - Blue line = validation
     - Red line = forecast
     - Green shaded bands = 96% CI
   - [ ] Chart title: "BARH — Final ETS (TSR Composite Forecast)"
   - [ ] Forecast data preview table (10 rows)
   - [ ] Column name: "ps_barh__FORECAST"
   - [ ] Download button: "⬇️  Download Full Forecast Data"

5. **Test Download**
   - [ ] Click download button
   - [ ] File downloads as "forecast_barh_365days.xlsx"
   - [ ] File opens in Excel (check for data)

6. **Navigation**
   - [ ] "Results" link in navbar unlocks (becomes clickable)
   - [ ] "Go to Results" button appears at bottom

**Expected Result**: ✅ Forecast runs without errors, metrics are reasonable

---

### Test 4: Results Page

1. **Navigate to Results**
   - [ ] Click "Go to Results" button
   - [ ] Progress bar: 1-3 Green (done), 4 Blue (active)
   - [ ] All 4 steps now show as "done"

2. **Trust Horizon Configuration**
   - [ ] Slider visible: 30-365 days, default 90
   - [ ] Shows: "Business Trust Horizon (days)"
   - [ ] "⚡ Analyse Events" button appears

3. **Run Event Analysis**
   - [ ] Set trust horizon to 90
   - [ ] Click "Analyse Events"
   - [ ] Spinner: "🔍  Detecting peak and low events..."
   - [ ] Success: "✅ Early warning analysis complete!"

4. **Output Sections**
   - [ ] Code block with print-style output:
     - "First Event Type: PEAK" or "LOW"
     - "Event Date: YYYY-MM-DD"
     - "Lead Time: X days"

5. **Early Warning Summary Table**
   - [ ] Table appears with columns:
     - Plant | Max Lead (days) | Avg Lead (days) | Total Events
   - [ ] Shows data for BARH, DADRI, KUDGI
   - [ ] Numbers are positive integers or 0

6. **Trusted Events Table**
   - [ ] Shows events within 90-day horizon
   - [ ] Columns: Plant | Trusted Events | Avg Lead (days) | Horizon (days)
   - [ ] Example: "BARH | 5 | 45.2 | 90"

7. **Bar Chart**
   - [ ] Title: "Average Early-Warning Lead Time..."
   - [ ] 3 bars (one per plant)
   - [ ] X-axis: Plant names (BARH, DADRI, KUDGI)
   - [ ] Y-axis: Lead time in days
   - [ ] Values displayed above bars

8. **Scatter Plot**
   - [ ] Title: "Distribution of Predicted PEAK / LOW Events"
   - [ ] Green dots = PEAK events
   - [ ] Red dots = LOW events
   - [ ] Y-axis: Plant names
   - [ ] X-axis: Days ahead
   - [ ] Vertical dashed line at trust horizon (90 days)
   - [ ] Legend shows event types

**Expected Result**: ✅ All event visualizations render correctly

---

### Test 5: Multi-Plant Testing

**Repeat Test 3-4 with different plants:**

1. **Test with DADRI**
   - [ ] Run forecast with DADRI, 180 days
   - [ ] Should show different metrics than BARH
   - [ ] Chart shows different series

2. **Test with KUDGI**
   - [ ] Run forecast with KUDGI, 90 days
   - [ ] Should complete faster (shorter horizon)
   - [ ] Verify metrics are reasonable

**Expected Result**: ✅ Different forecasts for different plants

---

### Test 6: Navigation & Locking

1. **Go Back to Home**
   - [ ] Click "🏠 Home" in navbar
   - [ ] Returns to Home page
   - [ ] Upload area still has data file
   - [ ] EDA-Results links still accessible

2. **Test Forward Navigation**
   - [ ] From Home, try clicking "Forecast" (locked) → should warn
   - [ ] Try clicking "Results" (locked) → should warn
   - [ ] Must go through EDA first to unlock Forecast

**Expected Result**: ✅ Navigation properly gated

---

## 🐛 Error Handling Tests

### Test 7: Invalid File Upload

1. **Upload non-Excel file**
   - [ ] Select .txt or .csv file
   - [ ] Should show rejection (only .xlsx/.xls accepted)
   - [ ] "Accepted formats: .xlsx / .xls" message visible

### Test 8: Missing Required Columns

1. **Upload Excel without Power columns**
   - [ ] File uploads but shows partial data
   - [ ] EDA should skip missing column sections gracefully
   - [ ] No crash (should handle missing columns with 'if col in df')

### Test 9: Port Already in Use

1. **Run app on already-used port**
   - [ ] Terminal shows: "Address already in use"
   - [ ] Use `streamlit run app.py --server.port 8502`
   - [ ] App launches on port 8502

**Expected Result**: ✅ Graceful error handling

---

## 📊 Performance Tests

### Test 10: Page Load Times

- [ ] Home page loads in < 1 sec
- [ ] EDA loads and renders charts in < 5 sec
- [ ] Forecast computation takes 10-15 sec (depends on hardware)
- [ ] Results loads in < 2 sec

### Test 11: Large File Handling

1. **Upload large file** (5000+ rows)
   - [ ] File uploads successfully
   - [ ] EDA processes without timeout
   - [ ] Forecast completes (may take longer)

**Expected Result**: ✅ Handles large datasets gracefully

---

## 🎨 UI/UX Tests

### Test 12: Responsive Design

1. **Resize browser**
   - [ ] Navbar adjusts properly
   - [ ] Content area responsive
   - [ ] Charts reflow (not cut off)

2. **Mobile simulation** (DevTools)
   - [ ] Layout adapts to narrow width
   - [ ] Buttons remain clickable
   - [ ] Charts still visible

### Test 13: Chart Interactions

1. **Hover over Plotly chart**
   - [ ] Tooltip shows values
   - [ ] Cursor changes to pointer

2. **Zoom/Pan**
   - [ ] Click and drag on chart → zoom
   - [ ] Reset button works
   - [ ] Legend items toggle series visibility

**Expected Result**: ✅ Charts fully interactive

---

## ✅ Final Verification Checklist

- [ ] Setup verification passes (python verify_setup.py)
- [ ] App launches without errors (streamlit run app.py)
- [ ] Home page displays correctly
- [ ] File upload works with valid Excel
- [ ] EDA loads all 7 sections without errors
- [ ] Forecast runs successfully for all 3 plants
- [ ] Results page displays events and visualizations
- [ ] Download buttons work and create valid Excel files
- [ ] Navigation links lock/unlock properly
- [ ] Progress bar updates correctly
- [ ] Performance is acceptable (< 15 sec per forecast)
- [ ] No crashes or unhandled exceptions
- [ ] All charts render correctly
- [ ] IndiGo theme styling is applied consistently

---

## 🚀 Sign-Off

When all tests pass:

1. ✅ Create test file: `TEST_RESULTS.txt` with today's date
2. ✅ Document any issues found (if any)
3. ✅ Note browser and Python versions used
4. ✅ Application is ready for deployment

---

## 📋 Test Result Template

```
NTPC FORECASTING DASHBOARD - TEST RESULTS
==========================================

Test Date: YYYY-MM-DD
Tester: [Your Name]

Python Version: [output of python --version]
Browser: [Chrome/Firefox/Edge] version X.X
OS: [Windows/macOS/Linux]

Test Results:
✅ Setup verification      PASSED
✅ Application launch      PASSED
✅ Home page               PASSED
✅ File upload             PASSED
✅ EDA processing          PASSED
✅ Forecast execution      PASSED
✅ Results analysis        PASSED
✅ Navigation gating       PASSED
✅ Download functionality  PASSED
✅ Error handling          PASSED
✅ Performance             PASSED
✅ UI/UX                   PASSED

Issues Found: None

Status: ✅ READY FOR DEPLOYMENT

Signature: _________________________ Date: _________
```

---

## 🎓 Notes

- Tests should be run in this order (dependencies)
- If any test fails, fix the issue and rerun from that point
- Document any bugs in `TEST_RESULTS.txt`
- Performance varies by hardware (laptop vs server)
- Excel files with >10K rows may take longer to process

---

**Happy Testing! 🎉**

For any issues, refer to QUICKSTART.md troubleshooting section.

---

*NTPC Forecasting Dashboard v1.0 - Testing Guide*
