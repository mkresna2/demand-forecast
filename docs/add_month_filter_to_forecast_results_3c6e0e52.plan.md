---
name: Add month filter to forecast results
overview: Add a month filter dropdown in the Demand Forecast tab to allow users to filter and view results for a specific month within the forecast period. The existing 'Forecast through month' setting in the sidebar will remain unchanged.
todos:
  - id: "1"
    content: Add month filter dropdown in Demand Forecast tab after the section header
    status: completed
  - id: "2"
    content: Create filtered dataframe based on selected month
    status: completed
  - id: "3"
    content: Update revenue charts to use filtered data
    status: completed
  - id: "4"
    content: Update occupancy charts to use filtered data
    status: completed
  - id: "5"
    content: Update historical context chart to use filtered data
    status: completed
  - id: "6"
    content: Update room type charts to use filtered data
    status: completed
  - id: "7"
    content: Update Daily Forecast Detail table to use filtered data
    status: completed
  - id: "8"
    content: Update download button to use filtered data
    status: completed
isProject: false
---

**Goal:** Add a month filter to the forecast results view so users can focus on a specific month's data.

**Current State:**

- The sidebar has "Forecast through month" (line 1101) which controls the forecast horizon (how many days to forecast)
- The Demand Forecast tab (`tab_fcast`) shows all forecast data from tomorrow through the end of the selected forecast month
- Data is displayed in charts and the "Daily Forecast Detail" table (line 1522)

**Implementation Plan:**

1. **Add Month Filter UI in Results Section** (after line 1330, inside `tab_fcast`):
  - Add a dropdown to select which month's results to view
  - Extract unique months from the `fcast` dataframe's `stay_date` column
  - Include an "All Months" option to show the full forecast period
2. **Filter Logic**:
  - Store the selected month filter in a variable
  - When a specific month is selected, filter the `fcast` dataframe to include only dates within that month
  - When "All Months" is selected, use the full `fcast` dataframe
3. **Update All Display Components to Use Filtered Data**:
  - Revenue charts (lines 1337-1381)
  - Occupancy charts (lines 1384-1440)
  - Historical context chart (lines 1443-1484)
  - Room type charts (lines 1487-1519)
  - Daily Forecast Detail table (lines 1521-1540)
  - Insights section (lines 1550+) - this should continue to use full data or be filtered based on selection
4. **Key Code Changes in `[hotel_demand_forecast.py](hotel_demand_forecast.py)`**:
  - After line 1330 (inside `tab_fcast`), add month filter dropdown
  - Create filtered dataframe based on selection
  - Replace references to `fcast` with the filtered dataframe in chart and table code

**Example of the change:**
After line 1330, add:

```python
# Month filter for results view
months_in_forecast = sorted(fcast["stay_date"].dt.to_period("M").unique())
month_labels = ["All Months"] + [str(m) for m in months_in_forecast]
selected_view_month = st.selectbox("Filter results by month", month_labels, index=0)

if selected_view_month != "All Months":
    selected_period = pd.Period(selected_view_month)
    fcast_filtered = fcast[fcast["stay_date"].dt.to_period("M") == selected_period].copy()
else:
    fcast_filtered = fcast.copy()
```

Then use `fcast_filtered` in all chart and table displays.