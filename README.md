# Retirement Income Planner

A Streamlit web app converted from the supplied `Untitled52.ipynb` retirement simulation model.

## What the app does

- Lets the user change initial wealth `W0`.
- Uses a fixed LTCI price of **AUD 81,153.88**.
- Recalculates the LTCI allocation as `omega_L = 81,153.88 / W0`.
- Keeps the notebook's GMDB allocation ratio (`omega_G = 0.20`) and core market, health-state, withdrawal and Age Pension logic.
- Runs Monte Carlo simulations and reports, for each year/age, the mean monthly suggested withdrawal, mean monthly Age Pension, and their sum for Strategy 1.
- Provides the other two notebook strategies in an expandable comparison table.
- Allows CSV download.

## Files

- `streamlit_app.py` — web interface.
- `retirement_model.py` — model logic adapted from the notebook.
- `requirements.txt` — Python dependencies.
- `README.md` — deployment instructions.

## Deploy on Streamlit Community Cloud

1. Upload these four files to the root of your GitHub repository `retirement-income-planner`.
2. Open Streamlit Community Cloud and create a new app.
3. Select this repository and branch.
4. Set the main file path to `streamlit_app.py`.
5. Click **Deploy**.

## Notes

The yearly table uses the same positive-value conditional-mean convention as the original notebook's multi-simulation aggregation. Age Pension is also summarized separately, and `total monthly income` is shown as withdrawal + Age Pension.

For faster public demos, start with 100–200 simulations. Larger simulation counts take longer on free cloud resources.

This application is intended for research/demonstration and is not personal financial advice.
