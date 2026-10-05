import pandas as pd
import streamlit as st
from retirement_model import LTCI_PRICE, run_retirement_plan

st.set_page_config(page_title="Retirement Income Planner", page_icon="📊", layout="wide")
st.title("Retirement Income Planner")
st.caption("Interactive retirement-income projection based on the uploaded notebook model")

with st.sidebar:
    st.header("Inputs")
    w0 = st.number_input("Initial wealth W0 (AUD)", min_value=110000.0, value=407328.0, step=10000.0, format="%.2f")
    sims = st.select_slider("Monte Carlo simulations", options=[50, 100, 200, 500, 1000], value=200)
    seed = st.number_input("Base random seed", min_value=0, value=2026, step=1)
    st.info(f"LTCI price is fixed at A${LTCI_PRICE:,.2f}.  omega_L = 81,153.88 / W0.")
    run = st.button("Run retirement plan", type="primary", use_container_width=True)

st.subheader("Model setup")
omega = LTCI_PRICE / w0
c1,c2,c3=st.columns(3)
c1.metric("Initial wealth", f"A${w0:,.2f}")
c2.metric("LTCI price", f"A${LTCI_PRICE:,.2f}")
c3.metric("LTCI allocation omega_L", f"{omega:.2%}")

if run:
    with st.spinner("Running simulations..."):
        try:
            result = run_retirement_plan(w0, sims, seed, years=33)
        except ValueError as e:
            st.error(str(e)); st.stop()
    df=pd.DataFrame(result["rows"])
    money_cols=[c for c in df.columns if "AUD" in c]
    df[money_cols]=df[money_cols].round(2)
    st.success("Simulation complete")
    st.subheader("Strategy 1: yearly mean monthly withdrawal plan")
    main=df[["Year","Age","Suggested monthly withdrawal (AUD)","Mean monthly Age Pension (AUD)","Mean total monthly income (AUD)"]].copy()
    st.dataframe(main, use_container_width=True, hide_index=True)
    st.line_chart(main.set_index("Age")[["Suggested monthly withdrawal (AUD)","Mean monthly Age Pension (AUD)","Mean total monthly income (AUD)"]])
    st.download_button("Download results as CSV", df.to_csv(index=False).encode("utf-8-sig"), "retirement_plan.csv", "text/csv")
    with st.expander("Compare all three strategies"):
        st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("Change W0 if needed, then click ‘Run retirement plan’.")

st.caption("For research/demo use. Results are stochastic model projections, not personal financial advice.")
