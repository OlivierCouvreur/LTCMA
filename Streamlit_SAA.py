"""
Strategic Asset Allocation (SAA) Portfolio Monte Carlo Simulator
================================================================

Description:
------------
This Streamlit application provides a comprehensive toolkit for financial analysts and portfolio managers 
to perform Monte Carlo simulations, optimization, efficient frontier analysis, scenario analysis, and risk assessments 
on Strategic Asset Allocation (SAA) portfolios.

The application leverages user-defined Long-Term Capital Market Assumptions (LTCMA) and correlation matrices 
to generate probabilistic projections of portfolio outcomes, evaluate risk metrics such as Value at Risk (VaR) 
and Conditional VaR (CVaR), and optimize portfolio weights based on risk-return objectives.

Key Features:
------------
- Monte Carlo Simulation with fat-tail distributions (Student-t)
- Portfolio Optimization (Max Return / Min Volatility)
- Efficient Frontier Visualization
- Capital Market Line (CML) calculation
- Historical Scenario Analysis
- Drawdown and Recovery Analysis
- Interactive and dynamic user interface
- Session saving and loading for reproducibility

Inputs:
-------
- **LTCMA Data**:
  - Expected Returns, Expected Volatility, Initial Asset Weights (SAA), and Bounds (Min, Max)
- **Correlation Matrix**:
  - Asset correlations reflecting diversification effects
- **Simulation Parameters**:
  - Initial Portfolio Value, Investment Horizon, Frequency, Number of Simulations, VaR settings
- **Historical Scenario Data**:
  - Excel-based scenario shocks for stress testing

Outputs:
--------
- **Portfolio Simulations**:
  - Distribution plots (Portfolio Value, Final Value)
  - Summary statistics (Mean, Median, Percentiles)
- **Optimization Results**:
  - Optimal asset weights, expected return, volatility
  - Efficient Frontier and Capital Market Line plots
- **Risk Metrics**:
  - VaR, CVaR, Drawdown statistics
- **Scenario Analysis**:
  - Scenario-based portfolio return impacts and contributions

Usage:
------
Run the application using Streamlit:
```
streamlit run Streamlit_SAA.py
```

Dependencies:
-------------
- Streamlit
- NumPy
- Pandas
- SciPy
- Matplotlib


IDEAS / SUGGESTIONS / TO-DO NEXT:
v4.x
- Export portfolio paths to XL, not CSV --  DONE v4.11.5
- change the layout to put all things related to the session into the sidebar --  DONE  v4.11.2
- Toggle the Y axis for simulations between Linear and Log ?  -- DONE v4.12.0

v5.x
- incorporate the historical simulation part, meaning we need an interface to upload historical returns
    this means finding a way to circumvent / or defining what to do when data is missing
- potentially, change the interface to allow two portfolio comparison, sort of before/after
- prepare a slim or stripped down version for demo and/or Retail clients
- version multi langues?
- Input table: add the possibility of grouping Assets by Group (Asset Classes or Liquid/Illiquid, or other?)
            + add the ability to put min/max on Groups
- revamp the "double(s)' target in a more general way, like
        set a target, define if static or linked to inflate, and give options to *2, *3, *X, Set Value
        maybe even give a timeline, like a big upcoming Cash Flow, to get back the proba to achieve it
        maybe, in optimizer, set this as a strict constraint (i need to reach this by then, with certainty / (or X% proba?) )


Author:
-------
[Olivier Couvreur]
[04/08/2025]

"""

# SAA Portfolio Monte Carlo Simulator
# ===================================

import streamlit as st
import pandas as pd
import numpy as np
#import sys
import matplotlib
matplotlib.use('Agg')  # explicitly set backend
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from io import BytesIO
from scipy.optimize import minimize
from scipy.stats import t
from pathlib import Path  # For safer file suffix handling
import pickle
import base64


APP_VERSION = "v4.12.0"

# ---- default data files (edit paths as needed) ----
DEFAULT_LTCMA_PATH = "Data/LTCMA.xlsx"
DEFAULT_CORR_PATH = "Data/Correlation Matrix.xlsx"
DEFAULT_SCENARIO_PATH = "Data/Scenarios.xlsx"



# Helper to detect DataFrame changes
#def df_hash(df: pd.DataFrame) -> int:
#    """Return an integer hash for a DataFrame."""
#    return int(pd.util.hash_pandas_object(df, index=True).sum())

# Ensure the correlation matrix has the proper shape
def ensure_corr_shape(assets: pd.Index, corr_df: pd.DataFrame | None) -> pd.DataFrame:
    if corr_df is None or getattr(corr_df, "empty", True):
        return pd.DataFrame(np.eye(len(assets)), index=assets, columns=assets)
    out = corr_df.reindex(index=assets, columns=assets)
    out = out.fillna(0.0)
    np.fill_diagonal(out.values, 1.0)
    return out


st.set_page_config(layout="wide")
st.title("SAA Portfolio Monte Carlo Simulator")

# Sidebar Inputs

# Initialize default session state
default_values = {
    "start_date": pd.to_datetime("2025-01-01"),
    "n_years": 10,
    "initial_value": 100.0,
    "frequency": "monthly",
    "n_sims": 2000,
    "use_optimized_weights": False
}

for key, val in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = val

if "simulation_has_run" not in st.session_state:
    st.session_state.simulation_has_run = False

# Additional defensive session state initialization
if "portfolio_paths" not in st.session_state:
    st.session_state["portfolio_paths"] = None

if "x_axis" not in st.session_state:
    st.session_state["x_axis"] = None

# v4.3 ----------
st.session_state.simulation_has_run = (
    st.session_state["portfolio_paths"] is not None
    and st.session_state["x_axis"] is not None
)
# v4.3 ----------


def get_current_ltcma_and_corr_for_save() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Try most recent edited table, then base default, then disk, then a tiny inline fallback
    ltc = st.session_state.get("prev_ltcma_df")
    if ltc is None:
        ltc = st.session_state.get("ltcma_base_default")
    if ltc is None:
        try:
            ltc = pd.read_excel(DEFAULT_LTCMA_PATH, index_col=0)
        except Exception:
            ltc = pd.DataFrame({
                "Exp Return": [0.06, 0.03, 0.08],
                "Exp Volatility": [0.10, 0.05, 0.15],
                "SAA": [0.5, 0.3, 0.2],
                "Min": [0.0, 0.0, 0.0],
                "Max": [1.0, 1.0, 1.0]
            }, index=["Equities", "Bonds", "Alternatives"])

    # light hygiene
    ltc = ltc.copy()
    ltc.index = ltc.index.astype(str).str.strip()
    ltc = ltc.loc[ltc.index != ""]

    # Corr uses whatever is in the store, reshaped to match the LTCMA index
    corr = ensure_corr_shape(ltc.index, st.session_state.get("corr_store"))
    return ltc, corr


# Sidebar Inputs referencing session_state directly
st.sidebar.header("Simulation Parameters")

with st.sidebar.expander("Session Management"):
    uploaded_session = st.file_uploader("Reload Saved Session", type=["pkl"], key="session_uploader")
    c1, c2 = st.columns(2)
 
    with c1:
        if st.button("💾 Save Session", key="save_session_main"):
            ltcma_current, corr_current = get_current_ltcma_and_corr_for_save()

            # Make optimized_weights storable & aligned (Series or None)
            opt = st.session_state.get("optimized_weights", None)
            if isinstance(opt, pd.Series):
                opt_to_save = opt.reindex(ltcma_current.index)
            elif isinstance(opt, np.ndarray) and len(opt) == len(ltcma_current.index):
                opt_to_save = pd.Series(opt, index=ltcma_current.index)
            else:
                opt_to_save = None

            session_data = {
                "ltcma_df": ltcma_current.copy(),
                "corr_matrix": corr_current.copy(),
                "optimized_weights": opt_to_save,
                "sim_params": {
                    "start_date": st.session_state["start_date"],
                    "n_years": st.session_state["n_years"],
                    "initial_value": st.session_state["initial_value"],
                    "frequency": st.session_state["frequency"],
                    "n_sims": st.session_state["n_sims"],
                    "use_optimized_weights": st.session_state["use_optimized_weights"],
                },
            }
            buffer = BytesIO()
            pickle.dump(session_data, buffer)
            b64 = base64.b64encode(buffer.getvalue()).decode()
            st.markdown(
                f'<a href="data:file/octet-stream;base64,{b64}" download="saa_session.pkl">Download Session File</a>',
                unsafe_allow_html=True
            )

    with c2:
        if st.button("↩️ Restore Defaults", key="restore_defaults", help="Load default LTCMA, Correlation, and Scenarios"):
            try:
                ltcma_default = pd.read_excel(DEFAULT_LTCMA_PATH, index_col=0)
                corr_default  = pd.read_excel(DEFAULT_CORR_PATH, index_col=0)
                scenarios_default = pd.read_excel(DEFAULT_SCENARIO_PATH)

                corr_default = corr_default.reindex(index=ltcma_default.index, columns=ltcma_default.index).fillna(0.0)
                np.fill_diagonal(corr_default.values, 1.0)

                st.session_state["ltcma_base_default"] = ltcma_default.copy()
                st.session_state["corr_store"] = corr_default.copy()
                st.session_state["corr_assets"] = tuple(ltcma_default.index.tolist())
                st.session_state["default_scenarios_df"] = scenarios_default.copy()

                # force editors to rebuild + clear derived state
                st.session_state.pop("ltcma_widget", None)
                st.session_state.pop("corr_widget", None)
                for k in ["portfolio_paths", "x_axis", "optimized_weights", "prev_ltcma_df", "prev_corr_df"]:
                    st.session_state.pop(k, None)

                st.success("Defaults restored.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to restore defaults: {e}")
    # --- end save/restore ---

    
    if uploaded_session is not None and st.session_state.get("session_loaded") != uploaded_session.name:
        try:
            session_data = pickle.load(uploaded_session)
            assert "ltcma_df" in session_data and "sim_params" in session_data, "Invalid session file"
            
            # Restore tables into stores (not widget keys)
            st.session_state["ltcma_base_default"] = session_data["ltcma_df"].copy()
            st.session_state["corr_store"] = session_data["corr_matrix"].copy()
            st.session_state["corr_assets"] = tuple(session_data["ltcma_df"].index.tolist())

            # force editors to rebuild using the restored data
            st.session_state.pop("ltcma_widget", None)
            st.session_state.pop("corr_widget", None)

            # clear derived outputs
            for k in ["portfolio_paths", "x_axis", "optimized_weights", "prev_ltcma_df", "prev_corr_df"]:
                st.session_state.pop(k, None)


            # Restore parameters explicitly
            sim_params = session_data.get("sim_params", {})
            for param_key, param_value in sim_params.items():
                st.session_state[param_key] = param_value

            st.success("Session reloaded successfully.")

            # Mark session as loaded to avoid repeated reload
            st.session_state["session_loaded"] = uploaded_session.name

            st.rerun()  # Safely refresh once
        except Exception as e:
            st.error(f"Failed to reload session: {e}")
            st.stop()

# UI Widgets using st.session_state
start_date = st.sidebar.date_input("Start Date", value=st.session_state["start_date"], key="start_date")
n_years = st.sidebar.slider("Investment Horizon (Years)", 1, 30, st.session_state["n_years"], key="n_years")
frequency = st.sidebar.selectbox(
    "Frequency", ["monthly", "quarterly", "yearly"],
    index=["monthly", "quarterly", "yearly"].index(st.session_state["frequency"]),
    key="frequency"
)
initial_value = st.sidebar.number_input("Initial Portfolio Value", value=st.session_state["initial_value"], key="initial_value")
n_sims = st.sidebar.slider("Number of Simulations", 100, 5000, st.session_state["n_sims"], step=100, key="n_sims")


with st.sidebar.expander("Optional Display Settings"):
    n_paths_to_plot = st.slider("Paths to Display", 0, 50, 0)
    n_extreme_paths = st.slider("Extreme Paths (for Avg)", 0, 10, 0)
    show_double_initial = st.checkbox("Show Double Initial Value", value=False)
    show_double_with_inflation = st.checkbox("Show Double with Inflation", value=False)
    inflation_rate = st.number_input("Inflation Rate (for compounding)", value=0.025, step=0.001, format="%.3f")
    use_log_scale = st.checkbox("Log scale (Y axis)", value=False)
    
with st.sidebar.expander("Value at Risk (VaR) Settings"):
    var_years = st.slider("VaR Horizon (Years)", 1, 10, 1)
    var_confidence = st.slider("VaR Confidence Level (%)", 90, 99, 95)
    fat_tail_df = st.slider("Tail Thickness (Student-t df)", min_value=3, max_value=30, value=5, step=1)
    
# Efficient Frontier Parameters
with st.sidebar.expander("Efficient Frontier Parameters"):
    risk_free_rate = st.number_input("Risk-Free Rate", value=0.02, step=0.001, format="%.3f")
    show_cml = st.checkbox("Show Capital Market Line (CML)", value=True)
    show_saa = st.checkbox("Show Current SAA Portfolio", value=True)

# Optimization Parameters
with st.sidebar.expander("Optimization"):
    optimization_mode = st.radio("Optimization Mode", ["Max Return", "Min Volatility"])
    max_vol_limit = st.number_input("Max Volatility (for Max Return)", value=0.20, step=0.001, format="%.3f")
    min_return_limit = st.number_input("Min Return (for Min Volatility)", value=0.04, step=0.001, format="%.3f")
    use_optimized_weights = st.checkbox("Use Optimized Weights in Simulation", value=False)

# Scenario Analysis
with st.sidebar.expander("Historical Scenario Analysis"):
    scenario_file = st.file_uploader("Upload Historical Scenarios", type=["xlsx"])
    selected_scenario = None
    scenarios_df = None
    use_opt_in_scenario = st.checkbox("Use Optimized Weights in Scenario", value=False)

    if scenario_file is not None:
        file_suffix = Path(scenario_file.name).suffix.lower()
        try:
            if file_suffix == ".xlsx":
                scenarios_df = pd.read_excel(scenario_file)
            else:
                st.warning("Unsupported file type for scenario upload.")
        except Exception as e:
            st.error(f"Error reading scenario file: {e}")

        if scenarios_df is not None and not scenarios_df.empty:
            scenario_names = scenarios_df.iloc[:, 0].astype(str).tolist()
            selected_scenario = st.selectbox("Select Scenario", scenario_names)

st.sidebar.markdown(
    f"""
    <style>
      .app-version-badge {{
        position: fixed;
        bottom: 10px;
        left: 12px;
        right: 12px;
        font-size: 0.8rem;
        color: #6b7280; /* muted grey */
        opacity: 0.9;
      }}
    </style>
    <div class="app-version-badge">Version {APP_VERSION}</div>
    """,
    unsafe_allow_html=True,
)


# --- LTCMA (live editor, no Apply) ---

DEFAULT_LTCMA = pd.DataFrame({
    "Exp Return": [0.06, 0.03, 0.08],
    "Exp Volatility": [0.10, 0.05, 0.15],
    "SAA": [0.5, 0.3, 0.2],
    "Min": [0.0, 0.0, 0.0],
    "Max": [1.0, 1.0, 1.0]
}, index=["Equities", "Bonds", "Alternatives"])

st.subheader("LTCMA Table")
uploaded_ltcma = st.file_uploader("Upload LTCMA File", type=["xls", "xlsx"], key="ltcma_upload", label_visibility="collapsed")

# Base table to show in the widget
if uploaded_ltcma is not None:
    base_ltcma = pd.read_excel(uploaded_ltcma, index_col=0)
    # reset the widget state so the upload actually shows up
    st.session_state.pop("ltcma_widget", None)
else:
    # if the widget already has state, it will ignore this base
    base_ltcma = st.session_state.get("ltcma_base_default", DEFAULT_LTCMA)

# Set the format to float    v4.11
for c in ["Exp Return", "Exp Volatility", "SAA", "Min", "Max"]:
    if c in base_ltcma.columns:
        base_ltcma[c] = pd.to_numeric(base_ltcma[c], errors="coerce").astype(float)

# Render the editor. IMPORTANT: use the return value; do NOT read st.session_state["ltcma_widget"].
ltcma_return = st.data_editor(
    base_ltcma,
    num_rows="dynamic",
    use_container_width=True,
    key="ltcma_widget"
)

# Light hygiene on a copy for calculations ONLY (don’t write back to widget)
ltcma_df = ltcma_return.copy()
for c in ["Exp Return", "Exp Volatility", "SAA", "Min", "Max"]:
    if c in ltcma_df.columns:
        ltcma_df[c] = pd.to_numeric(ltcma_df[c], errors="coerce")

ltcma_df = ltcma_df.dropna(how="all")
ltcma_df.index = ltcma_df.index.astype(str).str.strip()
ltcma_df = ltcma_df.loc[ltcma_df.index != ""]

# Change detection (no hashes). If table changed -> clear derived outputs
prev_ltcma = st.session_state.get("prev_ltcma_df")
if prev_ltcma is None or not ltcma_df.equals(prev_ltcma):
    st.session_state["prev_ltcma_df"] = ltcma_df.copy()
    st.session_state.pop("portfolio_paths", None)
    st.session_state.pop("x_axis", None)

# --- keep corr editor in lockstep with LTCMA assets ---
new_assets = tuple(ltcma_df.index.tolist())
prev_assets = st.session_state.get("corr_assets")

if prev_assets is None or prev_assets != new_assets:
    # realign stored corr to the new asset set and reset the editor widget
    st.session_state["corr_store"] = ensure_corr_shape(ltcma_df.index, st.session_state.get("corr_store"))
    st.session_state["corr_assets"] = new_assets
    st.session_state.pop("corr_widget", None)  # forces the editor to rebuild with new rows/cols

# --- Correlation Matrix (live editor, auto-align to LTCMA) ---

# Header + Symmetrize button on the right
hcol, bcol = st.columns([6, 1])
with hcol:
    st.subheader("Correlation Matrix")
with bcol:
    symm_click = st.button(
        "↔ Symmetrize",
        key="symmetrize_btn",
        help="Set to (A + Aᵀ)/2, clip to [-1,1], diagonal=1"
    )

uploaded_corr = st.file_uploader(
    "Upload Corr Matrix", type=["xls", "xlsx"], key="corr_upload", label_visibility="collapsed"
)

# Keep a separate store for corr (NOT the widget key)
corr_store: pd.DataFrame | None = st.session_state.get("corr_store")

# If user uploads, load + reset widget state so it shows
if uploaded_corr is not None:
    corr_store = pd.read_excel(uploaded_corr, index_col=0)
    st.session_state["corr_store"] = corr_store.copy()
    st.session_state.pop("corr_widget", None)

# Always align to current LTCMA assets
corr_base = ensure_corr_shape(ltcma_df.index, st.session_state.get("corr_store")).astype(float)

# Build numeric column config for the editor
float_config = {
    c: st.column_config.NumberColumn(label=c, min_value=-1.0, max_value=1.0, step=0.01, format="%.2f")
    for c in corr_base.columns
}

# Render the editor (never write to this key in code)
corr_return = st.data_editor(
    corr_base,
    use_container_width=True,
    column_config=float_config,
    key="corr_widget"
)

# Gentle hint if the matrix isn't symmetric
try:
    if not np.allclose(corr_return.values, corr_return.values.T, atol=1e-10, equal_nan=True):
        st.info("Matrix isn’t symmetric. Click ↔ Symmetrize to fix.")
except Exception:
    pass

# If the button is clicked, symmetrize, persist, refresh editor, and clear derived outputs
if symm_click:
    sym = (corr_return + corr_return.T) / 2.0
    sym = sym.clip(-1.0, 1.0)
    np.fill_diagonal(sym.values, 1.0)

    st.session_state["corr_store"] = sym.copy()
    st.session_state["prev_corr_df"] = sym.copy()

    # Clear dependent outputs and force the editor to rebuild with the symmetrized values
    st.session_state.pop("portfolio_paths", None)
    st.session_state.pop("x_axis", None)
    st.session_state.pop("corr_widget", None)
    st.rerun()

# Normal path (no click): use edited value as-is, with diag forced to 1
corr_matrix = corr_return.copy()
np.fill_diagonal(corr_matrix.values, 1.0)

# Persist clean copy to the store, and clear derived outputs if changed
if not corr_matrix.equals(st.session_state.get("prev_corr_df")):
    st.session_state["prev_corr_df"] = corr_matrix.copy()
    st.session_state["corr_store"] = corr_matrix.copy()
    st.session_state.pop("portfolio_paths", None)
    st.session_state.pop("x_axis", None)




# Simulation Input Check
if not ltcma_df.empty and not corr_matrix.empty and ltcma_df.index.equals(corr_matrix.index):
    ltcma_df = ltcma_df.loc[ltcma_df.index.intersection(corr_matrix.index)]

    frequency_map = {"monthly": 12, "quarterly": 4, "yearly": 1}
#    date_freq_map = {"monthly": "ME", "quarterly": "QE", "yearly": "YE"}
    steps_per_year = frequency_map[frequency]
    n_steps = n_years * steps_per_year
    dt = 1 / steps_per_year

    vols = ltcma_df["Exp Volatility"].values
    mu = ltcma_df["Exp Return"].values
    weights = ltcma_df["SAA"].values

    if not np.isclose(weights.sum(), 1.0):
        st.warning("Warning: Asset weights (SAA) do not sum to 100%.")
    if np.any(vols < 0) or np.any(mu < 0):
        st.warning("Warning: Negative volatility or return detected.")

    corr = corr_matrix.loc[ltcma_df.index, ltcma_df.index].values
    cov = np.outer(vols, vols) * corr


    if use_optimized_weights and "optimized_weights" in st.session_state:
        opt_weights = st.session_state["optimized_weights"]
        if isinstance(opt_weights, pd.Series):
            # Align by index to current LTCMA table
            try:
                weights_used = opt_weights.reindex(ltcma_df.index).fillna(0).values
            except Exception as e:
                st.warning(f"Optimized weights could not be aligned: {e}")
                weights_used = ltcma_df["SAA"].values
        elif isinstance(opt_weights, np.ndarray) and len(opt_weights) == len(ltcma_df):
            weights_used = opt_weights
        else:
            st.warning("Optimized weights do not match the current asset set. Re-run optimization.")
            weights_used = ltcma_df["SAA"].values
    else:
        weights_used = ltcma_df["SAA"].values


    expected_portfolio_return = np.dot(weights_used, mu)
    expected_portfolio_volatility = np.sqrt(weights_used.T @ cov @ weights_used)

    # Draw the 4 buttons
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        run_sim = st.button("Run Simulation")
    with col2:
        run_ef = st.button("Run Efficient Frontier")
    with col3:
        run_opt = st.button("Run Optimization")
    with col4:
        run_scenario = st.button("Run Scenario")

    # Efficient Frontier
    if run_ef:
        bounds = list(zip(ltcma_df["Min"], ltcma_df["Max"]))
        results = []

        def portfolio_stats(w):
            ret = np.dot(w, mu)
            vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
            return ret, vol

        def min_func_vol(w):
            return portfolio_stats(w)[1]

        target_returns = np.linspace(mu.min(), mu.max(), 50)
        for target in target_returns:
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                           {'type': 'eq', 'fun': lambda w: np.dot(w, mu) - target})
            result = minimize(min_func_vol, weights, method='SLSQP', bounds=bounds, constraints=constraints)
            if result.success:
                r, v = portfolio_stats(result.x)
                results.append((v, r))

        results = np.array(results)
        fig2, ax2 = plt.subplots()
        ax2.plot(results[:, 0], results[:, 1], 'b-', label='Efficient Frontier')

        if show_cml:
            max_sharpe_idx = np.argmax((results[:, 1] - risk_free_rate) / results[:, 0])
            cml_x = [0, results[max_sharpe_idx, 0]]
            cml_y = [risk_free_rate, results[max_sharpe_idx, 1]]
            ax2.plot(cml_x, cml_y, 'r--', label='Capital Market Line')

        if show_saa:
            ax2.scatter(expected_portfolio_volatility, expected_portfolio_return, color='orange', label='Current SAA', zorder=5)

        ax2.set_xlabel('Volatility')
        ax2.set_ylabel('Return')
        ax2.set_title('Efficient Frontier')
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

        ef_buf = BytesIO()
        fig2.savefig(ef_buf, format="png")
        ef_buf.seek(0)
        st.download_button("Download Efficient Frontier Chart", data=ef_buf, file_name="efficient_frontier.png", mime="image/png")

  
    # Simulate
    if run_sim:

        # --- Fixed seed for reproducible simulations ---
        # Change this number to any non-negative integer to get a different but still reproducible run.
        # Acceptable values: any Python int >= 0 (e.g., 0, 1, 42, 20250826).
        SIMULATION_SEED = 20250826
        sim_rng = np.random.default_rng(SIMULATION_SEED)
        # -----------------------------------------------


        # choose weights for the sim
        weights_to_use = (
            st.session_state.get("optimized_weights").reindex(ltcma_df.index).fillna(0).values
            if use_optimized_weights and "optimized_weights" in st.session_state
            else ltcma_df["SAA"].values
        )

        # pull inputs
        mu = ltcma_df["Exp Return"].values
        vols = ltcma_df["Exp Volatility"].values

        # user correlation, aligned to LTCMA assets
        corr = corr_matrix.loc[ltcma_df.index, ltcma_df.index].values
        cov = np.outer(vols, vols) * corr

        # expected stats for the summary tab, tied to the exact weights/cov used
        expected_portfolio_return = np.dot(weights_to_use, mu)
        expected_portfolio_volatility = np.sqrt(weights_to_use.T @ cov @ weights_to_use)

        steps_per_year = {"monthly": 12, "quarterly": 4, "yearly": 1}[frequency]
        n_steps = n_years * steps_per_year
        dt = 1 / steps_per_year

        # robust Cholesky for nearly PSD covariances
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals[eigvals < 0] = 0.0
            cov = eigvecs @ np.diag(eigvals) @ eigvecs.T + 1e-12 * np.eye(cov.shape[0])
            chol = np.linalg.cholesky(cov)

        # Student-t scaling so shocks have unit variance pre-correlation
        scale = np.sqrt(fat_tail_df / (fat_tail_df - 2))  # df > 2

        portfolio_paths = np.zeros((n_steps + 1, n_sims))
        portfolio_paths[0] = initial_value

        for sim in range(n_sims):
            prices = np.ones(len(weights_to_use))
            path = [initial_value]
            for _ in range(n_steps):
                z = t.rvs(fat_tail_df, size=len(weights_to_use), random_state=sim_rng) / scale  # v4.0.2 : use a fixed seed for random generator
                correlated_z = chol @ z
                prices *= np.exp(mu * dt + correlated_z * np.sqrt(dt))
                path.append(np.dot(prices, weights_to_use) * initial_value)
            portfolio_paths[:, sim] = path

        st.session_state["portfolio_paths"] = portfolio_paths

        # make the x-axis frequency match the chosen sim frequency
        freq_map = {"monthly": "ME", "quarterly": "QE", "yearly": "YE"}
        st.session_state["x_axis"] = pd.date_range(start=start_date, periods=n_steps + 1, freq=freq_map[frequency])

        st.session_state.simulation_has_run = True

        st.write("Weights used in simulation:")
        st.dataframe(pd.DataFrame({"Weight": weights_to_use}, index=ltcma_df.index))


    # Optimize
    if run_opt:
        bounds = list(zip(ltcma_df["Min"], ltcma_df["Max"]))

        def objective_max_return(w):
            return -np.dot(w, mu)

        def objective_min_vol(w):
            return np.sqrt(np.dot(w.T, np.dot(cov, w)))

        if optimization_mode == "Max Return":
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                           {'type': 'ineq', 'fun': lambda w: max_vol_limit - np.sqrt(w.T @ cov @ w)})
            result = minimize(objective_max_return, weights, method='SLSQP', bounds=bounds, constraints=constraints)
        else:  # Min Volatility
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                           {'type': 'ineq', 'fun': lambda w: np.dot(w, mu) - min_return_limit})
            result = minimize(objective_min_vol, weights, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            opt_w = result.x
            st.session_state["optimized_weights"] = pd.Series(opt_w, index=ltcma_df.index)

            opt_r = np.dot(opt_w, mu)
            opt_v = np.sqrt(opt_w.T @ cov @ opt_w)
            st.success("Optimization Successful")
            st.write("**Optimized Weights:**")
            formatted_weights = pd.DataFrame({"Weight": opt_w}, index=ltcma_df.index)
            formatted_weights["Weight"] = formatted_weights["Weight"].map("{:.4f}".format)
            st.dataframe(formatted_weights)
            
            fig_opt, ax_opt = plt.subplots()
            ax_opt.scatter(opt_v, opt_r, c='green', label='Optimized Portfolio')
            ax_opt.set_xlabel('Volatility')
            ax_opt.set_ylabel('Return')
            ax_opt.set_title('Optimized Portfolio Result')
            ax_opt.grid(True)
            ax_opt.legend()
            st.pyplot(fig_opt)

            opt_buf = BytesIO()
            fig_opt.savefig(opt_buf, format="png")
            opt_buf.seek(0)
            st.download_button("Download Optimization Chart",
                data=opt_buf,
                file_name="optimized_portfolio.png",
                mime="image/png")
        else:
            st.error("Optimization failed.")

    # Stress Test Scenario
    if run_scenario:
        if scenarios_df is None:
            st.error("No scenario file uploaded. Please upload a CSV or Excel file with scenarios.")
        elif selected_scenario is None:
            st.error("No scenario selected. Please select a scenario from the dropdown.")
        elif ltcma_df.empty:
            st.error("LTCMA data not available. Please upload or define LTCMA first.")
        else:
            st.subheader(f"Impact of Historical Scenario: {selected_scenario}")

            scenario_row = scenarios_df[scenarios_df.iloc[:, 0].astype(str) == selected_scenario]
            if scenario_row.empty:
                st.error("Selected scenario not found in the scenario file.")
            else:
                scenario_row = scenario_row.iloc[0, 1:]

                # --- Align scenario vector to current LTCMA assets ---
                # 1) normalize labels
                scenario_row.index = scenario_row.index.astype(str).str.strip()
                ltcma_assets = ltcma_df.index  # already stripped earlier

                # 2) align to LTCMA; extras are silently dropped, missing are filled with 0
                
                #extra_assets = [c for c in scenario_row.index if c not in ltcma_assets]
                scenario_aligned = scenario_row.reindex(ltcma_assets)
                missing_assets = scenario_aligned[scenario_aligned.isna()].index.tolist()
                scenario_aligned = scenario_aligned.fillna(0.0)

                # 3) choose weights as a Series aligned to LTCMA assets
                if use_opt_in_scenario and "optimized_weights" in st.session_state:
                    weights_used = (
                        st.session_state["optimized_weights"]
                        .reindex(ltcma_assets)
                        .fillna(0.0)
                    )
                else:
                    weights_used = ltcma_df["SAA"]  # already indexed by ltcma_assets

                # 4) compute portfolio scenario return (both Series share index)
                scenario_return = float((scenario_aligned * weights_used).sum())

                if missing_assets:
                    st.info(
                        "Assets missing in scenario (assumed 0% shock): "
                        + ", ".join(map(str, missing_assets))
                    )

                # 6) build contribution table using aligned vectors
                impact_df = pd.DataFrame(
                    {
                        "Asset Class": ltcma_assets,
                        "Shock Return": scenario_aligned.values,
                        "Weight": weights_used.values,
                    },
                    index=ltcma_assets,
                )
                impact_df["Contribution"] = impact_df["Shock Return"] * impact_df["Weight"]

                # Format numerical columns for display
                formatted_df = impact_df.copy()
                formatted_df["Shock Return"] = (formatted_df["Shock Return"] * 100).map("{:.2f}%".format)
                formatted_df["Contribution"] = (formatted_df["Contribution"] * 100).map("{:.2f}%".format)

                st.dataframe(formatted_df)



                fig, ax = plt.subplots(figsize=(7, 4))
                ax.bar(impact_df["Asset Class"], impact_df["Contribution"], color="cornflowerblue")
                ax.axhline(scenario_return, color='red', linestyle='--', label=f"Total: {scenario_return:.2%}")
                ax.set_title("Contribution to Portfolio Return under Scenario")
                ax.set_ylabel("Contribution")
                ax.set_xlabel("Asset Class")
                ax.tick_params(axis='x', rotation=30, labelsize=9)
                ax.legend()
                st.pyplot(fig)

                st.markdown(f"**Portfolio Return under Scenario '{selected_scenario}':** {scenario_return:.2%}")




#v4.4 ------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Simulation Chart", "Summary Statistics", "Value at Risk", "Drawdown Analysis"])

# small helper: do we have data to render?
has_data = (
    st.session_state.get("simulation_has_run")
    and isinstance(st.session_state.get("portfolio_paths"), np.ndarray)
    and st.session_state.get("x_axis") is not None
)

#v4.4 ------------------------------



with tab1:
    if not has_data:
        st.info("Run a simulation to see the chart.")
    else:
        portfolio_paths = st.session_state["portfolio_paths"]
        x_axis = st.session_state["x_axis"]

        p25 = np.percentile(portfolio_paths, 25, axis=1)
        p75 = np.percentile(portfolio_paths, 75, axis=1)
        p05 = np.percentile(portfolio_paths, 5, axis=1)
        p95 = np.percentile(portfolio_paths, 95, axis=1)
        median_path = np.median(portfolio_paths, axis=1)

        fig, ax = plt.subplots(figsize=(10, 6))

        # v4.12: switch to log scale if requested
        if use_log_scale:
            ax.set_yscale("log")

        if n_paths_to_plot > 0:
            for idx in np.random.choice(portfolio_paths.shape[1], n_paths_to_plot, replace=False):
                color = np.random.rand(3,)  # random RGB
                ax.plot(x_axis, portfolio_paths[:, idx], color=color, linewidth=0.5, alpha=0.4)

        if n_extreme_paths > 0:
            final_values = portfolio_paths[-1]
            best_idx = np.argsort(final_values)[-n_extreme_paths:]
            worst_idx = np.argsort(final_values)[:n_extreme_paths]

            ax.plot(x_axis, np.mean(portfolio_paths[:, best_idx], axis=1), color='green', label='Avg Best')
            ax.plot(x_axis, np.mean(portfolio_paths[:, worst_idx], axis=1), color='red', label='Avg Worst')

        if show_double_initial:
            ax.plot(x_axis, np.full(len(x_axis), 2 * initial_value), color='red', linestyle='--', linewidth=1, label='2x Initial Value')

        # Determine compounding intervals per year based on frequency
        periods_per_year = {"monthly": 12, "quarterly": 4, "yearly": 1}[frequency]

        # Total number of steps in the simulation
        n_steps = len(x_axis)

        # Compute compounded inflation-adjusted target: 2x initial, compounded
        growth_factor = (1 + inflation_rate / periods_per_year) ** np.arange(n_steps)
        compounded_double = initial_value * 2 * growth_factor

        if show_double_with_inflation:
            ax.plot(x_axis, compounded_double, color='darkorange', linestyle='--', linewidth=1, label='Inflation-Adj. 2x Target')
    
        ax.plot(x_axis, median_path, color='blue', label='Median')
        ax.fill_between(x_axis, p05, p95, color='lightblue', alpha=0.3, label='5–95%')
        ax.fill_between(x_axis, p25, p75, color='blue', alpha=0.2, label='25–75%')


        # Add labels at the end of percentile lines
        label_fontsize = 8
        x_last = x_axis[-1]+ pd.Timedelta(days=15)

        ax.text(x_last, median_path[-1], f"Median: {median_path[-1]:,.0f}", color="blue",
                fontsize=label_fontsize, va="center", ha="left")

        ax.text(x_last, p25[-1], f"25th: {p25[-1]:,.0f}", color="blue",
                fontsize=label_fontsize, va="center", ha="left", alpha=0.7)

        ax.text(x_last, p75[-1], f"75th: {p75[-1]:,.0f}", color="blue",
                fontsize=label_fontsize, va="center", ha="left", alpha=0.7)

        ax.text(x_last, p05[-1], f"5th: {p05[-1]:,.0f}", color="lightblue",
                fontsize=label_fontsize, va="center", ha="left", alpha=0.9)

        ax.text(x_last, p95[-1], f"95th: {p95[-1]:,.0f}", color="lightblue",
                fontsize=label_fontsize, va="center", ha="left", alpha=0.9)

        ax.set_title("Portfolio Value Over Time")
        ax.set_xlabel("Date")
        # ax.set_ylabel("Portfolio Value")  #v4.12
        ax.set_ylabel("Portfolio Value (log scale)" if use_log_scale else "Portfolio Value")
        
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)

        #ax.set_ylim(0, 20000)    # Forces th Y axis: Sets min=0, max=5000, # to delete later when no longer needed


        if use_log_scale:
            # ensure a positive bottom margin on log axes
            y_min = max(1e-6, min(p05.min(), p25.min(), median_path.min()))
            ax.set_ylim(bottom=y_min * 0.9)

        
        st.pyplot(fig)

        buf = BytesIO()
        fig.savefig(buf, format="png")


        percentile_df = pd.DataFrame({
            "5th Percentile": p05,
            "25th Percentile": p25,
            "Median": median_path,
            "75th Percentile": p75,
            "95th Percentile": p95
        }, index=x_axis)

        percentile_df.index.name = "Date"
        
        if show_double_initial:
            percentile_df["2x Initial Value"] = 2 * initial_value

        if show_double_with_inflation:
            percentile_df["Inflation-Adj. 2x Target"] = compounded_double

        # Write to Excel in memory
        excel_buf = BytesIO()
        with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
            percentile_df.to_excel(writer, sheet_name="Percentile Paths")

        excel_buf.seek(0)


        left_button, middle_button, right_button = st.columns([1, 1, 1])

        with left_button:
            st.download_button(
                label="Download Plot as PNG",
                data=buf.getvalue(),
                file_name="simulation_plot.png",
                mime="image/png"
            )
        with middle_button:
            st.download_button(
                label="Download Percentile Paths (Excel)",
                data=excel_buf,
                file_name="percentile_paths.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_percentile_paths"
            )
        with right_button:
            # Build a tidy DataFrame for Excel (columns = simulations, index = dates)
            paths_df = pd.DataFrame(
                portfolio_paths,
                index=x_axis,
                columns=[f"Sim_{i+1}" for i in range(portfolio_paths.shape[1])]
            )
            paths_df.index.name = "Date"

            # Write to XLSX in-memory
            paths_xlsx = BytesIO()
            with pd.ExcelWriter(paths_xlsx, engine="xlsxwriter") as writer:
                paths_df.to_excel(writer, sheet_name="Simulation Paths", index=True)

            paths_xlsx.seek(0)
            st.download_button(
                label="Download Simulation Paths (Excel)",
                data=paths_xlsx,
                file_name="simulated_paths.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_paths_xlsx"
            )

with tab2:
    if not has_data:
        st.info("Run a simulation to see summary statistics.")
    else:
        portfolio_paths = st.session_state["portfolio_paths"]
        x_axis = st.session_state["x_axis"]
        
        st.subheader("Summary Statistics")

        left_col_final, right_col_final = st.columns([1, 2])

        with left_col_final:
            
            st.markdown(f"**Expected Return:** {expected_portfolio_return:.2%}")
            st.markdown(f"**Expected Volatility:** {expected_portfolio_volatility:.2%}")
            mean_final = np.mean(portfolio_paths[-1])
            median_final = np.median(portfolio_paths[-1])
            pctiles = np.percentile(portfolio_paths[-1], [5, 25, 75, 95])
            st.markdown(f"**Final Value (Mean):** ${mean_final:,.2f}")
            st.markdown(f"**Final Value (Median):** ${median_final:,.2f}")
            st.markdown(f"**25–75% Range:** ${pctiles[1]:,.2f} – ${pctiles[2]:,.2f}")
            st.markdown(f"**5–95% Range:** ${pctiles[0]:,.2f} – ${pctiles[3]:,.2f}")

        with right_col_final:

            fig_hist, ax_hist = plt.subplots(figsize=(5, 4))
            final_values = portfolio_paths[-1]

            # Calculate 1st and 99th percentiles to limit x-axis range
            x_min, x_max = np.percentile(final_values, [1, 99])
            # Plot histogram within these bounds
            ax_hist.hist(final_values, bins=50, range=(0, x_max*1.5), color='skyblue', edgecolor='black')
            # Set explicit x-axis limits
            ax_hist.set_xlim(0, x_max*1.5)
            
            ax_hist.set_title("Distribution of Final Portfolio Values")
            ax_hist.set_xlabel("Final Value")
            ax_hist.set_ylabel("Frequency")

            # Download button for Final Value distribution
            hist_buf = BytesIO()
            fig_hist.savefig(hist_buf, format="png")
            hist_buf.seek(0)

            # Save to session state
            st.session_state["fig_hist"] = fig_hist
            st.session_state["buf_hist"] = hist_buf

            # Display & download
            st.pyplot(st.session_state["fig_hist"])

            # Export final portfolio values to Excel
            final_vals_df = pd.DataFrame({"Final Portfolio Value": final_values})
            excel_buf_final = BytesIO()
            with pd.ExcelWriter(excel_buf_final, engine="openpyxl") as writer:
                final_vals_df.to_excel(writer, sheet_name="Final Values", index=False)
            excel_buf_final.seek(0)

            left_button, right_button = st.columns([1, 1])

            with left_button:
                st.download_button(
                    label="Download Final Value Distribution Chart", key="download_final_value3",
                    data=st.session_state["buf_hist"],
                    file_name="final_value_distribution.png",
                    mime="image/png"
                )

            with right_button:
                st.download_button(
                    label="Download Final Values (Excel)",
                    data=excel_buf_final,
                    file_name="final_portfolio_values.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_final_values"
                )
 

with tab3:
    if not has_data:
        st.info("Run a simulation to see VaR.")
    else:
        portfolio_paths = st.session_state["portfolio_paths"]
        x_axis = st.session_state["x_axis"]

        st.subheader("Value at Risk Statistics")
        if var_years * steps_per_year <= n_steps:
            steps_for_var = int(var_years * steps_per_year)
            var_values = portfolio_paths[steps_for_var]
            var_threshold = np.percentile(var_values, 100 - var_confidence)
            var_loss = initial_value - var_threshold
            var_pct_loss = var_loss / initial_value

            cvar_values = var_values[var_values <= var_threshold]
            cvar_mean = np.mean(cvar_values) if len(cvar_values) > 0 else np.nan
            cvar_loss = initial_value - cvar_mean
            cvar_pct_loss = cvar_loss / initial_value

            left_col_var, right_col_var = st.columns([1, 2])

            with left_col_var:
                st.markdown(f"**{var_confidence}% {var_years}-Year Value at Risk (VaR):** With {var_confidence}% confidence, "
                    f"the portfolio value will not fall below ${var_threshold:,.2f}.")
                st.markdown(f"This represents a potential shortfall of ${var_loss:,.2f} ({var_pct_loss:.2%}) from the initial value.")
                st.markdown(
                    f"**{var_confidence}% Conditional VaR (CVaR):**\n"
                    f"Expected average shortfall beyond VaR is **${cvar_loss:,.2f} ({cvar_pct_loss:.2%})**."
                )
                st.markdown(f"**Average VaR:** {np.mean(var_values):.2f}")
                st.markdown(f"**Worst VaR:** {np.min(var_values):.2f}")
                st.markdown(f"**5th Percentile VaR:** {np.percentile(var_values, 5):.2f}")
                st.markdown(f"**25th Percentile VaR:** {np.percentile(var_values, 25):.2f}")
                st.markdown(f"**Median VaR:** {np.percentile(var_values, 50):.2f}")
                st.markdown(f"**75th Percentile VaR:** {np.percentile(var_values, 75):.2f}")
                st.markdown(f"**95th Percentile VaR:** {np.percentile(var_values, 95):.2f}")

            with right_col_var:
                
                var_threshold = np.percentile(var_values, 100 - var_confidence)
                fig_var, ax_var = plt.subplots(figsize=(5, 4))

                # Compute reasonable axis limits using percentiles
                x_min, x_max = np.percentile(var_values, [1, 99])
                # Plot histogram within these bounds
                ax_var.hist(var_values, bins=50, range=(x_min/2, x_max*1.5), color='lightgrey', edgecolor='black')
                # Explicitly set x-axis limits
                ax_var.set_xlim(x_min/2, x_max*1.5)

                ax_var.axvline(var_threshold, color='red', linestyle='--', label=f'{var_confidence}% VaR = {var_threshold:,.2f}')
                ax_var.set_title(f"{var_confidence}% VaR at Year {var_years}")
                ax_var.set_xlabel("Portfolio Value")
                ax_var.set_ylabel("Frequency")
                ax_var.legend()

                # Save buffer for download
                var_buf = BytesIO()
                fig_var.savefig(var_buf, format="png")
                var_buf.seek(0)
                st.session_state["fig_var"] = fig_var
                st.session_state["buf_var"] = var_buf

                # Display and download
                st.pyplot(fig_var)

                # Export final period returns for VaR to Excel
                var_df = pd.DataFrame({"Final Period Return": var_values})
                excel_buf_var = BytesIO()
                with pd.ExcelWriter(excel_buf_var, engine="openpyxl") as writer:
                    var_df.to_excel(writer, sheet_name="VaR Distribution", index=False)
                excel_buf_var.seek(0)

                left_button, right_button = st.columns([1, 1])

                with left_button:
                    st.download_button(
                        label=f"Download {var_confidence}% VaR Distribution Chart",
                        data=st.session_state["buf_var"],key="download_var2",
                        file_name="var_distribution.png",
                        mime="image/png"
                    )
                    
                with right_button:
                    st.download_button(
                       label="Download VaR Returns (Excel)",
                       data=excel_buf_var,
                       file_name="var_distribution.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       key="download_var_excel"
                   )
 
            
        else:
            st.warning("VaR horizon exceeds simulation length. Increase investment horizon or reduce VaR years.")

with tab4:
    if not has_data:
        st.info("Run a simulation to see drawdown analysis.")
    else:
        portfolio_paths = st.session_state["portfolio_paths"]
        x_axis = st.session_state["x_axis"]

        st.subheader("Drawdown Statistics")

        # Drawdown statistics
        max_drawdowns = []
        recovery_times = []
        
        for sim in range(portfolio_paths.shape[1]):
            path = portfolio_paths[:, sim]

            peak_val = path[0]
            peak_idx = 0
            max_dd = 0.0
            rec_time_for_max = np.nan  # reset for this simulation

            for i in range(1, len(path)):
                # update peak
                if path[i] > peak_val:
                    peak_val = path[i]
                    peak_idx = i

                # current drawdown from the most recent peak
                dd = 1.0 - path[i] / peak_val

                # found a deeper max drawdown → reset recovery to NaN and search ahead
                if dd > max_dd:
                    max_dd = dd
                    rec_time_for_max = np.nan  # Reset so we don’t carry over a prior recovery

                    # look forward for recovery to that peak before end of series
                    for j in range(i + 1, len(path)):
                        if path[j] >= peak_val:
                            rec_time_for_max = j - peak_idx
                            break

            max_drawdowns.append(max_dd)
            recovery_times.append(rec_time_for_max)

        left_col_dd, right_col_dd = st.columns([1, 2])

        with left_col_dd:
            st.markdown(f"**Average Max Drawdown:** {np.mean(max_drawdowns):.2%}")
            st.markdown(f"**Worst Max Drawdown:** {np.max(max_drawdowns):.2%}")
            st.markdown(f"**5th Percentile Max Drawdown:** {np.percentile(max_drawdowns, 5):.2%}")
            st.markdown(f"**25th Percentile Max Drawdown:** {np.percentile(max_drawdowns, 25):.2%}")
            st.markdown(f"**Median Max Drawdown:** {np.percentile(max_drawdowns, 50):.2%}")
            st.markdown(f"**75th Percentile Max Drawdown:** {np.percentile(max_drawdowns, 75):.2%}")
            st.markdown(f"**95th Percentile Max Drawdown:** {np.percentile(max_drawdowns, 95):.2%}")

            # recovery time stats only for simulations that actually recovered
            rec_array = np.array(recovery_times, dtype=float)
            recovered_mask = ~np.isnan(rec_array)
            recovered_share = 100.0 * recovered_mask.mean()

            freq_label = {"monthly": "months", "quarterly": "quarters", "yearly": "years"}[frequency]
            if recovered_mask.any():
                st.markdown(f"**Average Recovery Time ({freq_label}, recovered only):** {np.nanmean(rec_array):.1f}")
#               st.markdown(f"**Median Recovery Time ({freq_label}, recovered only):** {np.nanmedian(rec_array):.1f}")
            else:
                st.markdown(f"**Average Recovery Time ({freq_label}, recovered only):** n/a")

            st.markdown(f"**Recovered Paths:** {recovered_share:.1f}% "
                        f"({recovered_mask.sum()} of {len(rec_array)})")

        with right_col_dd:
            # Plot distribution of Max Drawdowns
            fig_ddist, ax_ddist = plt.subplots(figsize=(6, 4))
            ax_ddist.hist(max_drawdowns, bins=40, color='salmon', edgecolor='black')
            ax_ddist.set_title("Distribution of Maximum Drawdowns")
            ax_ddist.set_xlabel("Max Drawdown")
            ax_ddist.set_ylabel("Frequency")
            ax_ddist.xaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
            ax_ddist.grid(True)

            # Save buffer
            ddist_buf = BytesIO()
            fig_ddist.savefig(ddist_buf, format="png")
            ddist_buf.seek(0)
            st.session_state["fig_ddist"] = fig_ddist
            st.session_state["buf_ddist"] = ddist_buf

            # Display and download

            st.pyplot(st.session_state["fig_ddist"])

            # Export drawdown and recovery time data
            drawdown_df = pd.DataFrame({
                "Max Drawdown": max_drawdowns,
                "Recovery Time": recovery_times
            })

            excel_buf_dd = BytesIO()
            with pd.ExcelWriter(excel_buf_dd, engine="openpyxl") as writer:
                drawdown_df.to_excel(writer, sheet_name="Drawdowns", index=False)

            excel_buf_dd.seek(0)

            left_button, right_button = st.columns([1, 1])

            with left_button:
                st.download_button(
                    label="Download Max Drawdown Distribution Chart",
                    data=st.session_state["buf_ddist"],
                    file_name="max_drawdown_distribution.png",
                    mime="image/png",
                    key="download_drawdown_tab"
                )
                
            with right_button:
                st.download_button(
                    label="Download Drawdown Data (Excel)",
                    data=excel_buf_dd,
                    file_name="drawdown_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_drawdown_excel"
                )
            


