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
import sys
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


# ---- default data files (edit paths as needed) ----
DEFAULT_LTCMA_PATH = "Data/LTCMA.xlsx"
DEFAULT_CORR_PATH = "Data/Correlation Matrix.xlsx"
DEFAULT_SCENARIO_PATH = "Data/Scenarios.xlsx"


st.set_page_config(layout="wide")
st.title("SAA Portfolio Monte Carlo Simulator")

# Sidebar Inputs

# Initialize default session state
default_values = {
    "start_date": pd.to_datetime("2024-10-01"),
    "n_years": 10,
    "initial_value": 200.0,
    "frequency": "monthly",
    "n_sims": 2000,
    "use_optimized_weights": False  # ← add this line
}

for key, val in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Sidebar Inputs referencing session_state directly
st.sidebar.header("Simulation Parameters")

with st.sidebar.expander("Session Management"):
    uploaded_session = st.file_uploader("Reload Saved Session", type=["pkl"], key="session_uploader")
    if uploaded_session is not None and st.session_state.get("session_loaded") != uploaded_session.name:
        try:
            session_data = pickle.load(uploaded_session)

            # Restore tables
            st.session_state["ltcma_df"] = session_data["ltcma_df"]
            st.session_state["corr_matrix"] = session_data["corr_matrix"]

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
    #scenario_file = st.file_uploader("Upload Historical Scenarios (CSV or Excel)", type=["csv", "xlsx"])
    scenario_file = st.file_uploader("Upload Historical Scenarios", type=["xlsx"])
    selected_scenario = None
    scenarios_df = None
    use_opt_in_scenario = st.checkbox("Use Optimized Weights in Scenario", value=False)

    if scenario_file is not None:
        file_suffix = Path(scenario_file.name).suffix.lower()
        try:
            if file_suffix == ".csv":
                scenarios_df = pd.read_csv(scenario_file)
            elif file_suffix == ".xlsx":
                scenarios_df = pd.read_excel(scenario_file)
            else:
                st.warning("Unsupported file type for scenario upload.")
        except Exception as e:
            st.error(f"Error reading scenario file: {e}")

        if scenarios_df is not None and not scenarios_df.empty:
            #st.write("### Debug: Loaded Scenario Data")
            #st.write(scenarios_df.head())
            scenario_names = scenarios_df.iloc[:, 0].astype(str).tolist()
            selected_scenario = st.selectbox("Select Scenario", scenario_names)

# v3.1  Save / Restore controls
col_save, col_restore , col_dummy = st.columns([1, 1, 2])

with col_save:
    if st.button("💾 Save Session", key="save_session_main"):
        session_data = {
            "ltcma_df": st.session_state["ltcma_df"],
            "corr_matrix": st.session_state["corr_matrix"],
            "optimized_weights": st.session_state.get("optimized_weights", None),
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
        href = f'<a href="data:file/octet-stream;base64,{b64}" download="saa_session.pkl">Download Session File</a>'
        st.markdown(href, unsafe_allow_html=True)

with col_restore:
    if st.button("↩️ Restore Defaults", key="restore_defaults", help="Load default LTCMA, Correlation, and Scenarios"):
        try:
            ltcma_default = pd.read_excel(DEFAULT_LTCMA_PATH, index_col=0)
            corr_default = pd.read_excel(DEFAULT_CORR_PATH, index_col=0)
            scenarios_default = pd.read_excel(DEFAULT_SCENARIO_PATH)

            # basic alignment and hygiene
            # ensure corr rows/cols match LTCMA index
            corr_default = corr_default.reindex(index=ltcma_default.index, columns=ltcma_default.index)
            corr_default.fillna(0.0, inplace=True)
            np.fill_diagonal(corr_default.values, 1.0)

            # write into session and clear derived caches
            st.session_state["ltcma_df"] = ltcma_default.copy()
            st.session_state["prev_ltcma"] = ltcma_default.copy()

            st.session_state["corr_matrix"] = corr_default.copy()
            st.session_state["prev_corr_matrix"] = corr_default.copy()

            # store scenarios for use if no upload provided
            st.session_state["default_scenarios_df"] = scenarios_default.copy()

            # clear outputs that depend on inputs
            for k in ["portfolio_paths", "x_axis", "optimized_weights"]:
                st.session_state.pop(k, None)

            st.success("Defaults restored.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to restore defaults: {e}")
#v3.1


# Defaults
DEFAULT_LTCMA = pd.DataFrame({
    "Exp Return": [0.06, 0.03, 0.08],
    "Exp Volatility": [0.10, 0.05, 0.15],
    "SAA": [0.5, 0.3, 0.2],
    "Min": [0.0, 0.0, 0.0],
    "Max": [1.0, 1.0, 1.0]
}, index=["Equities", "Bonds", "Alternatives"])

# LTCMA Section

ltcma_col1, ltcma_col2 = st.columns([2, 1])
with ltcma_col1:
    st.subheader("LTCMA Table")
with ltcma_col2:
    uploaded_ltcma = st.file_uploader("Upload LTCMA File", type=["xls", "xlsx"], key="ltcma_upload", label_visibility="collapsed")

if uploaded_ltcma is not None:
    try:
        ltcma_df = pd.read_excel(uploaded_ltcma, index_col=0)
        # Enforce Min/Max columns to be float
        for col in ["Min", "Max"]:
            if col in ltcma_df.columns:
                ltcma_df[col] = ltcma_df[col].astype(float)
        
        st.session_state["ltcma_df"] = ltcma_df.copy()
        st.session_state["prev_ltcma"] = ltcma_df.copy()
        st.session_state.pop("portfolio_paths", None)
        st.session_state.pop("x_axis", None)
    except Exception as e:
        st.error(f"Failed to read LTCMA file: {e}")

ltcma_df = st.session_state.get("ltcma_df", DEFAULT_LTCMA.copy())
asset_names = ltcma_df.index.tolist()
ltcma_df = st.data_editor(ltcma_df, num_rows="dynamic", use_container_width=True, key="ltcma_editor")
# Re-enforce float types after user editing
for col in ["Min", "Max"]:
    if col in ltcma_df.columns:
        ltcma_df[col] = ltcma_df[col].astype(float)

# Synchronize Correlation Matrix with LTCMA
def sync_corr_with_ltcma(ltcma_df, corr_matrix=None):
    assets = ltcma_df.index

    if corr_matrix is None or corr_matrix.empty:
        corr_matrix = pd.DataFrame(np.eye(len(assets)), index=assets, columns=assets)
    else:
        corr_matrix = corr_matrix.reindex(index=assets, columns=assets)
        corr_matrix.fillna(0.0, inplace=True)
        np.fill_diagonal(corr_matrix.values, 1.0)

    return corr_matrix

corr_matrix = sync_corr_with_ltcma(ltcma_df, st.session_state.get("corr_matrix"))
st.session_state["corr_matrix"] = corr_matrix

prev_ltcma = st.session_state.get("prev_ltcma")
if prev_ltcma is not None and not ltcma_df.equals(prev_ltcma):
    st.session_state.pop("portfolio_paths", None)
    st.session_state.pop("x_axis", None)
st.session_state["prev_ltcma"] = ltcma_df.copy()
st.session_state["ltcma_df"] = ltcma_df.copy()

# Correlation Section
corr_col1, corr_col2 ,corr_col3 = st.columns([1, 1, 1])
with corr_col1:
    st.subheader("Correlation Matrix")
with corr_col3:
    uploaded_corr = st.file_uploader("Upload Corr Matrix", type=["xls", "xlsx"], key="corr_upload", label_visibility="collapsed")
with corr_col2:
    apply_symmetry = st.button("↔", help="Apply Symmetry")


if uploaded_corr is not None:
    try:
        corr_matrix = pd.read_excel(uploaded_corr, index_col=0)
        st.session_state["corr_matrix"] = corr_matrix.copy()
        st.session_state["prev_corr_matrix"] = corr_matrix.copy()
        st.session_state.pop("portfolio_paths", None)
        st.session_state.pop("x_axis", None)
    except Exception as e:
        st.error(f"Failed to read correlation matrix: {e}")

existing_corr = st.session_state.get("corr_matrix")

corr_matrix = sync_corr_with_ltcma(ltcma_df, st.session_state.get("corr_matrix"))
st.session_state["corr_matrix"] = corr_matrix

float_config = {
    col: st.column_config.NumberColumn(
        label=col, min_value=-1.0, max_value=1.0, step=0.01, format="%.2f"
    ) for col in corr_matrix.columns
}
corr_matrix = st.data_editor(
    corr_matrix,
    use_container_width=True,
    disabled=False,
    column_config=float_config,
    key="corr_editor"
)

prev_corr = st.session_state.get("prev_corr_matrix")
if prev_corr is not None and not corr_matrix.equals(prev_corr):
    st.session_state.pop("portfolio_paths", None)
    st.session_state.pop("x_axis", None)
st.session_state["prev_corr_matrix"] = corr_matrix.copy()
st.session_state["corr_matrix"] = corr_matrix.copy()

if apply_symmetry:
    sym_corr = (corr_matrix + corr_matrix.T) / 2.0
    np.fill_diagonal(sym_corr.values, 1.0)
    st.session_state["corr_matrix"] = sym_corr
    st.session_state.pop("portfolio_paths", None)
    st.session_state.pop("x_axis", None)
    st.rerun()

if not corr_matrix.equals(corr_matrix.T):
    st.error("Correlation matrix must be symmetric.")
    st.stop()




# Simulation Input Check
if not ltcma_df.empty and not corr_matrix.empty and ltcma_df.index.equals(corr_matrix.index):
    ltcma_df = ltcma_df.loc[ltcma_df.index.intersection(corr_matrix.index)]
    st.session_state["ltcma_df"] = ltcma_df.copy()

    frequency_map = {"monthly": 12, "quarterly": 4, "yearly": 1}
    date_freq_map = {"monthly": "ME", "quarterly": "QE", "yearly": "YE"}
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
        weights_to_use = st.session_state.get("optimized_weights") if use_optimized_weights and "optimized_weights" in st.session_state else ltcma_df["SAA"].values

        # Recalculate expected return and volatility with actual weights used
        expected_portfolio_return = np.dot(weights_to_use, mu)
        expected_portfolio_volatility = np.sqrt(weights_to_use.T @ cov @ weights_to_use)

        mu = ltcma_df["Exp Return"].values
        vols = ltcma_df["Exp Volatility"].values
        corr = np.identity(len(weights_to_use))
        cov = np.outer(vols, vols) * corr
        steps_per_year = {"monthly": 12, "quarterly": 4, "yearly": 1}[frequency]
        n_steps = n_years * steps_per_year
        dt = 1 / steps_per_year

        chol = np.linalg.cholesky(cov)
        portfolio_paths = np.zeros((n_steps + 1, n_sims))
        portfolio_paths[0] = initial_value

        for sim in range(n_sims):
            prices = np.ones(len(weights_to_use))
            path = [initial_value]
            for _ in range(n_steps):
                z = t.rvs(fat_tail_df, size=len(weights_to_use))
                correlated_z = chol @ z
                prices *= np.exp(mu * dt + correlated_z * np.sqrt(dt))
                path.append(np.dot(prices, weights_to_use) * initial_value)
            portfolio_paths[:, sim] = path

        st.session_state["portfolio_paths"] = portfolio_paths
        st.session_state["x_axis"] = pd.date_range(start=start_date, periods=n_steps + 1, freq="ME")

        st.write("**Weights Used in Simulation:**")
        st.dataframe(pd.DataFrame({"Weight": weights_to_use}, index=ltcma_df.index))


        final_values = portfolio_paths[-1]
        steps_for_var = int(var_years * steps_per_year)
        if steps_for_var > n_steps:
            st.warning("VaR horizon exceeds simulation length. Increase investment horizon or reduce VaR years.")
            var_values = final_values
        else:
            var_values = portfolio_paths[steps_for_var]

 

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
        elif "ltcma_df" not in st.session_state:
            st.error("LTCMA data not available. Please upload or define LTCMA first.")
        else:
            ltcma_df = st.session_state["ltcma_df"]
            st.subheader(f"Impact of Historical Scenario: {selected_scenario}")

            scenario_row = scenarios_df[scenarios_df.iloc[:, 0].astype(str) == selected_scenario]
            if scenario_row.empty:
                st.error("Selected scenario not found in the scenario file.")
            else:
                scenario_row = scenario_row.iloc[0, 1:]

                if len(scenario_row) != len(ltcma_df):
                    st.error("Scenario does not match number of asset classes in LTCMA.")
                else:

                    # NEW LOGIC: use optimized weights if checkbox is checked and weights are available
                    if use_opt_in_scenario and "optimized_weights" in st.session_state:
                        weights_used = st.session_state["optimized_weights"]
                    else:
                        weights_used = ltcma_df["SAA"].values
                    scenario_return = np.dot(scenario_row.values, weights_used)
                    impact_df = pd.DataFrame({
                        "Asset Class": scenario_row.index,
                        "Shock Return": scenario_row.values,
                        "Weight": pd.Series(weights_used, index=ltcma_df.index)
                    })
                    impact_df["Contribution"] = impact_df["Shock Return"] * impact_df["Weight"]

                    # Format numerical columns manually before displaying
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


#######################################################


if "portfolio_paths" in st.session_state and "x_axis" in st.session_state:
    portfolio_paths = st.session_state["portfolio_paths"]
    x_axis = st.session_state["x_axis"]

    tab1, tab2, tab3, tab4 = st.tabs(["Simulation Chart", "Summary Statistics", "Value at Risk", "Drawdown Analysis"])

    with tab1:
        p25 = np.percentile(portfolio_paths, 25, axis=1)
        p75 = np.percentile(portfolio_paths, 75, axis=1)
        p05 = np.percentile(portfolio_paths, 5, axis=1)
        p95 = np.percentile(portfolio_paths, 95, axis=1)
        median_path = np.median(portfolio_paths, axis=1)

        fig, ax = plt.subplots(figsize=(10, 6))

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
#            ax.axhline(2 * initial_value, color='red', linestyle='--', linewidth=1, label='2x Initial Value')
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
        ax.set_ylabel("Portfolio Value")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
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
            st.download_button(
                label="Download Simulation Paths (CSV)",
                data=pd.DataFrame(portfolio_paths, index=x_axis).to_csv().encode('utf-8'),
                file_name='simulated_paths.csv', key="download_path_2",
                mime='text/csv'
            )

    with tab2:
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
 #           ax_hist.hist(final_values, bins=50, color='skyblue', edgecolor='black')

            # Calculate 1st and 99th percentiles to limit x-axis range
            x_min, x_max = np.percentile(final_values, [1, 99])
            # Plot histogram within these bounds
            ax_hist.hist(final_values, bins=50, range=(0, x_max*1.5), color='skyblue', edgecolor='black')
            # Set explicit x-axis limits
            ax_hist.set_xlim(0, x_max*1.5)
            
            ax_hist.set_title("Distribution of Final Portfolio Values")
            ax_hist.set_xlabel("Final Value")
            ax_hist.set_ylabel("Frequency")
       #    st.pyplot(fig_hist)

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

 #               ax_var.hist(var_values, bins=50, color='lightgrey', edgecolor='black')

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
                    rec_time_for_max = np.nan  # IMPORTANT: reset so we don’t carry over a prior recovery

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
            




