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
- move some charts from Matplotlib to Plotly, to get better interactions for users
        Simulation chart: DONE
        v5.2 makes it optional, with a toggle between the Matplotlib and the Plotly version
        
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

-v5.3: Add a password
-v5.4: Add a Username/password with hashed value for stronger security
-v6 new user interface for board members

Author:
-------
[Olivier Couvreur]
[04/08/2025]

"""


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from plotly import graph_objects as go
from io import BytesIO
from scipy.optimize import minimize
from scipy.stats import t
from pathlib import Path  
import pickle
import base64
import hashlib, hmac, binascii  
from base64 import b64encode
import time
from urllib.parse import quote_plus
from collections.abc import Mapping
import pyotp
import json, secrets, re
import uuid, datetime as dt
from urllib.parse import quote as urlquote

AUTH_STORE_PATH = "Data/auth_store.json"


APP_VERSION = "v8.0.1"

# ---- default data files (edit paths as needed) ----
DEFAULT_LTCMA_PATH = "Data/LTCMA.xlsx"
DEFAULT_CORR_PATH = "Data/Correlation Matrix.xlsx"
DEFAULT_SCENARIO_PATH = "Data/Scenarios.xlsx"

DEFAULT_BASELINE_SESSION_PATH = "Data/baseline_session.pkl" 
DEFAULT_BASELINE_SCENARIO_PATH = "Data/Scenarios.xlsx"

# --- Admin & reset helpers ---

DATA_DIR = Path("Data")
RESET_REQUESTS_PATH = DATA_DIR / "pw_reset_requests.json"

def _json_load(path: Path) -> dict:
    try: return json.loads(path.read_text())
    except Exception: return {}

def _json_save(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))

def add_reset_request(username: str) -> str:
    """Create/overwrite a reset request for username; returns a short admin token."""
    db = _json_load(RESET_REQUESTS_PATH)
    token = secrets.token_urlsafe(12)
    db[username] = {"ts": int(time.time()), "token": token}
    _json_save(RESET_REQUESTS_PATH, db)
    return token

def pop_reset_request(username: str):
    """Consume & remove a reset request if present."""
    db = _json_load(RESET_REQUESTS_PATH)
    rec = db.pop(username, None)
    _json_save(RESET_REQUESTS_PATH, db)
    return rec

def list_reset_requests() -> dict:
    return _json_load(RESET_REQUESTS_PATH)

def pbkdf2_make(password: str, salt_hex: str | None = None, iterations: int = 150_000) -> tuple[str, str, int]:
    """Produce (salt_hex, hash_hex, iterations) for storage in secrets.toml."""
    if not salt_hex:
        salt_hex = secrets.token_hex(16)   # 16 bytes -> 32 hex chars
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac("sha256", (password or "").encode("utf-8"), salt, iterations)
    return salt_hex, dk.hex(), iterations

def new_totp_secret() -> str:
    return pyotp.random_base32()

def totp_provisioning_uri(username: str, secret: str, issuer: str = "SAA App") -> str:
    return pyotp.TOTP(secret).provisioning_uri(name=username, issuer_name=issuer)

def toml_user_block(username: str, salt_hex: str, hash_hex: str, iterations: int, role: str, totp_secret: str) -> str:
    # Keep EXACT quoting/format to avoid TOML parse errors
    return (
        f'[auth.users.{username}]\n'
        f'salt = "{salt_hex}"\n'
        f'hash = "{hash_hex}"\n'
        f'iterations = {iterations}\n'
        f'role = "{role}"\n'
        f'totp_secret = "{totp_secret}"\n'
    )


def ensure_corr_shape(assets: pd.Index, corr_df: pd.DataFrame | None) -> pd.DataFrame:
    if corr_df is None or getattr(corr_df, "empty", True):
        return pd.DataFrame(np.eye(len(assets)), index=assets, columns=assets)
    out = corr_df.reindex(index=assets, columns=assets)
    out = out.fillna(0.0)
    np.fill_diagonal(out.values, 1.0)
    return out

def require_cap(flag: str):
    if not cap.get(flag, False):
        st.warning("You don’t have permission to perform this action.")
        st.stop()

        
st.set_page_config(layout="wide")
st.title("Portfolio Analytics")

if "view" not in st.session_state:
    st.session_state["view"] = "home"

def get_view_param(default: str = "home") -> str:
    return st.session_state.get("view", default)

def set_view_param(view: str):
    st.session_state["view"] = view
    try:
        st.query_params["view"] = view
    except Exception:
        qp = st.experimental_get_query_params()
        qp = {k: (v[0] if isinstance(v, list) else v) for k, v in qp.items()}
        qp["view"] = view
        st.experimental_set_query_params(**qp)

def _get_view_from_query():
    try:
        v = st.query_params.get("view", None)
    except Exception:
        v = st.experimental_get_query_params().get("view", [None])
        v = v[0] if isinstance(v, list) and v else None
    return v

_qv = _get_view_from_query()
if _qv is not None:  # accept "home", "sim", etc.
    st.session_state["view"] = _qv


def _get_qp_single(key: str):
    try:
        val = st.query_params.get(key, None)  # Streamlit >= 1.30
        return val if isinstance(val, str) else (val[0] if isinstance(val, list) and val else None)
    except Exception:
        qp = st.experimental_get_query_params()
        v = qp.get(key, [None])
        return v[0] if isinstance(v, list) and v else None


def apply_sim_overrides_from_query():
    sd = _get_qp_single("sd")
    if sd:
        try:
            st.session_state["start_date"] = pd.to_datetime(sd).date()
        except Exception:
            pass

    yrs = _get_qp_single("yrs")
    if yrs:
        try: st.session_state["n_years"] = int(yrs)
        except: pass

    freq = _get_qp_single("freq")
    if freq in ("monthly", "quarterly", "yearly"):
        st.session_state["frequency"] = freq

    iv = _get_qp_single("iv")
    if iv:
        try: st.session_state["initial_value"] = float(iv)
        except: pass

    ns = _get_qp_single("ns")
    if ns:
        try: st.session_state["n_sims"] = int(ns)
        except: pass

    uopt = _get_qp_single("uopt")
    if uopt is not None:
        st.session_state["use_optimized_weights"] = (uopt in ("1","true","True","yes","y"))

    scen = _get_qp_single("scen")
    if scen:
        st.session_state["viewer_scenario_select"] = scen
        
if not st.session_state.get("_qp_applied_once", False):
    apply_sim_overrides_from_query()
    st.session_state["_qp_applied_once"] = True


def apply_session_dict(session_data: dict, *, overwrite_params: bool = False):
    """Apply a session dict (ltcma_df, corr_matrix, sim_params, optimized_weights) into state."""
    assert "ltcma_df" in session_data and "sim_params" in session_data, "Invalid session file"
    st.session_state["ltcma_base_default"] = session_data["ltcma_df"].copy()
    st.session_state["corr_store"] = session_data["corr_matrix"].copy()
    st.session_state["corr_assets"] = tuple(session_data["ltcma_df"].index.tolist())
    st.session_state.pop("ltcma_widget", None)
    st.session_state.pop("corr_widget", None)
    for k in ["portfolio_paths", "x_axis", "optimized_weights", "prev_ltcma_df", "prev_corr_df"]:
        st.session_state.pop(k, None)
    sim_params = session_data.get("sim_params", {})
    for param_key, param_value in sim_params.items():
        if overwrite_params:
            st.session_state[param_key] = param_value
        else:
            if param_key not in st.session_state:
                st.session_state[param_key] = param_value
    if session_data.get("optimized_weights") is not None:
        st.session_state["optimized_weights"] = session_data["optimized_weights"].copy()


def load_baseline_session_and_scenarios(*, overwrite_params: bool = True):
    """Load the baseline .pkl session from disk + baseline scenarios Excel."""
    try:
        with open(DEFAULT_BASELINE_SESSION_PATH, "rb") as f:
            session_data = pickle.load(f)
        apply_session_dict(session_data, overwrite_params=overwrite_params)
        scen_df = pd.read_excel(DEFAULT_BASELINE_SCENARIO_PATH)
        st.session_state["default_scenarios_df"] = scen_df.copy()
        if overwrite_params:
            st.session_state["_qp_applied_once"] = True
        st.success("Baseline session and scenarios loaded.")
        st.rerun()
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
    except AssertionError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Failed to load baseline: {e}")


def load_image_b64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return b64encode(f.read()).decode()
    except Exception:
        # fallback tiny transparent pixel if image missing
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAMAASsJTYQAAAAASUVORK5CYII="


def image_tile(label: str, img_path: str, target_view: str, height_px: int = 220):
    img_b64 = load_image_b64(img_path)
    qs = auth_query_suffix()
    sd = st.session_state.get("start_date")
    try:
        sd_str = pd.to_datetime(sd).date().isoformat()
    except Exception:
        sd_str = str(sd)

    yrs  = st.session_state.get("n_years")
    freq = st.session_state.get("frequency")
    iv   = st.session_state.get("initial_value")
    ns   = st.session_state.get("n_sims")
    uopt = 1 if st.session_state.get("use_optimized_weights") else 0
    scen_sel = st.session_state.get("viewer_scenario_select")
    scen_q = f"&scen={quote_plus(str(scen_sel))}" if scen_sel else ""
    extra = f"&sd={sd_str}&yrs={yrs}&freq={freq}&iv={iv}&ns={ns}&uopt={uopt}{scen_q}"

    st.markdown(
        f"""
        <a href="?view={target_view}{qs}{extra}" target="_self" style="text-decoration:none;">
          <div style="
              position:relative;
              width:100%;
              height:{height_px}px;
              border-radius:16px;
              overflow:hidden;
              box-shadow:0 6px 18px rgba(0,0,0,.15);
              background-image:url('data:image/png;base64,{img_b64}');
              background-size:cover;
              background-position:center;
              filter:saturate(1.02);
          ">
            <div style="
                position:absolute; left:0; right:0; bottom:0;
                padding:12px 16px;
                background:linear-gradient(180deg, rgba(0,0,0,0) 0%, rgba(0,0,0,.55) 70%);
                color:white; font-weight:700; font-size:1.05rem; letter-spacing:.2px;
            ">
              {label}
            </div>
          </div>
        </a>
        """,
        unsafe_allow_html=True
    )


def viewer_back_link():
    if st.button("◀ Back to menu", key="back_to_menu_btn"):
        set_view_param("home")
        st.rerun()


def render_logo_top_right(img_path: str, height_px: int = 42):
    img_b64 = load_image_b64(img_path)
    st.markdown(
        f"""
        <div style="
            width:100%;
            display:flex;
            justify-content:flex-end;
            margin: -6px 0 10px 0;
        ">
            <img
                src="data:image/png;base64,{img_b64}"
                alt="Logo"
                style="
                    height: clamp(32px, 8vw, {height_px}px);
                    border-radius: 6px;
                    box-shadow: 0 2px 8px rgba(0,0,0,.18);
                "
            />
        </div>
        """,
        unsafe_allow_html=True,
    )


def center_plot(fig, ratio=(1, 4, 1), figsize=(7, 4)):
    if figsize:
        fig.set_size_inches(*figsize)
    left, mid, right = st.columns(ratio)
    with mid:
        st.pyplot(fig)


def get_user_record(username: str):
    uname = (username or "").strip().lower()
    # 1) Writable JSON overrides
    store = _load_auth_store()
    rec = (store.get("users", {}) or {}).get(uname)
    if rec:
        return rec
    # 2) Fall back to secrets.toml
    auth = st.secrets.get("auth", {})
    users = dict(auth.get("users", {}))
    return users.get(uname)


def verify_password_pbkdf2(username: str, password: str) -> bool:
    rec = get_user_record(username)
    if not rec:
        st.warning("User not found.")
        return False

    def _clean_hex(s: str) -> str:
        # remove spaces, zero-width chars, BOMs, and optional "0x" prefix
        return (
            str(s)
            .strip()
            .replace(" ", "")
            .replace("\u200b", "")
            .replace("\ufeff", "")
            .lower()
            .removeprefix("0x")
        )

    try:
        salt_hex = _clean_hex(rec["salt"])
        hash_hex = _clean_hex(rec["hash"])
        salt = bytes.fromhex(salt_hex)              # parse, don't pre-judge length
        expected = bytes.fromhex(hash_hex)
        iterations = int(rec.get("iterations", 150_000))
    except Exception as e:
        st.error(f"Secrets format looks off: {e}")
        return False

    # PBKDF2-HMAC-SHA256 yields 32 bytes; enforce that with a clearer message
    if len(expected) != 32:
        st.error(f"Hash must be 64 hex chars (got {len(hash_hex)}).")
        return False

    dk = hashlib.pbkdf2_hmac("sha256", (password or "").encode("utf-8"), salt, iterations)
    return hmac.compare_digest(dk, expected)


REQUIRE_2FA = bool(st.secrets.get("auth", {}).get("require_2fa", True))

def verify_totp(username: str, code: str) -> bool:
    """Verify a 6-digit TOTP code for the given user."""
    rec = get_user_record(username)
    if not rec:
        return False
    secret = (rec.get("totp_secret") or "").strip()
    if not secret:
        # If 2FA is required globally but user has no secret, do not allow bypass.
        return not REQUIRE_2FA
    try:
        totp = pyotp.TOTP(secret)
        # Accept codes in neighboring step to be tolerant of slight clock drift
        code_digits = "".join(ch for ch in str(code) if ch.isdigit())
        return bool(totp.verify(code_digits, valid_window=1))
    except Exception:
        return False


def _auth_signing_key() -> bytes:
    return (st.secrets.get("auth", {}).get("signing_key") or "dev-signing-key").encode("utf-8")

def _sign_token(s: str) -> str:
    return hmac.new(_auth_signing_key(), s.encode("utf-8"), hashlib.sha256).hexdigest()

def _get_qp(key: str):
    try:
        val = st.query_params.get(key, None)
        return val if isinstance(val, str) else (val[0] if isinstance(val, list) and val else None)
    except Exception:
        qp = st.experimental_get_query_params()
        v = qp.get(key)
        if v is None:
            return None
        return v if isinstance(v, str) else (v[0] if isinstance(v, list) and v else None)

def try_autologin_from_query() -> bool:
    u = (_get_qp("auth_user") or "").strip().lower()
    exp = _get_qp("auth_exp")
    sig = _get_qp("auth_sig")
    if not u or not exp or not sig:
        return False
    try:
        exp_i = int(exp)
    except Exception:
        return False
    if exp_i < int(time.time()):
        return False
    token = f"{u}|{exp_i}"
    if not hmac.compare_digest(sig, _sign_token(token)):
        return False
    rec = get_user_record(u)
    if not rec:
        return False
    st.session_state.auth = {"ok": True, "user": u, "role": rec.get("role", "user")}
    return True


def issue_autologin_token(username: str, ttl_seconds: int = 8 * 3600):
    exp = int(time.time()) + ttl_seconds
    token = f"{username}|{exp}"
    sig = _sign_token(token)
    try:
        st.query_params.update({"auth_user": username, "auth_exp": str(exp), "auth_sig": sig})
    except Exception:
        current = st.experimental_get_query_params()
        current = {k: (v[0] if isinstance(v, list) else v) for k, v in current.items()}
        current.update({"auth_user": username, "auth_exp": str(exp), "auth_sig": sig})
        st.experimental_set_query_params(**current)


def auth_query_suffix() -> str:
    """Return '&auth_user=...&auth_exp=...&auth_sig=...' if present, else ''.
       Use it to append to <a href> so tokens survive hard navigations.
    """
    u, e, s = _get_qp("auth_user"), _get_qp("auth_exp"), _get_qp("auth_sig")
    if u and e and s:
        return f"&auth_user={u}&auth_exp={e}&auth_sig={s}"
    a = st.session_state.get("auth", {})
    if a and a.get("ok") and a.get("user"):
        return ""
    return ""


def _load_auth_store() -> dict:
    p = Path(AUTH_STORE_PATH)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_auth_store(store: dict) -> bool:
    try:
        p = Path(AUTH_STORE_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(store, indent=2), encoding="utf-8")
        tmp.replace(p)
        return True
    except Exception as e:
        st.error(f"Failed to write auth store: {e}")
        return False


def _pbkdf2_hash(password: str, *, salt_hex: str | None = None, iterations: int = 150_000):
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return salt.hex(), dk.hex(), iterations

def _password_is_strong(pw: str) -> tuple[bool, str]:
    if len(pw) < 10:
        return False, "Must be at least 10 characters."
    classes = sum(bool(re.search(r, pw)) for r in [r"[a-z]", r"[A-Z]", r"\d", r"[^A-Za-z0-9]"])
    if classes < 3:
        return False, "Use at least three of: lowercase, uppercase, number, symbol."
    return True, ""

def set_user_password(username: str, new_password: str) -> bool:
    uname = (username or "").strip().lower()
    rec = get_user_record(uname)
    if not rec:
        st.error("User not found.")
        return False

    ok, msg = _password_is_strong(new_password)
    if not ok:
        st.error(msg)
        return False

    iterations = int(rec.get("iterations", 150_000))
    salt_hex, hash_hex, iterations = _pbkdf2_hash(new_password, iterations=iterations)

    # merge with existing fields so we keep role / totp_secret / etc.
    new_rec = dict(rec)
    new_rec.update({"salt": salt_hex, "hash": hash_hex, "iterations": iterations, "must_change": False})

    store = _load_auth_store()
    users = store.setdefault("users", {})
    users[uname] = new_rec
    return _save_auth_store(store)

def _read_tabular_file(uploaded_file: "UploadedFile") -> pd.DataFrame:
    name = (uploaded_file.name or "").lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    return df

def _find_date_and_value_cols(df: pd.DataFrame) -> tuple[str, str, str]:
    """
    Return (date_col, numeric_col, kind) where kind in {"value","return"}.
    - If a column name contains 'return', treat as returns (fraction or %).
    - Otherwise use the first numeric column as 'value'.
    """
    cols = [c for c in df.columns if str(c).strip() != ""]
    # pick date-like col (prefer 'date' names)
    date_candidates = [c for c in cols if "date" in str(c).lower()]
    date_col = date_candidates[0] if date_candidates else cols[0]

    # numeric candidates
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        # try to coerce all but date to numeric
        for c in cols:
            if c == date_col: 
                continue
            try:
                pd.to_numeric(df[c])
                num_cols.append(c)
            except Exception:
                pass
    if not num_cols:
        raise ValueError("No numeric column found for values/returns.")

    # prefer 'return' if present
    ret_cols = [c for c in num_cols if "return" in str(c).lower()]
    if ret_cols:
        return date_col, ret_cols[0], "return"
    else:
        return date_col, num_cols[0], "value"

def parse_actual_upload(uploaded_file) -> dict:
    """
    Returns a dict with:
      - 'base1': pd.Series indexed by datetime, normalized to base 1.0
      - 'first_value': original first numeric level (if kind='value'), else 1.0 for returns
      - 'kind': 'value' or 'return'
      - 'start_end': (start_date, end_date)
    """
    df = _read_tabular_file(uploaded_file)
    date_col, num_col, kind = _find_date_and_value_cols(df)

    out = df[[date_col, num_col]].dropna()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).sort_values(date_col)
    out = out.set_index(date_col)

    series = pd.to_numeric(out[num_col], errors="coerce").dropna()

    if kind == "return":
        # returns may be in %, detect and convert if needed
        med = series.abs().median()
        if med > 0.5:   # probably given in %
            series = series / 100.0
        base1 = (1.0 + series).cumprod()
        first_value = 1.0
    else:
        # level series -> normalize to base 1.0
        first_value = float(series.iloc[0])
        base1 = series / first_value

    base1.name = "Actual (base=1)"
    return {
        "base1": base1,
        "first_value": first_value,
        "kind": kind,
        "start_end": (base1.index.min(), base1.index.max()),
    }

def _resample_to_sim_freq(s: pd.Series, frequency: str) -> pd.Series:
    """Resample to month/quarter/year end."""
    freq_map = {"monthly": "ME", "quarterly": "QE", "yearly": "YE"}
    freq = freq_map.get(frequency, "M")
    try:
        return s.resample(freq).last().dropna()
    except Exception:
        return s



# @st.cache_data  # cache suggested by Claude, try without it
def compute_drawdown_stats_vectorized(portfolio_paths_tuple):
    """Vectorized drawdown calculation - much faster"""
    portfolio_paths = np.array(portfolio_paths_tuple)
    n_steps, n_sims = portfolio_paths.shape
    
    # Calcul vectorisé des peaks cumulatifs
    cummax = np.maximum.accumulate(portfolio_paths, axis=0)
    
    # Drawdowns pour chaque pas de temps
    drawdowns = 1.0 - portfolio_paths / cummax
    
    # Max drawdown par simulation
    max_drawdowns = drawdowns.max(axis=0).tolist()
    
    # Recovery times (plus complexe, garde une boucle légère)
    recovery_times = []
    for sim in range(n_sims):
        path = portfolio_paths[:, sim]
        dd = drawdowns[:, sim]
        max_dd_idx = dd.argmax()
        
        # Find when it recovers
        peak_at_max_dd = cummax[max_dd_idx, sim]
        recovery_idx = np.where(path[max_dd_idx:] >= peak_at_max_dd)[0]
        
        if len(recovery_idx) > 0:
            recovery_times.append(float(recovery_idx[0]))
        else:
            recovery_times.append(np.nan)
    
    return max_drawdowns, recovery_times






AUTH_REQUIRED = bool(st.secrets.get("auth", {}).get("require", False))

if AUTH_REQUIRED:
    if "auth" not in st.session_state:
        st.session_state.auth = {"ok": False, "user": None, "role": None}

    if not st.session_state.auth["ok"]:
        try_autologin_from_query()


    if not st.session_state.auth["ok"]:
        # restore from URL if possible (keeps your existing behavior)
        try_autologin_from_query()

    if not st.session_state.auth["ok"]:
        st.subheader("Restricted access")

        # Which screen are we on? 'login' (user+pass) or '2fa' (one-time code)
        stage = st.session_state.setdefault("auth_stage", "login")


        if stage == "login":
            colU, colP = st.columns([1, 1])
            with colU:
                username = st.text_input("Username", key="login_username").strip().lower()
            with colP:
                password = st.text_input("Password", type="password", key="login_password")

            trust_device = st.checkbox("Trust this device", value=True,
                                   help="Keeps you signed in with a secure token.")

            if st.button("Sign in"):


                if verify_password_pbkdf2(username, password):
                    rec = get_user_record(username)
                    has_totp = bool(rec and rec.get("totp_secret"))
                    need_2fa = REQUIRE_2FA and has_totp  # only require 2FA if globally on AND the user has a secret

                    if REQUIRE_2FA and not has_totp:
                        st.error("2FA is required but not configured for this account. Ask an admin to add a TOTP secret.")
                        st.stop()

                    if not need_2fa:
                        # finish login immediately (password-only path)
                        role = rec.get("role", "user")
                        st.session_state.auth = {"ok": True, "user": username, "role": role}

                        # auto-load baseline for analyst/viewer (keep your behavior)
                        if role in ("analyst", "viewer"):
                            st.session_state["_auto_baseline_after_login"] = True

                        ttl = (7 * 24 * 3600) if trust_device else (8 * 3600)
                        issue_autologin_token(username, ttl_seconds=ttl)
                        st.rerun()
                    else:
                        # proceed to 2FA screen
                        st.session_state["pending_user"] = username
                        st.session_state["pending_ttl"] = (7 * 24 * 3600) if trust_device else (8 * 3600)
                        st.session_state["auth_stage"] = "2fa"
                        st.rerun()










            # under the login form, add:
            forgot_col, _ = st.columns([1, 3])
            with forgot_col:
                if st.button("Forgot password?"):
                    st.session_state["auth_stage"] = "forgot"
                    st.rerun()


            st.stop()

        elif stage == "2fa":
            pending_user = st.session_state.get("pending_user")
            if not pending_user:
                # Safety fallback
                st.session_state["auth_stage"] = "login"
                st.rerun()

            st.write(f"User: **{pending_user}**")
            totp_code = st.text_input("Authenticator code", type="password", max_chars=8, key="totp_input")

            colA, colB = st.columns([1, 1])
            with colA:
                if st.button("Verify code"):
                    if verify_totp(pending_user, totp_code):
                        rec = get_user_record(pending_user)
                        role = rec.get("role", "user")
                        st.session_state.auth = {"ok": True, "user": pending_user, "role": role}

                        # auto-load baseline for analyst/viewer
                        if role in ("analyst", "viewer"):
                            st.session_state["_auto_baseline_after_login"] = True

                        # finalize token
                        ttl = int(st.session_state.get("pending_ttl", 8 * 3600))
                        issue_autologin_token(pending_user, ttl_seconds=ttl)

                        # clean up stage state
                        st.session_state.pop("pending_user", None)
                        st.session_state.pop("pending_ttl", None)
                        st.session_state.pop("auth_stage", None)

                        # after successful TOTP verify, before final st.rerun()
                        rec = get_user_record(pending_user)
                        if rec.get("must_change"):
                            # park the user in a temporary stage to set a new password
                            st.session_state["auth_stage"] = "pwreset"
                            st.session_state["pending_user"] = pending_user
                            st.stop()
                        
                        st.rerun()
                    else:
                        st.error("Invalid or expired code. Please try again.")

            with colB:
                if st.button("Back"):
                    st.session_state["auth_stage"] = "login"
                    st.session_state.pop("pending_user", None)
                    st.session_state.pop("pending_ttl", None)
                    st.rerun()

            st.stop()

        
        elif stage == "pwreset":
            pending_user = st.session_state.get("pending_user")
            st.subheader("Set a new password")
            with st.form("forced_pw_reset", clear_on_submit=True):
                new1 = st.text_input("New password", type="password")
                new2 = st.text_input("Confirm new password", type="password")
                submit = st.form_submit_button("Save new password")
            if submit:
                if new1 != new2:
                    st.error("Passwords do not match.")
                else:
                    if set_user_password(pending_user, new1):
                        # finish sign-in like normal
                        rec = get_user_record(pending_user)
                        role = rec.get("role", "user")
                        st.session_state.auth = {"ok": True, "user": pending_user, "role": role}
                        issue_autologin_token(pending_user, ttl_seconds=int(st.session_state.get("pending_ttl", 8 * 3600)))
                        st.session_state.pop("pending_user", None)
                        st.session_state.pop("pending_ttl", None)
                        st.session_state.pop("auth_stage", None)
                        st.success("Password updated.")
                        st.rerun()
        

        elif stage == "forgot":
            st.write("Reset your password")
            uname = st.text_input("Username", key="forgot_user").strip().lower()
            if st.button("Request reset"):
                if get_user_record(uname):
                    token = add_reset_request(uname)
                    st.success("Request recorded. Please contact an admin to finalize your reset.")
                    # Show a short token the admin will see in their panel
                    st.caption("Give this token to your admin (or they’ll see it in the panel):")
                    st.code(token, language="text")
                else:
                    st.error("Unknown username.")
            if st.button("Back to sign in"):
                st.session_state["auth_stage"] = "login"
                st.rerun()
            st.stop()

        


    if st.session_state.pop("_auto_baseline_after_login", False):
        load_baseline_session_and_scenarios(overwrite_params=True)
        st.stop()

    if st.session_state.auth.get("role") in ("analyst", "viewer") \
       and "ltcma_base_default" not in st.session_state \
       and not st.session_state.get("_baseline_attempted", False):
        st.session_state["_baseline_attempted"] = True
        load_baseline_session_and_scenarios(overwrite_params=False)
        st.stop()

    st.sidebar.success(f"Signed in as {st.session_state.auth['user']} ({st.session_state.auth['role']})")

    # after: st.sidebar.success(f"Signed in as ...")
    with st.sidebar.expander("Account"):
        st.markdown("**Change password**")
        with st.form("change_pw_form", clear_on_submit=True):
            curr = st.text_input("Current password", type="password")
            new1 = st.text_input("New password", type="password")
            new2 = st.text_input("Confirm new password", type="password")
            submit = st.form_submit_button("Update password")

        if submit:
            user = st.session_state.auth["user"]
            if not verify_password_pbkdf2(user, curr):
                st.error("Current password is incorrect.")
            elif new1 != new2:
                st.error("New passwords do not match.")
            else:
                if set_user_password(user, new1):
                    st.success("Password updated. It takes effect immediately.")

    ROLE = (st.session_state.auth["role"] if AUTH_REQUIRED else "admin")
    if ROLE == "admin":
        with st.sidebar.expander("Admin • Users"):
            target = st.text_input("Username to force reset", key="force_reset_user")
            if st.button("Require password change"):
                store = _load_auth_store()
                users = store.setdefault("users", {})
                rec = users.get(target.strip().lower()) or get_user_record(target)
                if not rec:
                    st.error("User not found in store or secrets.")
                else:
                    # write to store (so it becomes writable/overridable)
                    out = dict(rec); out["must_change"] = True
                    users[target.strip().lower()] = out
                    if _save_auth_store(store):
                        st.success("User will be required to set a new password on next sign-in.")


    # --- Admin: Users & Roles ---
    if ROLE == "admin":
        with st.sidebar.expander("Admin: Users & Roles", expanded=False):

            # 3.1 Create user
            st.markdown("**Create user**")
            cu1, cu2 = st.columns([1, 1])
            with cu1:
                new_username = st.text_input("Username (lowercase)", key="adm_new_user").strip().lower()
            with cu2:
                new_role = st.selectbox("Role", ["admin", "analyst", "viewer"], key="adm_new_role")

            tmp_pw = st.text_input("Temporary password", type="password", key="adm_new_tmp_pw")
            regen2fa = st.checkbox("Generate 2FA secret (recommended)", value=True, key="adm_new_2fa")

            if st.button("Generate user TOML"):
                if not new_username or not tmp_pw:
                    st.warning("Username and temporary password required.")
                else:
                    salt_hex, hash_hex, iters = pbkdf2_make(tmp_pw)
                    totp_secret = new_totp_secret() if regen2fa else (get_user_record(new_username) or {}).get("totp_secret", "") or new_totp_secret()
                    uri = totp_provisioning_uri(new_username, totp_secret, issuer="SAA App")
                    block = toml_user_block(new_username, salt_hex, hash_hex, iters, new_role, totp_secret)
    
                    st.success("Paste this block into your secrets.toml (exactly):")
                    st.code(block, language="toml")

                    # Optional QR via Google Charts (note: requests an external image)
                    qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=180x180&data={urlquote(uri)}"
                    st.markdown(f"**TOTP secret:** `{totp_secret}`")
                    st.markdown(f"[Provisioning URI]({uri})")
                    st.image(qr_url, caption="Scan in Microsoft Authenticator / Google Authenticator")

            st.divider()

            # 3.2 Change role of an existing user
            st.markdown("**Change user role**")
            all_users = sorted((st.secrets.get("auth", {}).get("users", {}) or {}).keys())
            sel_user = st.selectbox("User", all_users, key="adm_role_user") if all_users else None
            new_role2 = st.selectbox("New role", ["admin", "analyst", "viewer"], key="adm_role_role")
            if st.button("Generate role update TOML", disabled=not sel_user):
                rec = get_user_record(sel_user)
                if not rec:
                    st.error("User not found in secrets.")
                else:
                    block = toml_user_block(
                        sel_user,
                        rec["salt"].strip(),
                        rec["hash"].strip(),
                        int(rec.get("iterations", 150_000)),
                        new_role2,
                        (rec.get("totp_secret") or "").strip()
                    )
                    st.success("Paste this block (role updated) into secrets.toml:")
                    st.code(block, language="toml")

            st.divider()

            # 3.3 Service password reset requests
            st.markdown("**Password reset requests**")
            reqs = list_reset_requests()
            if not reqs:
                st.caption("No pending requests.")
            else:
                # pick first or choose one
                user_reset = st.selectbox("Pending user", sorted(reqs.keys()), key="adm_reset_user")
                token_shown = reqs.get(user_reset, {}).get("token", "")
                st.caption(f"Verification token for this request: `{token_shown}`")

                new_tmp_pw = st.text_input("New temporary password", type="password", key="adm_reset_tmp_pw")
                reset_2fa = st.checkbox("Reset 2FA (user lost authenticator)", value=False, key="adm_reset_re2fa")

                if st.button("Generate reset TOML", disabled=not user_reset):
                    rec = get_user_record(user_reset)
                    if not rec:
                        st.error("User not found in secrets.")
                    elif not new_tmp_pw:
                        st.error("Enter a temporary password.")
                    else:
                        salt_hex, hash_hex, iters = pbkdf2_make(new_tmp_pw)
                        totp_secret = new_totp_secret() if reset_2fa else (rec.get("totp_secret") or new_totp_secret())
                        uri = totp_provisioning_uri(user_reset, totp_secret, issuer="SAA App")
                        block = toml_user_block(user_reset, salt_hex, hash_hex, iters, rec.get("role", "viewer"), totp_secret)

                        st.success("Paste this block into secrets.toml to complete the reset:")
                        st.code(block, language="toml")
                        st.markdown(f"**TOTP secret:** `{totp_secret}`")
                        st.markdown(f"[Provisioning URI]({uri})")
                        qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=180x180&data={urlquote(uri)}"
                        st.image(qr_url, caption="Scan in authenticator")
    
                        # mark the request as done
                        pop_reset_request(user_reset)





    if st.sidebar.button("Sign out"):
        curr_view = st.session_state.get("view", "home")
        st.session_state.pop("auth", None)
        try:
            st.query_params.update({"auth_user": None, "auth_exp": None, "auth_sig": None, "view": curr_view})
        except Exception:
            qp = st.experimental_get_query_params()
            qp = {k: (v[0] if isinstance(v, list) else v) for k, v in qp.items()}
            qp.pop("auth_user", None); qp.pop("auth_exp", None); qp.pop("auth_sig", None)
            qp["view"] = curr_view
            st.experimental_set_query_params(**qp)
        st.rerun()


ROLE = (st.session_state.auth["role"] if AUTH_REQUIRED else "admin")

CAPS = {
    "admin":   {"edit_ltcma": True,  "optimize": True,  "scenarios": True,  "downloads": True},
    "analyst": {"edit_ltcma": True,  "optimize": True,  "scenarios": True,  "downloads": True},
    "viewer":  {"edit_ltcma": False, "optimize": False, "scenarios": True,  "downloads": False},
}
cap = CAPS.get(ROLE, CAPS["viewer"])


IS_VIEWER = (ROLE == "viewer")
VIEW = get_view_param("home")

render_logo_top_right("Data/Logo.png", height_px=120)  # size of the logo


def _invalidate_sim():
    for k in ("portfolio_paths", "x_axis", "fig_hist", "buf_hist", "fig_var", "buf_var",
              "fig_ddist", "buf_ddist"):
        st.session_state.pop(k, None)
    st.session_state.simulation_has_run = False


# Sidebar Inputs

# Initialize default session state
default_values = {
    "start_date": pd.to_datetime("2024-12-31"),
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

if "portfolio_paths" not in st.session_state:
    st.session_state["portfolio_paths"] = None

if "x_axis" not in st.session_state:
    st.session_state["x_axis"] = None

st.session_state.simulation_has_run = (
    st.session_state["portfolio_paths"] is not None
    and st.session_state["x_axis"] is not None
)


def get_current_ltcma_and_corr_for_save() -> tuple[pd.DataFrame, pd.DataFrame]:
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

    ltc = ltc.copy()
    ltc.index = ltc.index.astype(str).str.strip()
    ltc = ltc.loc[ltc.index != ""]

    corr = ensure_corr_shape(ltc.index, st.session_state.get("corr_store"))
    return ltc, corr



def serialize_current_session_for_pickle() -> dict:
    """Build a baseline-ready payload from the current UI/session."""
    ltcma_current, corr_current = get_current_ltcma_and_corr_for_save()

    opt = st.session_state.get("optimized_weights", None)
    if isinstance(opt, pd.Series):
        opt_to_save = opt.reindex(ltcma_current.index)
    elif isinstance(opt, np.ndarray) and len(opt) == len(ltcma_current.index):
        opt_to_save = pd.Series(opt, index=ltcma_current.index)
    else:
        opt_to_save = None

    sim_params = {
        "start_date": st.session_state["start_date"],
        "n_years": st.session_state["n_years"],
        "initial_value": st.session_state["initial_value"],
        "frequency": st.session_state["frequency"],
        "n_sims": st.session_state["n_sims"],
        "use_optimized_weights": st.session_state["use_optimized_weights"],
    }

    return {
        "ltcma_df": ltcma_current.copy(),
        "corr_matrix": corr_current.copy(),
        "optimized_weights": opt_to_save,
        "sim_params": sim_params,
    }

def write_baseline_session_from_state(path: str = DEFAULT_BASELINE_SESSION_PATH) -> bool:
    if ROLE != "admin":
        st.warning("Only admins can replace the baseline.")
        return False
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = serialize_current_session_for_pickle()
        with open(p, "wb") as f:
            pickle.dump(payload, f)
        return True
    except Exception as e:
        st.error(f"Failed to write baseline: {e}")
        return False

st.sidebar.header("Simulation Parameters")

if IS_VIEWER:
    if st.sidebar.button(
        "↩️ Reset to baseline",
        key="reset_baseline_viewer",
        help="Reload the baseline session (.pkl) and predefined scenarios",
        use_container_width=True,
    ):
        load_baseline_session_and_scenarios(overwrite_params=True)
else:
    with st.sidebar.expander("Session Management"):
        uploaded_session = st.file_uploader(
            "Reload Saved Session",
            type=["pkl"],
            key="session_uploader"
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Save Session", key="save_session_main"):
                ltcma_current, corr_current = get_current_ltcma_and_corr_for_save()
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
            if st.button(
                "↩️ Reset Session",
                key="restore_defaults_baseline",
                help="Load baseline .pkl session plus predefined scenarios"
            ):
                load_baseline_session_and_scenarios(overwrite_params=True)
       
        if ROLE == "admin":
            st.divider()
            st.markdown("#### Baseline (admin)")
            st.caption(f"Overwrite: **{DEFAULT_BASELINE_SESSION_PATH}**")
            confirm_overwrite = st.checkbox(
                "Yes, overwrite the baseline .pkl on disk",
                key="confirm_overwrite_baseline"
            )
            if st.button("Make current settings the baseline", disabled=not confirm_overwrite):
                if write_baseline_session_from_state():
                    st.success("Baseline updated.")
                    if st.button("Reload baseline now"):
                        load_baseline_session_and_scenarios(overwrite_params=True)

        if uploaded_session is not None and st.session_state.get("session_loaded") != uploaded_session.name:
            try:
                session_data = pickle.load(uploaded_session)
                assert "ltcma_df" in session_data and "sim_params" in session_data, "Invalid session file"

                st.session_state["ltcma_base_default"] = session_data["ltcma_df"].copy()
                st.session_state["corr_store"] = session_data["corr_matrix"].copy()
                st.session_state["corr_assets"] = tuple(session_data["ltcma_df"].index.tolist())

                st.session_state.pop("ltcma_widget", None)
                st.session_state.pop("corr_widget", None)

                for k in ["portfolio_paths", "x_axis", "optimized_weights", "prev_ltcma_df", "prev_corr_df"]:
                    st.session_state.pop(k, None)

                sim_params = session_data.get("sim_params", {})
                for param_key, param_value in sim_params.items():
                    st.session_state[param_key] = param_value

                st.success("Session reloaded successfully.")
                st.session_state["session_loaded"] = uploaded_session.name
                st.rerun()
            except Exception as e:
                st.error(f"Failed to reload session: {e}")
                st.stop()


start_date = st.sidebar.date_input(
    "Start Date",
    key="start_date",
    on_change=_invalidate_sim,
)


initial_value = st.sidebar.number_input(
    "Initial Portfolio Value",
    key="initial_value",
    on_change=_invalidate_sim,
)

if IS_VIEWER:
    frequency = st.session_state.setdefault("frequency", "monthly")
    st.sidebar.caption(f"🔒 Frequency: **{frequency.capitalize()}** (fixed)")
else:
    frequency = st.sidebar.selectbox(
        "Frequency",
        ["monthly", "quarterly", "yearly"],
        index=["monthly", "quarterly", "yearly"].index(st.session_state["frequency"]),
        key="frequency",
        on_change=_invalidate_sim,
    )

n_years = st.sidebar.slider(
    "Investment Horizon (Years)",
    1, 30,
    key="n_years",
    on_change=_invalidate_sim,
)

n_sims = st.sidebar.slider(
    "Number of Simulations",
    100, 5000,
    step=100,
    key="n_sims",
    on_change=_invalidate_sim,
)


if IS_VIEWER:
    with st.sidebar.expander("Optional Display Settings"):
        show_double_initial = st.checkbox(
            "Show Double Initial Value",
            value=st.session_state.get("show_double_initial", False),
            key="show_double_initial",
        )
        show_double_with_inflation = st.checkbox(
            "Show Double with Inflation",
            value=st.session_state.get("show_double_with_inflation", False),
            key="show_double_with_inflation",
        )
    chart_engine  = st.session_state.setdefault("chart_engine", "Plotly (interactive)")
    chart_height  = st.session_state.setdefault("chart_height", 620)
    use_log_scale = st.session_state.setdefault("use_log_scale", False)
    inflation_rate = st.session_state.setdefault("inflation_rate", 0.025)

else:
    with st.sidebar.expander("Optional Display Settings"):
        chart_engine = st.radio(
            "Chart engine",
            ["Plotly (interactive)", "Matplotlib (static)"],
            index=["Plotly (interactive)", "Matplotlib (static)"].index(
                st.session_state.get("chart_engine", "Plotly (interactive)")
            ),
            key="chart_engine",
            help="Plotly is touch-friendly; Matplotlib is simpler and light for small screens.",
        )
        chart_height = st.slider("Chart height (px)", 360, 900,
                                 value=st.session_state.get("chart_height", 620), step=10, key="chart_height")
        use_log_scale = st.checkbox("Log scale (Y axis)",
                                    value=st.session_state.get("use_log_scale", False), key="use_log_scale")
        show_double_initial = st.checkbox("Show Double Initial Value",
                                          value=st.session_state.get("show_double_initial", False), key="show_double_initial")
        show_double_with_inflation = st.checkbox("Show Double with Inflation",
                                                 value=st.session_state.get("show_double_with_inflation", False), key="show_double_with_inflation")
        inflation_rate = st.number_input("Inflation Rate (for compounding)",
                                         value=float(st.session_state.get("inflation_rate", 0.025)),
                                         step=0.001, format="%.3f", key="inflation_rate")



    with st.sidebar.expander("Actual portfolio (optional)"):
        uploaded_actual = st.file_uploader(
            "Upload CSV/XLSX with Date + Value or Date + Return",
            type=["csv", "xlsx"],
            key="actual_upload",
        )
        st.checkbox("Show actual portfolio on chart", value=True, key="actual_show")
        st.checkbox("Scale actuals to Initial Value", value=True, key="actual_scale_to_initial")

        if uploaded_actual is not None:
            try:
                parsed = parse_actual_upload(uploaded_actual)
                st.session_state["actual_series_base1"] = parsed["base1"]
                st.session_state["actual_first_value"] = parsed["first_value"]
                st.session_state["actual_kind"] = parsed["kind"]
                st.success(
                    f"Loaded actuals {parsed['kind']} from "
                    f"{parsed['start_end'][0].date()} to {parsed['start_end'][1].date()} "
                    f"({len(parsed['base1'])} points)."
                )
            except Exception as e:
                st.error(f"Could not parse actuals: {e}")

    
if not IS_VIEWER:
    with st.sidebar.expander("Value at Risk (VaR) Settings"):
        var_years      = st.slider("VaR Horizon (Years)", 1, 10, 1, key="var_years")
        var_confidence = st.slider("VaR Confidence Level (%)", 90, 99, 95, key="var_confidence")
        fat_tail_df    = st.slider("Tail Thickness (Student-t df)", 3, 30, 5, 1, key="fat_tail_df")
else:
    var_years      = st.session_state.setdefault("var_years", 1)
    var_confidence = st.session_state.setdefault("var_confidence", 95)
    fat_tail_df    = st.session_state.setdefault("fat_tail_df", 5)

    
if not IS_VIEWER:
    with st.sidebar.expander("Efficient Frontier Parameters"):
        risk_free_rate = st.number_input("Risk-Free Rate", value=0.02, step=0.001, format="%.3f", key="risk_free_rate")
        show_cml = st.checkbox("Show Capital Market Line (CML)", value=True, key="show_cml")
        show_saa = st.checkbox("Show Current SAA Portfolio", value=True, key="show_saa")
else:
    risk_free_rate = st.session_state.setdefault("risk_free_rate", 0.02)
    show_cml       = st.session_state.setdefault("show_cml", True)
    show_saa       = st.session_state.setdefault("show_saa", True)


if not IS_VIEWER:
    with st.sidebar.expander("Optimization"):
        optimization_mode = st.radio("Optimization Mode", ["Max Return", "Min Volatility"], key="optimization_mode")
        max_vol_limit    = st.number_input("Max Volatility (for Max Return)", value=0.20, step=0.005, format="%.3f", key="max_vol_limit")
        min_return_limit = st.number_input("Min Return (for Min Volatility)", value=0.04, step=0.005, format="%.3f", key="min_return_limit")
        st.checkbox("Use Optimized Weights in Simulation", key="use_optimized_weights")
else:
    optimization_mode = st.session_state.setdefault("optimization_mode", "Max Return")
    max_vol_limit     = st.session_state.setdefault("max_vol_limit", 0.20)
    min_return_limit  = st.session_state.setdefault("min_return_limit", 0.04)
    st.session_state.setdefault("use_optimized_weights", False)

use_optimized_weights = st.session_state.get("use_optimized_weights", False)


selected_scenario = None
scenarios_df = None

if not IS_VIEWER:
    with st.sidebar.expander("Historical Scenario Analysis"):
        scenario_file = st.file_uploader("Upload Historical Scenarios", type=["xlsx"])
        st.checkbox("Use Optimized Weights in Scenario", key="use_opt_in_scenario")
        use_opt_in_scenario = st.session_state.get("use_opt_in_scenario", False)

        if scenario_file is not None:
            file_suffix = Path(scenario_file.name).suffix.lower()
            try:
                if file_suffix == ".xlsx":
                    scenarios_df = pd.read_excel(scenario_file)
                else:
                    st.warning("Unsupported file type for scenario upload.")
            except Exception as e:
                st.error(f"Error reading scenario file: {e}")

        if scenarios_df is None:
            _default_scen = st.session_state.get("default_scenarios_df")
            if _default_scen is not None and not _default_scen.empty:
                st.caption("Using preloaded baseline scenarios")
                scenarios_df = _default_scen.copy()

        if scenarios_df is not None and not scenarios_df.empty:
            scenario_names = scenarios_df.iloc[:, 0].astype(str).tolist()
            selected_scenario = st.selectbox("Select Scenario", scenario_names)

else:
    st.session_state.setdefault("use_opt_in_scenario", False)


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

DEFAULT_LTCMA = pd.DataFrame({
    "Exp Return": [0.06, 0.03, 0.08],
    "Exp Volatility": [0.10, 0.05, 0.15],
    "SAA": [0.5, 0.3, 0.2],
    "Min": [0.0, 0.0, 0.0],
    "Max": [1.0, 1.0, 1.0]
}, index=["Equities", "Bonds", "Alternatives"])


if not IS_VIEWER:
    st.subheader("LTCMA Table")

    uploaded_ltcma = st.file_uploader(
        "Upload LTCMA File",
        type=["xls", "xlsx"],
        key="ltcma_upload",
        label_visibility="collapsed",
    )

    if uploaded_ltcma is not None:
        base_ltcma = pd.read_excel(uploaded_ltcma, index_col=0)
        st.session_state.pop("ltcma_widget", None)
    else:
        base_ltcma = st.session_state.get("ltcma_base_default", DEFAULT_LTCMA)

    for c in ["Exp Return", "Exp Volatility", "SAA", "Min", "Max"]:
        if c in base_ltcma.columns:
            base_ltcma[c] = pd.to_numeric(base_ltcma[c], errors="coerce").astype(float)

    ltcma_return = st.data_editor(
        base_ltcma,
        num_rows="dynamic",
        width="stretch",
        disabled=not cap["edit_ltcma"],
        key="ltcma_widget",
    )

    ltcma_df = ltcma_return.copy()

else:
    base_ltcma = st.session_state.get("ltcma_base_default")
    if base_ltcma is None:
        try:
            base_ltcma = pd.read_excel(DEFAULT_LTCMA_PATH, index_col=0)
        except Exception:
            base_ltcma = DEFAULT_LTCMA
    base_ltcma = base_ltcma.copy()
    for c in ["Exp Return", "Exp Volatility", "SAA", "Min", "Max"]:
        if c in base_ltcma.columns:
            base_ltcma[c] = pd.to_numeric(base_ltcma[c], errors="coerce").astype(float)
    ltcma_df = base_ltcma


ltcma_df = ltcma_df.dropna(how="all")
ltcma_df.index = ltcma_df.index.astype(str).str.strip()
ltcma_df = ltcma_df.loc[ltcma_df.index != ""]

_prev_ltcma = st.session_state.get("prev_ltcma_df")
if _prev_ltcma is None or not ltcma_df.equals(_prev_ltcma):
    st.session_state["prev_ltcma_df"] = ltcma_df.copy()
    st.session_state.pop("portfolio_paths", None)
    st.session_state.pop("x_axis", None)

new_assets = tuple(ltcma_df.index.tolist())
prev_assets = st.session_state.get("corr_assets")
if prev_assets is None or prev_assets != new_assets:
    st.session_state["corr_store"] = ensure_corr_shape(
        ltcma_df.index, st.session_state.get("corr_store")
    )
    st.session_state["corr_assets"] = new_assets
    st.session_state.pop("corr_widget", None)

if not IS_VIEWER:
    hcol, bcol = st.columns([6, 1])
    with hcol:
        st.subheader("Correlation Matrix")
    with bcol:
        symm_click = st.button(
            "↔ Symmetrize",
            key="symmetrize_btn",
            help="Set to (A + Aᵀ)/2, clip to [-1,1], diagonal=1",
        )

    uploaded_corr = st.file_uploader(
        "Upload Corr Matrix",
        type=["xls", "xlsx"],
        key="corr_upload",
        label_visibility="collapsed",
    )

    if uploaded_corr is not None:
        corr_store = pd.read_excel(uploaded_corr, index_col=0)
        st.session_state["corr_store"] = corr_store.copy()
        st.session_state.pop("corr_widget", None)

    corr_base = ensure_corr_shape(
        ltcma_df.index, st.session_state.get("corr_store")
    ).astype(float)

    float_config = {
        c: st.column_config.NumberColumn(
            label=c, min_value=-1.0, max_value=1.0, step=0.01, format="%.2f"
        )
        for c in corr_base.columns
    }

    corr_return = st.data_editor(
        corr_base,
        width="stretch",
        column_config=float_config,
        disabled=not cap["edit_ltcma"],
        key="corr_widget",
    )

    try:
        if not np.allclose(
            corr_return.values, corr_return.values.T, atol=1e-10, equal_nan=True
        ):
            st.info("Matrix isn’t symmetric. Click ↔ Symmetrize to fix.")
    except Exception:
        pass

    if symm_click:
        sym = (corr_return + corr_return.T) / 2.0
        sym = sym.clip(-1.0, 1.0)
        np.fill_diagonal(sym.values, 1.0)
        st.session_state["corr_store"] = sym.copy()
        st.session_state["prev_corr_df"] = sym.copy()
        st.session_state.pop("portfolio_paths", None)
        st.session_state.pop("x_axis", None)
        st.session_state.pop("corr_widget", None)
        st.rerun()

    corr_matrix = corr_return.copy()
else:
    corr_matrix = ensure_corr_shape(
        ltcma_df.index, st.session_state.get("corr_store")
    ).astype(float)

np.fill_diagonal(corr_matrix.values, 1.0)

_prev_corr = st.session_state.get("prev_corr_df")
if _prev_corr is None or not corr_matrix.equals(_prev_corr):
    st.session_state["prev_corr_df"] = corr_matrix.copy()
    st.session_state["corr_store"] = corr_matrix.copy()
    st.session_state.pop("portfolio_paths", None)
    st.session_state.pop("x_axis", None)

if IS_VIEWER and VIEW == "stats":
    st.subheader("Current portfolio statistics")
    try:
        mu = ltcma_df["Exp Return"].values
        vols = ltcma_df["Exp Volatility"].values
        w = ltcma_df["SAA"].values
        cov = np.outer(vols, vols) * corr_matrix.values
        exp_r = float(np.dot(w, mu))
        exp_v = float(np.sqrt(w @ cov @ w))

        rf = float(st.session_state.get("risk_free_rate", 0.03))  #v7.1.1 Sharpe Ratio
        sharpe = (exp_r - rf) / exp_v if exp_v > 0 else float("nan")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Expected Return", f"{exp_r:.2%}")
        with c2:
            st.metric("Expected Volatility", f"{exp_v:.2%}")
        with c3:
            st.metric("Sharpe (vs Rf)", f"{sharpe:.2f}", help=f"Risk-free: {rf:.2%}")

        df_pct = (ltcma_df[["SAA", "Exp Return", "Exp Volatility"]].copy() * 100)

        st.dataframe(
            df_pct,
            column_config={
                "SAA": st.column_config.NumberColumn("SAA", format="%.1f%%"),
                "Exp Return": st.column_config.NumberColumn("Exp Return", format="%.2f%%"),
                "Exp Volatility": st.column_config.NumberColumn("Exp Volatility", format="%.2f%%"),
            },
            width="stretch",
        )
  
    except Exception:
        pass


if not ltcma_df.empty and not corr_matrix.empty and ltcma_df.index.equals(corr_matrix.index):
    ltcma_df = ltcma_df.loc[ltcma_df.index.intersection(corr_matrix.index)]

    frequency_map = {"monthly": 12, "quarterly": 4, "yearly": 1}
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

    IS_VIEWER = (ROLE == "viewer")
    view = get_view_param("home")  # uses helpers you defined above

    if IS_VIEWER and view in ("sim", "ef", "stats"):
        viewer_back_link()

    if IS_VIEWER:
        if view in ("home", "", None):
            st.subheader("Executive Menu")
            r1c1, r1c2 = st.columns(2)

            with r1c1:
                image_tile("View current portfolio statistics", "Data/Portfolio Statistics.png", "stats", height_px=230)

            with r1c2:
                image_tile("Run simulation paths", "Data/Simulations.png", "sim", height_px=230)

            st.markdown("<div style='height: 18px'></div>", unsafe_allow_html=True)

            r2c1, r2c2 = st.columns(2)

            with r2c1:
                image_tile("See the efficient frontier",
                           "Data/Efficient Frontier.png",        
                           "ef", height_px=230)
            with r2c2:
                image_tile("Run stress test scenarios",
                           "Data/Stress Test Scenario.png",
                           "scen", height_px=230)

            run_sim = run_ef = run_opt = run_scenario = False
            show_stats = False
            st.stop()

        run_sim = (view == "sim")
        run_ef = (view == "ef")
        run_opt = False 
        run_scenario = (view == "scen")
        show_stats = (view == "stats")

        if IS_VIEWER and view == "scen":
            tl, lbl_col, tr = st.columns([1, 1, 3])
            with tl:
                viewer_back_link()

            with lbl_col:
                st.markdown(
                    "<div style='margin-top:10px; white-space:nowrap; font-weight:600;'>Pick a scenario →</div>",
                    unsafe_allow_html=True
                )
            with tr:
                scen_df = st.session_state.get("default_scenarios_df")
                if scen_df is None or scen_df.empty:
                    try:
                        scen_df = pd.read_excel(DEFAULT_BASELINE_SCENARIO_PATH)
                        st.session_state["default_scenarios_df"] = scen_df.copy()
                    except Exception:
                        scen_df = None

                if scen_df is not None and not scen_df.empty:
                    scenario_names = scen_df.iloc[:, 0].astype(str).tolist()
                    current = st.session_state.get("viewer_scenario_select")
                    if current not in scenario_names:
                        current = scenario_names[0]

                    sel = st.selectbox(
                        "Scenario",
                        scenario_names,
                        index=scenario_names.index(current),
                        key="viewer_scenario_select",
                        label_visibility="collapsed"
                    )

                    try:
                        st.query_params.update({"scen": sel})
                    except Exception:
                        qp = st.experimental_get_query_params()
                        qp.update({"scen": sel})
                        st.experimental_set_query_params(**qp)
                else:
                    st.warning("No scenarios available.")

    else:
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            run_sim = st.button("Run Simulation")
        with col2:
            run_ef = st.button("Run Efficient Frontier")
        with col3:
            run_opt = st.button("Run Optimization", disabled=not cap["optimize"])
        with col4:
            run_scenario = st.button("Run Scenario", disabled=not cap["scenarios"])
        show_stats = False


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


        pct0 = FuncFormatter(lambda v, pos: f"{v:.0%}")   # 0 decimals (e.g., 12%)
        pct1 = FuncFormatter(lambda v, pos: f"{v:.1%}")   # 1 decimal (e.g., 12.3%)

        ax2.xaxis.set_major_formatter(pct0)  
        ax2.yaxis.set_major_formatter(pct1)

        center_plot(fig2, ratio=(1, 4, 1), figsize=(10, 7))
        
        ef_buf = BytesIO()
        fig2.savefig(ef_buf, format="png")
        ef_buf.seek(0)
        st.download_button("Download Efficient Frontier Chart", data=ef_buf, file_name="efficient_frontier.png", mime="image/png")

  
    if run_sim:

        SIMULATION_SEED = 20250826
        sim_rng = np.random.default_rng(SIMULATION_SEED)

        weights_to_use = (
            st.session_state.get("optimized_weights").reindex(ltcma_df.index).fillna(0).values
            if use_optimized_weights and "optimized_weights" in st.session_state
            else ltcma_df["SAA"].values
        )

        mu = ltcma_df["Exp Return"].values
        vols = ltcma_df["Exp Volatility"].values

        corr = corr_matrix.loc[ltcma_df.index, ltcma_df.index].values
        cov = np.outer(vols, vols) * corr

        expected_portfolio_return = np.dot(weights_to_use, mu)
        expected_portfolio_volatility = np.sqrt(weights_to_use.T @ cov @ weights_to_use)

        steps_per_year = {"monthly": 12, "quarterly": 4, "yearly": 1}[frequency]
       
        n_steps = n_years * steps_per_year
        dt = 1 / steps_per_year

        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals[eigvals < 0] = 0.0
            cov = eigvecs @ np.diag(eigvals) @ eigvecs.T + 1e-12 * np.eye(cov.shape[0])
            chol = np.linalg.cholesky(cov)

        # Student-t scaling so shocks have unit variance pre-correlation
        scale = np.sqrt(fat_tail_df / (fat_tail_df - 2))  # df > 2







#        portfolio_paths = np.zeros((n_steps + 1, n_sims))
#        portfolio_paths[0] = initial_value
#
#        progress_bar = st.progress(0)
#        status_text = st.empty()
#    
#        for sim in range(n_sims):
#            prices = np.ones(len(weights_to_use))
#            path = [initial_value]
#            for _ in range(n_steps):
#                z = t.rvs(fat_tail_df, size=len(weights_to_use), random_state=sim_rng) / scale  # v4.0.2 : use a fixed seed for random generator
#                correlated_z = chol @ z
#                prices *= np.exp(mu * dt + correlated_z * np.sqrt(dt))
#                path.append(np.dot(prices, weights_to_use) * initial_value)
#            portfolio_paths[:, sim] = path
#            # Update progress
#            if sim % 100 == 0:
#                progress_bar.progress((sim + 1) / n_sims)
#                status_text.text(f"Simulation {sim + 1}/{n_sims}")



        portfolio_paths = np.zeros((n_steps + 1, n_sims))
        portfolio_paths[0] = initial_value

        progress_bar = st.progress(0)
        status_text = st.empty()

        prices = np.ones((len(weights_to_use), n_sims))  # Vectorisé

        # v8 Vectorized calculations

        for step in range(n_steps):
            z = t.rvs(fat_tail_df, size=(len(weights_to_use), n_sims), random_state=sim_rng) / scale
            correlated_z = chol @ z
            prices *= np.exp(mu[:, None] * dt + correlated_z * np.sqrt(dt))
            portfolio_paths[step + 1] = np.dot(weights_to_use, prices) * initial_value
            if step % 1000 == 0:
                progress_bar.progress((step + 1) / n_sims)
                status_text.text(f"Simulation {step + 1}/{n_sims}")

    







        progress_bar.empty()
        status_text.empty()

        st.session_state["portfolio_paths"] = portfolio_paths

        freq_map = {"monthly": "ME", "quarterly": "QE", "yearly": "YE"}
        st.session_state["x_axis"] = pd.date_range(start=start_date, periods=n_steps + 1, freq=freq_map[frequency])
        st.session_state.simulation_has_run = True

        if not IS_VIEWER:
            with st.expander("Weights used in simulation", expanded=False):
                wdf = pd.DataFrame({"Weight": weights_to_use}, index=ltcma_df.index)
                wdf_pct = (wdf * 100).rename(columns={"Weight": "Weight (%)"})
                st.dataframe(
                    wdf_pct,
                    column_config={"Weight (%)": st.column_config.NumberColumn(format="%.1f%%")},
                    width="stretch",
                )
       

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

            opt_df = pd.DataFrame({"Weight (%)": opt_w * 100}, index=ltcma_df.index)
            st.dataframe(
                opt_df,
                column_config={"Weight (%)": st.column_config.NumberColumn(format="%.1f%%")},
                width="stretch",
            )

            fig_opt, ax_opt = plt.subplots()
            ax_opt.scatter(opt_v, opt_r, c='green', label='Optimized Portfolio')
            ax_opt.set_xlabel('Volatility')
            ax_opt.set_ylabel('Return')
            ax_opt.set_title('Optimized Portfolio Result')
            ax_opt.grid(True)
            ax_opt.legend()


            pct0 = FuncFormatter(lambda v, pos: f"{v:.0%}")
            pct1 = FuncFormatter(lambda v, pos: f"{v:.1%}")
            ax_opt.xaxis.set_major_formatter(pct0)
            ax_opt.yaxis.set_major_formatter(pct1)
            center_plot(fig_opt, ratio=(1, 4, 1), figsize=(10, 7))

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

        use_opt_in_scenario = bool(st.session_state.get("use_opt_in_scenario", False))

        if IS_VIEWER:
            if scenarios_df is None:
                scenarios_df = st.session_state.get("default_scenarios_df")

            if scenarios_df is None or scenarios_df.empty:
                try:
                    scenarios_df = pd.read_excel(DEFAULT_BASELINE_SCENARIO_PATH)
                    st.session_state["default_scenarios_df"] = scenarios_df.copy()
                except Exception:
                    scenarios_df = None

            if selected_scenario is None and scenarios_df is not None and not scenarios_df.empty:
                scenario_names = scenarios_df.iloc[:, 0].astype(str).tolist()
                if scenario_names:
                    selected_scenario = st.session_state.setdefault("viewer_scenario_select", scenario_names[0])

        if IS_VIEWER:
            selected_scenario = st.session_state.get("viewer_scenario_select", selected_scenario)
        
        if scenarios_df is None:
            st.error("No scenarios available. Upload an Excel (.xlsx) in the sidebar (analyst/admin) or load the baseline.")
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
                scenario_row.index = scenario_row.index.astype(str).str.strip()
                ltcma_assets = ltcma_df.index
                scenario_aligned = scenario_row.reindex(ltcma_assets)
                missing_assets = scenario_aligned[scenario_aligned.isna()].index.tolist()
                scenario_aligned = pd.to_numeric(scenario_aligned, errors="coerce").fillna(0.0)  #v6.11
                if use_opt_in_scenario and "optimized_weights" in st.session_state:
                    weights_used = (
                        st.session_state["optimized_weights"]
                        .reindex(ltcma_assets)
                        .fillna(0.0)
                    )
                else:
                    weights_used = ltcma_df["SAA"]

                scenario_return = float((scenario_aligned * weights_used).sum())

                if missing_assets:
                    st.info(
                        "Assets missing in scenario (assumed 0% shock): "
                        + ", ".join(map(str, missing_assets))
                    )

                impact_df = pd.DataFrame(
                    {
                        "Shock Return": scenario_aligned.values,
                        "Weight": weights_used.values,
                    },
                    index=ltcma_assets,
                )
                impact_df.index.name = "Asset Class"
                impact_df["Contribution"] = impact_df["Shock Return"] * impact_df["Weight"]

                disp = pd.DataFrame(
                    {
                        "Weight (%)":        impact_df["Weight"] * 100,
                        "Shock Return (%)":  impact_df["Shock Return"] * 100,
                        "Contribution (%)":  impact_df["Contribution"] * 100,
                    },
                    index=impact_df.index,
                )

                st.dataframe(
                    disp,
                    column_config={
                        "Weight (%)":       st.column_config.NumberColumn(format="%.1f%%"),
                        "Shock Return (%)": st.column_config.NumberColumn(format="%.2f%%"),
                        "Contribution (%)": st.column_config.NumberColumn(format="%.2f%%"),
                    },
                    width="stretch",
                )

                # Waterfall chart of contributions (Plotly)
                labels   = list(impact_df.index) + ["Total"]
                measures = ["relative"] * len(impact_df) + ["total"]
                values   = list(impact_df["Contribution"].values) + [scenario_return]

                fig = go.Figure(go.Waterfall(
                    orientation="v",
                    measure=measures,
                    x=labels,
                    y=values,
                    text=[f"{v:.2%}" for v in impact_df["Contribution"]] + [f"{scenario_return:.2%}"],
                    textposition="outside",
                    connector={"line": {"width": 1}},
                ))

                fig.update_layout(
                    title=f"Scenario Contribution Waterfall — {selected_scenario}",
                    yaxis=dict(title="Contribution", tickformat=".0%"),
                    xaxis=dict(title="Asset Class"),
                    showlegend=False,
                    height=420,
                    margin=dict(l=40, r=40, t=36, b=20),
                )

                # v7.1  +20% padding around the numeric axis
                mn = min(0, min(values))
                mx = max(0, max(values))
                span = mx - mn if mx != mn else max(abs(mx), 1.0)
                pad = 0.2 * span
                fig.update_yaxes(range=[mn - pad, mx + pad])

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displaylogo": False,
                        "toImageButtonOptions": {"format": "png", "filename": "scenario_waterfall", "scale": 2},
                    },
                )
                st.markdown(f"**Portfolio Return under Scenario '{selected_scenario}':** {scenario_return:.2%}")



SHOW_TABS = (ROLE != "viewer") or (VIEW == "sim")

if SHOW_TABS:

    if IS_VIEWER:
        t_sim, t_sum, t_dd = st.tabs(
            ["Simulation Chart", "Summary Statistics", "Drawdown Analysis"]
        )
    else:
        t_sim, t_sum, t_var, t_dd = st.tabs(
            ["Simulation Chart", "Summary Statistics", "Value at Risk", "Drawdown Analysis"]
        )

    has_data = (
        st.session_state.get("simulation_has_run")
        and isinstance(st.session_state.get("portfolio_paths"), np.ndarray)
        and st.session_state.get("x_axis") is not None
    )

    with t_sim:
        if not has_data:
            st.info("Run a simulation to see the chart.")
        else:
            portfolio_paths = st.session_state["portfolio_paths"]
            x_axis = st.session_state["x_axis"]

            p25 = np.percentile(portfolio_paths, 25, axis=1)
            p75 = np.percentile(portfolio_paths, 75, axis=1)
            p05 = np.percentile(portfolio_paths, 5, axis=1)
            p95 = np.percentile(portfolio_paths, 95, axis=1)
            p50 = np.median(portfolio_paths, axis=1)

            periods_per_year = {"monthly": 12, "quarterly": 4, "yearly": 1}[frequency]
            n_steps = len(x_axis)
            years_elapsed = np.arange(n_steps) / periods_per_year
            two_x_line = np.full(n_steps, 2.0 * initial_value, dtype=float)
            compounded_double = 2.0 * initial_value * (1.0 + inflation_rate) ** years_elapsed


            # ==== Build actual-on-grid series (file-scale consistent + clip) ====
            actual_x, actual_y = None, None
            if st.session_state.get("actual_series_base1") is not None and st.session_state.get("actual_show", False):
                s = st.session_state["actual_series_base1"].copy()  # base=1.0 at file's first row
                s = s.sort_index()
                s = s[~s.index.duplicated(keep="last")].astype(float)

                # Resample to the sim frequency (month/quarter/year end) using your helper
                s_freq = _resample_to_sim_freq(s, frequency)

                # Place on the simulation grid; don't fill outside the actual data span
                on_grid = s_freq.reindex(st.session_state["x_axis"])

                first_idx = on_grid.first_valid_index()
                last_idx  = on_grid.last_valid_index()
                if first_idx is not None and last_idx is not None and first_idx <= last_idx:
                    # Keep only the segment where we actually have data; fill inside that window
                    seg = on_grid.loc[first_idx:last_idx].ffill()

                    # Constant rebase:
                    # - If "scale to initial" ON → multiply base1 by chart Initial Value
                    # - Otherwise → multiply base1 by the file's first value (i.e., recover raw levels)
                    scale = (
                        float(st.session_state.get("initial_value", 100.0))
                        if st.session_state.get("actual_scale_to_initial", True)
                        else float(st.session_state.get("actual_first_value", 1.0))
                    )
                    if not np.isfinite(scale) or scale <= 0:
                        scale = 1.0

                    scaled_seg = seg * scale  # NO pin-to-first-grid rebase

                    mask = scaled_seg.notna().values
                    if mask.any():
                        actual_x = scaled_seg.index[mask]
                        actual_y = scaled_seg.values[mask]


            if chart_engine.startswith("Plotly"):
                def _positives(a):
                    a = np.asarray(a, float)
                    a[~np.isfinite(a)] = np.nan
                    a[a <= 0] = np.nan
                    return a

                if use_log_scale:
                    p05s, p25s, p50s, p75s, p95s = map(_positives, (p05, p25, p50, p75, p95))
                    tgt_2x = _positives(two_x_line) if show_double_initial else None
                    tgt_inf = _positives(compounded_double) if show_double_with_inflation else None
                else:
                    p05s, p25s, p50s, p75s, p95s = p05, p25, p50, p75, p95
                    tgt_2x = two_x_line if show_double_initial else None
                    tgt_inf = compounded_double if show_double_with_inflation else None

                fig = go.Figure()

                fig.add_trace(go.Scatter(x=x_axis, y=p95s, name="95th percentile",
                                         mode="lines", line=dict(width=1),
                                         hovertemplate="95th: %{y:,.0f}<extra></extra>"))
                fig.add_trace(go.Scatter(x=x_axis, y=p05s, name="5th percentile",
                                         mode="lines", line=dict(width=1),
                                         fill="tonexty", fillcolor="rgba(100,149,237,0.15)",
                                         hovertemplate="5th: %{y:,.0f}<extra></extra>"))
                fig.add_trace(go.Scatter(x=x_axis, y=p75s, name="75th percentile",
                                         mode="lines", line=dict(width=1),
                                         hovertemplate="75th: %{y:,.0f}<extra></extra>"))
                fig.add_trace(go.Scatter(x=x_axis, y=p25s, name="25th percentile",
                                         mode="lines", line=dict(width=1),
                                         fill="tonexty", fillcolor="rgba(100,149,237,0.30)",
                                         hovertemplate="25th: %{y:,.0f}<extra></extra>"))
                fig.add_trace(go.Scatter(x=x_axis, y=p50s, name="Median (50th)",
                                         mode="lines", line=dict(width=2, color="black"),
                                         hovertemplate="Median: %{y:,.0f}<extra></extra>"))
                if tgt_2x is not None:
                    fig.add_trace(go.Scatter(x=x_axis, y=tgt_2x, name="2x Initial Value",
                                             mode="lines", line=dict(dash="dash"),
                                             hovertemplate="%{y:,.0f}<extra></extra>"))
                if tgt_inf is not None:
                    fig.add_trace(go.Scatter(x=x_axis, y=tgt_inf, name="Inflation-Adj. 2x Target",
                                             mode="lines", line=dict(dash="dot"),
                                             hovertemplate="%{y:,.0f}<extra></extra>"))

                fig.update_layout(
                    height=chart_height,
                    title=dict(text="Portfolio Value Over Time", x=0.5),
                    margin=dict(l=40, r=40, t=36, b=20),
                    legend=dict(
                        orientation="h",
                        x=0.5, xanchor="center",
                        y=-0.12, yanchor="top", 
                        bgcolor="rgba(255,255,255,0.6)",
                        bordercolor="rgba(0,0,0,0.1)",
                        borderwidth=1
                    ),
                    hovermode="x"
                )

                fig.update_xaxes(title="Date", showspikes=True, spikemode="across", spikesnap="cursor")
                if use_log_scale:
                    series = [p05s, p25s, p50s, p75s, p95s]
                    if tgt_2x is not None: series.append(tgt_2x)
                    if tgt_inf is not None: series.append(tgt_inf)
                    flat = np.concatenate([s[np.isfinite(s)] for s in series if s is not None])
                    ymin = max(1e-6, float(np.nanmin(flat)))
                    ymax = float(np.nanmax(flat))
                    fig.update_yaxes(type="log", title="Portfolio Value (log)", range=[np.log10(ymin*0.95), np.log10(ymax*1.05)])
                else:
                    fig.update_yaxes(title="Portfolio Value")

                x_last = x_axis[-1]
                def _annot(y):
                    if y is not None and np.isfinite(y[-1]):
                        fig.add_annotation(x=x_last, y=y[-1], text=f"{y[-1]:,.0f}",
                                           showarrow=False, xanchor="left", yanchor="middle", xshift=8, font=dict(size=10))
                for arr in (p95s, p75s, p50s, p25s, p05s):
                    _annot(arr)

                if actual_x is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=actual_x, y=actual_y,
                            name="Actual portfolio",
                            mode="lines+markers",
                            line=dict(width=2),
                            hovertemplate="Actual: %{y:,.0f}<extra></extra>"
                        )
                    )

                # Lock x-axis to the simulation range (optionally keep a tiny pad on the right for labels)
                x0 = st.session_state["x_axis"][0]
                x1 = st.session_state["x_axis"][-1]
                fig.update_xaxes(range=[x0, x1])  # no left drift before sim start

                


                st.plotly_chart(
                    fig, use_container_width=True,
                    config={
                        "displaylogo": False,
                        "toImageButtonOptions": {"format": "png", "filename": "simulation_plot", "scale": 2}
                    }
                )

            else:
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

                    periods_per_year = {"monthly": 12, "quarterly": 4, "yearly": 1}[frequency]
                    n_steps = len(x_axis)
                    two_x_line = np.full(n_steps, 2 * initial_value, dtype=float)
                    growth_factor = (1 + inflation_rate / periods_per_year) ** np.arange(n_steps)
                    compounded_double = initial_value * 2 * growth_factor

                    fig, ax = plt.subplots(figsize=(10, 6))

                    if use_log_scale:
                        ax.set_yscale("log")

                    ax.plot(x_axis, median_path, color="blue", label="Median")
                    ax.fill_between(x_axis, p05, p95, color="lightblue", alpha=0.3, label="5–95%")
                    ax.fill_between(x_axis, p25, p75, color="blue", alpha=0.2, label="25–75%")

                    if show_double_initial:
                        ax.plot(x_axis, two_x_line, color="red", linestyle="--", linewidth=1, label="2x Initial Value")

                    if show_double_with_inflation:
                        ax.plot(x_axis, compounded_double, color="darkorange", linestyle="--", linewidth=1,
                                label="Inflation-Adj. 2x Target")

                    label_fontsize = 8
                    x_last = x_axis[-1] + pd.Timedelta(days=15)
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
                    ax.set_ylabel("Portfolio Value (log scale)" if use_log_scale else "Portfolio Value")
                    ax.grid(True)
                    ax.legend()
                    plt.xticks(rotation=45)

                    if use_log_scale:
                        y_min = max(1e-6, min(p05.min(), p25.min(), median_path.min()))
                        ax.set_ylim(bottom=y_min * 0.9)

                    if actual_x is not None:
                        ax.plot(actual_x, actual_y, label="Actual portfolio", linewidth=2)
                        ax.legend()

                    ax.set_xlim(st.session_state["x_axis"][0], st.session_state["x_axis"][-1])
                    ax.margins(x=0)  # no extra padding on the left


                    st.pyplot(fig)

            percentile_df = pd.DataFrame({
                "5th Percentile": p05,
                "25th Percentile": p25,
                "Median": p50,
                "75th Percentile": p75,
                "95th Percentile": p95
            }, index=x_axis)
            percentile_df.index.name = "Date"
            if show_double_initial:
                percentile_df["2x Initial Value"] = 2 * initial_value
            if show_double_with_inflation:
                percentile_df["Inflation-Adj. 2x Target"] = compounded_double

            paths_df = pd.DataFrame(
                portfolio_paths, index=x_axis,
                columns=[f"Sim_{i+1}" for i in range(portfolio_paths.shape[1])]
            )
            paths_df.index.name = "Date"


            # --- Admin-only: FULL simulation paths export ---
            if ROLE == "admin":
                # (optional) checkbox so we only build on demand
                if st.checkbox("Prepare FULL Simulation Paths (Excel) — large file", value=False, key="prep_paths_xlsx"):
                    @st.cache_data
                    def build_paths_xlsx(paths: np.ndarray, x_axis: pd.DatetimeIndex) -> bytes:
                        df = pd.DataFrame(paths, index=x_axis, columns=[f"Sim_{i+1}" for i in range(paths.shape[1])])
                        df.index.name = "Date"
                        bio = BytesIO()
                        with pd.ExcelWriter(bio, engine="xlsxwriter") as w:
                            df.to_excel(w, sheet_name="Simulation Paths", index=True)
                        return bio.getvalue()

                    paths_bytes = build_paths_xlsx(portfolio_paths, x_axis)
                    st.download_button(
                        "Download Simulation Paths (Excel)",
                        data=paths_bytes,
                        file_name="simulated_paths.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_paths_xlsx",
                    )



    with t_sum:
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
                st.markdown(f"**Final Value (Mean):** ${mean_final:,.0f}")
                st.markdown(f"**5th Percentile :** ${pctiles[0]:,.0f}")
                st.markdown(f"**25th Percentile :** ${pctiles[1]:,.0f}")
                st.markdown(f"**Final Value (Median):** ${median_final:,.0f}")
                st.markdown(f"**75th Percentile :** ${pctiles[2]:,.0f}")
                st.markdown(f"**95th Percentile :** ${pctiles[3]:,.0f}")
                
                
            with right_col_final:

                fig_hist, ax_hist = plt.subplots(figsize=(5, 4))
                final_values = portfolio_paths[-1]

                x_min, x_max = np.percentile(final_values, [1, 99])
                ax_hist.hist(final_values, bins=50, range=(0, x_max*1.5), color='skyblue', edgecolor='black')
                ax_hist.set_xlim(0, x_max*1.5)
            
                ax_hist.set_title("Distribution of Final Portfolio Values")
                ax_hist.set_xlabel("Final Value")
                ax_hist.set_ylabel("Frequency")

                hist_buf = BytesIO()
                fig_hist.savefig(hist_buf, format="png")
                hist_buf.seek(0)

                st.session_state["fig_hist"] = fig_hist
                st.session_state["buf_hist"] = hist_buf

                st.pyplot(st.session_state["fig_hist"])

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
 
    if not IS_VIEWER:
        with t_var:
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
                            f"the portfolio value will not fall below ${var_threshold:,.1f}.")
                        st.markdown(f"This represents a potential shortfall of ${var_loss:,.1f} ({var_pct_loss:.1%}) from the initial value.")
                        st.markdown(
                            f"**{var_confidence}% Conditional VaR (CVaR):**\n"
                            f"Expected average shortfall beyond VaR is **${cvar_loss:,.1f} ({cvar_pct_loss:.1%})**."
                        )
                        st.markdown(f"**Average VaR:** {np.mean(var_values):.1f}")
                        st.markdown(f"**Worst VaR:** {np.min(var_values):.1f}")
                        st.markdown(f"**5th Percentile VaR:** {np.percentile(var_values, 5):.1f}")
                        st.markdown(f"**25th Percentile VaR:** {np.percentile(var_values, 25):.1f}")
                        st.markdown(f"**Median VaR:** {np.percentile(var_values, 50):.1f}")
                        st.markdown(f"**75th Percentile VaR:** {np.percentile(var_values, 75):.1f}")
                        st.markdown(f"**95th Percentile VaR:** {np.percentile(var_values, 95):.1f}")

                    with right_col_var:
                
                        var_threshold = np.percentile(var_values, 100 - var_confidence)
                        fig_var, ax_var = plt.subplots(figsize=(5, 4))

                        x_min, x_max = np.percentile(var_values, [1, 99])
                        ax_var.hist(var_values, bins=50, range=(x_min/2, x_max*1.5), color='lightgrey', edgecolor='black')
                        ax_var.set_xlim(x_min/2, x_max*1.5)

                        ax_var.axvline(var_threshold, color='red', linestyle='--', label=f'{var_confidence}% VaR = {var_threshold:,.1f}')
                        ax_var.set_title(f"{var_confidence}% VaR at Year {var_years}")
                        ax_var.set_xlabel("Portfolio Value")
                        ax_var.set_ylabel("Frequency")
                        ax_var.legend()

                        var_buf = BytesIO()
                        fig_var.savefig(var_buf, format="png")
                        var_buf.seek(0)
                        st.session_state["fig_var"] = fig_var
                        st.session_state["buf_var"] = var_buf

                        st.pyplot(fig_var)

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

    with t_dd:
        if not has_data:
            st.info("Run a simulation to see drawdown analysis.")
        else:
            portfolio_paths = st.session_state["portfolio_paths"]
            x_axis = st.session_state["x_axis"]
            st.subheader("Drawdown Statistics")

#            max_drawdowns = []
#            recovery_times = []

            max_drawdowns, recovery_times = compute_drawdown_stats_vectorized(
                tuple(map(tuple, portfolio_paths))
            )


        
#            for sim in range(portfolio_paths.shape[1]):
#                path = portfolio_paths[:, sim]
#
#                peak_val = path[0]
#                peak_idx = 0
#                max_dd = 0.0
#                rec_time_for_max = np.nan
#
#                for i in range(1, len(path)):
#                    if path[i] > peak_val:
#                        peak_val = path[i]
#                        peak_idx = i
#
#                    dd = 1.0 - path[i] / peak_val
#
#                    if dd > max_dd:
#                        max_dd = dd
#                        rec_time_for_max = np.nan
#
#                        for j in range(i + 1, len(path)):
#                            if path[j] >= peak_val:
#                                rec_time_for_max = j - peak_idx
#                                break
#
#                max_drawdowns.append(max_dd)
#                recovery_times.append(rec_time_for_max)








            left_col_dd, right_col_dd = st.columns([1, 2])

            with left_col_dd:
                st.markdown(f"**Average Max Drawdown:** {np.mean(max_drawdowns):.1%}")
                st.markdown(f"**Worst Max Drawdown:** {np.max(max_drawdowns):.1%}")
                st.markdown(f"**5th Percentile Max Drawdown:** {np.percentile(max_drawdowns, 5):.1%}")
                st.markdown(f"**25th Percentile Max Drawdown:** {np.percentile(max_drawdowns, 25):.1%}")
                st.markdown(f"**Median Max Drawdown:** {np.percentile(max_drawdowns, 50):.1%}")
                st.markdown(f"**75th Percentile Max Drawdown:** {np.percentile(max_drawdowns, 75):.1%}")
                st.markdown(f"**95th Percentile Max Drawdown:** {np.percentile(max_drawdowns, 95):.1%}")

                rec_array = np.array(recovery_times, dtype=float)
                recovered_mask = ~np.isnan(rec_array)
                recovered_share = 100.0 * recovered_mask.mean()

                freq_label = {"monthly": "months", "quarterly": "quarters", "yearly": "years"}[frequency]
                if recovered_mask.any():
                    st.markdown(f"**Average Recovery Time ({freq_label}, recovered only):** {np.nanmean(rec_array):.1f}")
                else:
                    st.markdown(f"**Average Recovery Time ({freq_label}, recovered only):** n/a")

                st.markdown(f"**Recovered Paths:** {recovered_share:.1f}% "
                            f"({recovered_mask.sum()} of {len(rec_array)})")

            with right_col_dd:
                fig_ddist, ax_ddist = plt.subplots(figsize=(6, 4))
                ax_ddist.hist(max_drawdowns, bins=40, color='salmon', edgecolor='black')
                ax_ddist.set_title("Distribution of Maximum Drawdowns")
                ax_ddist.set_xlabel("Max Drawdown")
                ax_ddist.set_ylabel("Frequency")
                ax_ddist.xaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
                ax_ddist.grid(True)

                ddist_buf = BytesIO()
                fig_ddist.savefig(ddist_buf, format="png")
                ddist_buf.seek(0)
                st.session_state["fig_ddist"] = fig_ddist
                st.session_state["buf_ddist"] = ddist_buf

                st.pyplot(st.session_state["fig_ddist"])

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
            


