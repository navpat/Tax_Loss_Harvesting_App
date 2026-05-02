"""
app.py
------
Streamlit UI only. All simulation logic is in model.py, all plotting in charts.py.

Run with:
    streamlit run app.py

Install dependencies:
    pip install streamlit numpy matplotlib pandas
"""

import pandas as pd
import streamlit as st

from model import run_monte_carlo, summarize, LEVERAGE_LABELS
from charts import fig_tax_and_cncl, fig_net_benefit, fig_leverage_sensitivity


# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Tax Overlay Simulator",
    page_icon="📉",
    layout="wide"
)

st.title("Long-Short Overlay: Tax Benefit on Concentrated Position")
st.caption(
    "Models how a factor-based long-short overlay harvests losses to offset "
    "the embedded capital gain in a concentrated stock position. "
    "Based on Liberman, Krasner, Sosner & Freitas (AQR, 2023)."
)
st.divider()


# ─────────────────────────────────────────────
# Sidebar — inputs
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("Parameters")

    st.subheader("Concentrated position")
    W0        = st.slider("Position value ($M)",          1.0, 30.0,  5.0, 0.5)
    B0        = st.slider("Cost basis ($M)",              0.0,  W0,   0.5, 0.1)
    stock_vol = st.slider("Stock annual vol (%)",          10,   70,   30,   1) / 100

    st.subheader("Overlay")
    K0         = st.slider("Overlay capital ($M)",        0.5, 15.0,  3.0, 0.5)
    sigma_idio = st.slider("Idiosyncratic vol (%)",        10,   50,   25,   1) / 100

    st.subheader("Market assumptions")
    rm    = st.slider("Market return (%/yr)",             -5,   20,    8,   1) / 100
    alpha = st.slider("Factor alpha (%/yr)",             0.0,  5.0,  1.5, 0.25) / 100

    st.subheader("Costs & tax")
    t_LT       = st.slider("LT capital gains tax rate (%)", 15.0, 35.0, 23.8, 0.1) / 100
    fin_cost   = st.slider("Financing cost (%/yr per L)",    0.5,  2.0,  1.0, 0.1) / 100
    t_cost_bps = st.slider("Transaction cost (bps)",           5,   20,   10,   1)

    st.subheader("Simulation")
    n_sims = st.select_slider("Monte Carlo paths", [100, 200, 500, 1000], 200)
    years  = st.slider("Horizon (years)", 5, 15, 10, 1)


# ─────────────────────────────────────────────
# Pack params — single dict passed to model
# ─────────────────────────────────────────────

params = dict(
    W0=W0, B0=B0, K0=K0,
    sigma_idio=sigma_idio, rm=rm, alpha=alpha,
    stock_vol=stock_vol, t_LT=t_LT,
    fin_cost=fin_cost, t_cost_bps=t_cost_bps,
    years=years,
)


# ─────────────────────────────────────────────
# Run simulation (cached on params + n_sims)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_run(params_frozen, n_sims):
    return run_monte_carlo(dict(params_frozen), n_sims=n_sims, seed=42)


with st.spinner("Running simulation..."):
    t_axis, results = cached_run(tuple(sorted(params.items())), n_sims)


# ─────────────────────────────────────────────
# Summary metric cards
# ─────────────────────────────────────────────

st.subheader(f"Year-{years} summary")

summary = summarize(results, t_axis, year=years)
cols    = st.columns(4)

for col, row in zip(cols, summary):
    col.metric(
        label=row["label"],
        value=f"${row['net_benefit']:.2f}M net benefit",
        delta=f"${row['tax_saved']:.2f}M saved  |  CNCL {row['cncl_pct']:.0f}%",
    )

st.divider()


# ─────────────────────────────────────────────
# Tabs — charts
# ─────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Tax liability & CNCL", "Net benefit & costs", "Leverage sensitivity"])

with tab1:
    st.pyplot(fig_tax_and_cncl(results, t_axis))

with tab2:
    st.pyplot(fig_net_benefit(results, t_axis))

with tab3:
    st.pyplot(fig_leverage_sensitivity(results, years))


# ─────────────────────────────────────────────
# Results table
# ─────────────────────────────────────────────

st.divider()
st.subheader("Detailed results table")

rows = []
for yr in [1, 3, 5, 7, years]:
    rows.extend(summarize(results, t_axis, year=yr))

df = pd.DataFrame(rows).rename(columns={
    "label":       "Strategy",
    "year":        "Year",
    "cncl_pct":    "CNCL (%)",
    "tax_no":      "Tax — no overlay ($M)",
    "tax_with":    "Tax — with overlay ($M)",
    "tax_saved":   "Tax saved ($M)",
    "net_benefit": "Net benefit ($M)",
})

st.dataframe(
    df.drop(columns=["leverage"]),
    use_container_width=True,
    hide_index=True,
)


# ─────────────────────────────────────────────
# Theory footnote
# ─────────────────────────────────────────────

st.divider()
with st.expander("Model details"):
    st.markdown(r"""
**CNCL harvest rate per period:**

$$\mathbb{E}[\Delta \text{CNCL}_t] \approx (1+2L) \cdot K_t \cdot \sigma_{\text{idio}} \cdot \phi\!\left(\frac{r_m}{\sigma_{\text{idio}}}\right)$$

where $\phi$ is the standard normal PDF. Higher leverage $L$ scales gross notional linearly;
higher idiosyncratic vol means more stocks are underwater even in rising markets.

**Tax bill with overlay:**

$$\text{Tax}(t) = t_{LT} \cdot \max\!\bigl(W_t - B_0 - \text{CNCL}_t \cdot K_0,\; 0\bigr)$$

**Net benefit:**

$$\text{Net benefit}(t) = \underbrace{t_{LT} \cdot \text{CNCL}_t \cdot K_0}_{\text{tax saved}} - \underbrace{(c_{\text{fin}} + c_{\text{trans}}) \cdot t}_{\text{cumulative costs}}$$

Path noise on CNCL ($\epsilon \sim \mathcal{N}(1, 0.09)$) replicates vintage dispersion
from Exhibit 1 of the paper (wide 10th–90th percentile spread).
    """)