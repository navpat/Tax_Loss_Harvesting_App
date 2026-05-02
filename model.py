"""
model.py
--------
Pure simulation logic — no UI, no plotting.
All functions are stateless and independently testable.
"""

import numpy as np


# ─────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────

def phi(x):
    """Standard normal PDF."""
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


def expected_harvest_rate(rm, sigma_idio):
    """
    Expected CNCL harvested per unit of gross notional per unit time.

    E[loss per unit gross notional] = sigma_idio * phi(rm / sigma_idio)

    Derivation: stock return r_i = rm + eps_i, eps_i ~ N(0, sigma_idio^2)
    Stock is harvestable when r_i < 0. Expected loss per stock integrates
    over the truncated normal, simplifying to this closed form.
    """
    z = rm / sigma_idio
    return sigma_idio * phi(z)


# ─────────────────────────────────────────────
# Single path simulation
# ─────────────────────────────────────────────

def simulate_path(
    W0, B0, K0, L,
    sigma_idio, rm, alpha, stock_vol,
    t_LT, fin_cost, t_cost_bps,
    years, rng
):
    """
    Simulate one monthly path of the concentrated position + overlay.

    Parameters
    ----------
    W0          : float  — concentrated stock value ($M)
    B0          : float  — cost basis ($M)
    K0          : float  — overlay capital ($M)
    L           : float  — leverage (0.5 = 150/50, 1.0 = 200/100, 1.5 = 250/150)
    sigma_idio  : float  — idiosyncratic vol of stocks in overlay universe
    rm          : float  — annual market return
    alpha       : float  — annual factor alpha
    stock_vol   : float  — annual vol of the concentrated stock
    t_LT        : float  — long-term capital gains tax rate
    fin_cost    : float  — financing cost per unit of L per year
    t_cost_bps  : int    — transaction cost in bps per dollar traded
    years       : int    — simulation horizon
    rng         : np.random.Generator

    Returns
    -------
    dict with keys:
        tax_no      : np.array (years+1,) — tax bill without overlay
        tax_with    : np.array (years+1,) — tax bill with overlay
        cncl_pct    : np.array (years+1,) — CNCL as % of K0
        net_benefit : np.array (years+1,) — tax saved minus cumulative costs
        cum_costs   : np.array (years+1,) — cumulative financing + t-costs
    """
    dt = 1 / 12
    steps = years * 12

    annual_turnover  = 5 + L * 4            # rough calibration from paper
    monthly_fin      = fin_cost * L / 12
    monthly_tcost    = (t_cost_bps / 10000) * annual_turnover / 12

    Wt         = W0
    Kt         = K0
    cncl       = 0.0
    cum_costs  = 0.0

    harvest_rate = expected_harvest_rate(rm, sigma_idio)
    overlay_vol  = sigma_idio * np.sqrt(1 + 2 * L)

    record_at = set(range(0, steps + 1, 12))

    out_tax_no   = []
    out_tax_with = []
    out_cncl_pct = []
    out_net_ben  = []
    out_costs    = []

    for step in range(steps + 1):

        if step in record_at:
            embedded_gain    = max(Wt - B0, 0.0)
            tax_no           = t_LT * embedded_gain
            cncl_offset      = min(cncl, embedded_gain)
            tax_with         = t_LT * max(embedded_gain - cncl_offset, 0.0)
            tax_saving       = tax_no - tax_with

            out_tax_no.append(tax_no)
            out_tax_with.append(tax_with)
            out_cncl_pct.append(100.0 * cncl / K0)
            out_net_ben.append(tax_saving - cum_costs)
            out_costs.append(cum_costs)

        if step == steps:
            break

        # concentrated stock — geometric Brownian motion
        Wt *= np.exp(
            (rm - 0.5 * stock_vol**2) * dt
            + stock_vol * np.sqrt(dt) * rng.normal()
        )

        # overlay capital — grows at (rm + alpha - costs)
        net_ret = (rm + alpha
                   - monthly_fin * 12
                   - monthly_tcost * 12) * dt
        Kt *= np.exp(
            (net_ret - 0.5 * overlay_vol**2 * dt)
            + overlay_vol * np.sqrt(dt) * rng.normal()
        )

        # CNCL harvested this month
        gross_notional = (1 + 2 * L) * Kt
        delta_cncl     = harvest_rate * gross_notional * dt
        # vintage dispersion: market environment sensitivity
        shock          = max(rng.normal(1.0, 0.3), 0.0)
        cncl          += delta_cncl * shock

        # costs this month
        cum_costs += (monthly_fin + monthly_tcost) * Kt

    return {
        "tax_no":      np.array(out_tax_no),
        "tax_with":    np.array(out_tax_with),
        "cncl_pct":    np.array(out_cncl_pct),
        "net_benefit": np.array(out_net_ben),
        "cum_costs":   np.array(out_costs),
    }


# ─────────────────────────────────────────────
# Monte Carlo runner
# ─────────────────────────────────────────────

LEVERAGE_LEVELS = [0.0, 0.5, 1.0, 1.5]
LEVERAGE_LABELS = {
    0.0: "No overlay",
    0.5: "L=0.5  (150/50)",
    1.0: "L=1.0  (200/100)",
    1.5: "L=1.5  (250/150)",
}


def run_monte_carlo(params, n_sims=200, seed=42):
    """
    Run Monte Carlo simulation for all leverage levels.

    Parameters
    ----------
    params : dict — all model parameters (passed directly to simulate_path)
    n_sims : int  — number of paths per leverage level
    seed   : int  — random seed

    Returns
    -------
    t_axis  : np.array (years+1,)
    results : dict[float -> dict[str -> np.array(n_sims, years+1)]]
              keyed by leverage level, then by output variable
    """
    rng    = np.random.default_rng(seed)
    years  = params["years"]
    t_axis = np.arange(0, years + 1)
    results = {}

    for L in LEVERAGE_LEVELS:
        paths = {k: [] for k in
                 ["tax_no", "tax_with", "cncl_pct", "net_benefit", "cum_costs"]}

        for _ in range(n_sims):
            path = simulate_path(L=L, rng=rng, **params)
            for k in paths:
                paths[k].append(path[k])

        results[L] = {k: np.array(v) for k, v in paths.items()}

    return t_axis, results


# ─────────────────────────────────────────────
# Summary statistics (used by both app and tests)
# ─────────────────────────────────────────────

def summarize(results, t_axis, year=10):
    """
    Return a list of summary dicts for a given year across leverage levels.
    Useful for tables and metric cards.
    """
    yr_idx = int(year)   # index == year since t_axis is 0,1,...,years
    rows = []
    for L in LEVERAGE_LEVELS:
        r = results[L]
        tn  = np.median(r["tax_no"][:, yr_idx])
        tw  = np.median(r["tax_with"][:, yr_idx])
        nb  = np.median(r["net_benefit"][:, yr_idx])
        cncl= np.median(r["cncl_pct"][:, yr_idx])
        rows.append({
            "leverage":      L,
            "label":         LEVERAGE_LABELS[L],
            "year":          year,
            "cncl_pct":      round(cncl, 1),
            "tax_no":        round(tn, 2),
            "tax_with":      round(tw, 2),
            "tax_saved":     round(tn - tw, 2),
            "net_benefit":   round(nb, 2),
        })
    return rows