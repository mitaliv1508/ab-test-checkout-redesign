"""
Generate simulated data for an e-commerce checkout redesign A/B test.

Scenario
--------
An online retailer tests a new one-page checkout (treatment) against the
existing three-step checkout (control). Visitors who reach the checkout
page during the 14-day experiment window are randomly assigned 50/50.

Ground truth built into the simulation (what the analysis should recover):
- True conversion lift: +1.0 pp on mobile, +0.4 pp on desktop (~ +0.75 pp blended)
- No true effect on average order value (AOV)
- No true effect on refund rate (guardrail)
- Faster checkout completion time in treatment
- Effect is stable over time (no novelty effect)

Because the ground truth is known, the project doubles as a validation of
the statistical pipeline: does the analysis recover the true effect?
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_VISITORS = 96_000
START_DATE = "2026-06-01"
N_DAYS = 14

rng = np.random.default_rng(SEED)


def simulate() -> pd.DataFrame:
    # --- visitor attributes -------------------------------------------------
    user_id = np.arange(1, N_VISITORS + 1)

    # traffic is heavier on weekends
    day_weights = np.array([1.0, 0.95, 0.9, 0.95, 1.05, 1.35, 1.4] * 2)
    day_probs = day_weights / day_weights.sum()
    day_idx = rng.choice(N_DAYS, size=N_VISITORS, p=day_probs)
    date = pd.to_datetime(START_DATE) + pd.to_timedelta(day_idx, unit="D")

    group = np.where(rng.random(N_VISITORS) < 0.5, "treatment", "control")
    device = np.where(rng.random(N_VISITORS) < 0.62, "mobile", "desktop")
    user_type = np.where(rng.random(N_VISITORS) < 0.55, "new", "returning")
    traffic_source = rng.choice(
        ["search", "direct", "social", "email"],
        size=N_VISITORS,
        p=[0.40, 0.25, 0.20, 0.15],
    )

    # --- conversion probability --------------------------------------------
    p = np.where(device == "mobile", 0.052, 0.080)          # device baseline
    p = p + np.where(user_type == "returning", 0.015, 0.0)  # loyalty bump
    p = p + np.select(
        [traffic_source == "email", traffic_source == "social"],
        [0.010, -0.008],
        default=0.0,
    )
    # true treatment effect: one-page checkout helps mobile users most
    lift = np.where(device == "mobile", 0.010, 0.004)
    p = p + np.where(group == "treatment", lift, 0.0)
    # small day-level noise (shared shocks: promos, weather, etc.)
    day_noise = rng.normal(0, 0.002, N_DAYS)
    p = np.clip(p + day_noise[day_idx], 0.001, 0.999)

    converted = rng.random(N_VISITORS) < p

    # --- post-conversion outcomes -------------------------------------------
    n = N_VISITORS
    # AOV: lognormal, slightly higher on desktop, NO treatment effect
    aov_mu = np.where(device == "desktop", np.log(78), np.log(66))
    order_value = np.round(rng.lognormal(aov_mu, 0.55), 2)
    order_value = np.where(converted, order_value, np.nan)

    # checkout completion time (seconds): treatment is faster
    t_mu = np.where(group == "treatment", 140, 185)
    checkout_sec = np.round(np.clip(rng.normal(t_mu, 55), 25, None), 0)
    checkout_sec = np.where(converted, checkout_sec, np.nan)

    # refunds (guardrail): 4.0% vs 4.3% — a real but tiny, non-significant gap
    refund_p = np.where(group == "treatment", 0.043, 0.040)
    refunded = np.where(converted, rng.random(n) < refund_p, False)

    df = pd.DataFrame(
        {
            "user_id": user_id,
            "date": date.strftime("%Y-%m-%d"),
            "group": group,
            "device": device,
            "user_type": user_type,
            "traffic_source": traffic_source,
            "converted": converted.astype(int),
            "order_value": order_value,
            "checkout_seconds": checkout_sec,
            "refunded": refunded.astype(int),
        }
    )
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "checkout_ab_test.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = simulate()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows -> {out}")
    print(df.groupby("group")["converted"].agg(["count", "mean"]))
