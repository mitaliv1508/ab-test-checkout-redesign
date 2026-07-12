"""
End-to-end statistical analysis of the checkout redesign A/B test.

Implements the core statistics from first principles (numpy + stdlib
NormalDist) rather than calling black-box library functions — every
formula is visible and auditable.

Steps
-----
1. Sanity checks: sample ratio mismatch (SRM), missing data
2. Power analysis: minimum detectable effect for this sample size
3. Primary metric: conversion rate — two-proportion z-test + 95% CI
4. Secondary metrics: AOV (bootstrap CI), checkout time
5. Guardrail: refund rate
6. Segmentation: device & user type, Holm-corrected
7. Novelty-effect check: daily treatment effect over time
8. Business impact estimate

Outputs: figures/*.png and results/summary.json
"""

import json
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

Z = NormalDist()  # standard normal
ALPHA = 0.05
RNG = np.random.default_rng(7)

C_CONTROL, C_TREAT = "#8896a5", "#2f6fde"


# --------------------------------------------------------------------------
# Statistical helpers (from first principles)
# --------------------------------------------------------------------------
def two_proportion_ztest(x1, n1, x2, n2):
    """H0: p1 == p2. Returns z, two-sided p, diff, 95% CI for the diff."""
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se_pool
    p_val = 2 * (1 - Z.cdf(abs(z)))
    # unpooled SE for the confidence interval
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zc = Z.inv_cdf(1 - ALPHA / 2)
    diff = p2 - p1
    return z, p_val, diff, (diff - zc * se, diff + zc * se)


def srm_check(n1, n2, expected_ratio=0.5):
    """Chi-square (1 df) goodness-of-fit test for sample ratio mismatch."""
    total = n1 + n2
    exp = total * expected_ratio
    chi2 = (n1 - exp) ** 2 / exp + (n2 - exp) ** 2 / exp
    p_val = 2 * (1 - Z.cdf(np.sqrt(chi2)))  # chi2(1) survival via normal
    return chi2, p_val


def mde_two_proportions(p_base, n_per_arm, power=0.80):
    """Minimum detectable absolute lift (two-sided alpha, given power)."""
    za = Z.inv_cdf(1 - ALPHA / 2)
    zb = Z.inv_cdf(power)
    return (za + zb) * np.sqrt(2 * p_base * (1 - p_base) / n_per_arm)


def welch_test(a, b):
    """Welch z-test for difference in means (normal approx, large n)."""
    m1, m2 = a.mean(), b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    z = (m2 - m1) / se
    p_val = 2 * (1 - Z.cdf(abs(z)))
    zc = Z.inv_cdf(1 - ALPHA / 2)
    diff = m2 - m1
    return z, p_val, diff, (diff - zc * se, diff + zc * se)


def bootstrap_mean_diff(a, b, n_boot=10_000):
    """Percentile bootstrap 95% CI for mean(b) - mean(a)."""
    a, b = np.asarray(a), np.asarray(b)
    idx_a = RNG.integers(0, len(a), (n_boot, len(a)))
    idx_b = RNG.integers(0, len(b), (n_boot, len(b)))
    diffs = b[idx_b].mean(axis=1) - a[idx_a].mean(axis=1)
    return np.percentile(diffs, [2.5, 97.5])


def holm_correction(pvals):
    """Holm-Bonferroni adjusted p-values."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, (m - rank) * pvals[i])
        adj[i] = min(running_max, 1.0)
    return adj


# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------
df = pd.read_csv(ROOT / "data" / "checkout_ab_test.csv", parse_dates=["date"])
ctrl = df[df["group"] == "control"]
trt = df[df["group"] == "treatment"]
summary = {"n_total": len(df), "n_control": len(ctrl), "n_treatment": len(trt)}

print(f"Loaded {len(df):,} visitors "
      f"({len(ctrl):,} control / {len(trt):,} treatment)\n")

# 1. Sanity checks -----------------------------------------------------------
chi2, p_srm = srm_check(len(ctrl), len(trt))
summary["srm"] = {"chi2": round(chi2, 3), "p_value": round(p_srm, 4)}
print(f"[SRM check] chi2={chi2:.3f}, p={p_srm:.4f} -> "
      f"{'OK: no sample ratio mismatch' if p_srm > 0.01 else 'WARNING: SRM!'}")

# 2. Power analysis ----------------------------------------------------------
p_base = ctrl["converted"].mean()
mde = mde_two_proportions(p_base, len(ctrl))
summary["power"] = {
    "baseline_conversion": round(p_base, 4),
    "mde_abs_pp": round(mde * 100, 3),
    "mde_relative_pct": round(mde / p_base * 100, 2),
}
print(f"[Power] baseline={p_base:.2%}; MDE at 80% power = "
      f"{mde*100:.2f} pp ({mde/p_base:.1%} relative)\n")

# 3. Primary metric: conversion ----------------------------------------------
x1, n1 = ctrl["converted"].sum(), len(ctrl)
x2, n2 = trt["converted"].sum(), len(trt)
z, p_val, diff, ci = two_proportion_ztest(x1, n1, x2, n2)
rel = diff / (x1 / n1)
summary["primary_conversion"] = {
    "control_rate": round(x1 / n1, 5),
    "treatment_rate": round(x2 / n2, 5),
    "abs_lift_pp": round(diff * 100, 3),
    "rel_lift_pct": round(rel * 100, 2),
    "ci95_pp": [round(ci[0] * 100, 3), round(ci[1] * 100, 3)],
    "z": round(z, 2),
    "p_value": float(f"{p_val:.2e}"),
    "significant": bool(p_val < ALPHA),
}
print(f"[Primary: conversion] control={x1/n1:.2%}, treatment={x2/n2:.2%}")
print(f"  lift = +{diff*100:.2f} pp ({rel:+.1%} relative), "
      f"95% CI [{ci[0]*100:.2f}, {ci[1]*100:.2f}] pp, z={z:.2f}, p={p_val:.1e}\n")

# 4. Secondary metrics ---------------------------------------------------------
aov_c = ctrl.loc[ctrl["converted"] == 1, "order_value"].dropna()
aov_t = trt.loc[trt["converted"] == 1, "order_value"].dropna()
_, p_aov, d_aov, _ = welch_test(aov_c, aov_t)
ci_aov = bootstrap_mean_diff(aov_c, aov_t)
summary["aov"] = {
    "control": round(aov_c.mean(), 2),
    "treatment": round(aov_t.mean(), 2),
    "diff": round(d_aov, 2),
    "bootstrap_ci95": [round(ci_aov[0], 2), round(ci_aov[1], 2)],
    "p_value": round(p_aov, 3),
    "significant": bool(p_aov < ALPHA),
}
print(f"[Secondary: AOV] ${aov_c.mean():.2f} vs ${aov_t.mean():.2f}, "
      f"diff=${d_aov:+.2f}, bootstrap CI [{ci_aov[0]:.2f}, {ci_aov[1]:.2f}], "
      f"p={p_aov:.3f} -> {'significant' if p_aov < ALPHA else 'no difference'}")

sec_c = ctrl.loc[ctrl["converted"] == 1, "checkout_seconds"].dropna()
sec_t = trt.loc[trt["converted"] == 1, "checkout_seconds"].dropna()
_, p_sec, d_sec, ci_sec = welch_test(sec_c, sec_t)
summary["checkout_seconds"] = {
    "control": round(sec_c.mean(), 1),
    "treatment": round(sec_t.mean(), 1),
    "diff": round(d_sec, 1),
    "ci95": [round(ci_sec[0], 1), round(ci_sec[1], 1)],
    "p_value": float(f"{p_sec:.2e}"),
}
print(f"[Secondary: checkout time] {sec_c.mean():.0f}s vs {sec_t.mean():.0f}s, "
      f"diff={d_sec:+.0f}s, p={p_sec:.1e}")

# 5. Guardrail: refunds --------------------------------------------------------
rx1, rn1 = ctrl.loc[ctrl["converted"] == 1, "refunded"].agg(["sum", "count"])
rx2, rn2 = trt.loc[trt["converted"] == 1, "refunded"].agg(["sum", "count"])
_, p_ref, d_ref, ci_ref = two_proportion_ztest(rx1, rn1, rx2, rn2)
summary["guardrail_refunds"] = {
    "control_rate": round(rx1 / rn1, 4),
    "treatment_rate": round(rx2 / rn2, 4),
    "diff_pp": round(d_ref * 100, 2),
    "ci95_pp": [round(ci_ref[0] * 100, 2), round(ci_ref[1] * 100, 2)],
    "p_value": round(p_ref, 3),
    "passed": bool(p_ref >= ALPHA or d_ref <= 0),
}
print(f"[Guardrail: refunds] {rx1/rn1:.2%} vs {rx2/rn2:.2%}, "
      f"diff={d_ref*100:+.2f} pp, p={p_ref:.3f} -> "
      f"{'PASS (no significant harm)' if p_ref >= ALPHA else 'REVIEW'}\n")

# 6. Segmentation --------------------------------------------------------------
segments, seg_p = [], []
for col in ["device", "user_type"]:
    for val in sorted(df[col].unique()):
        c = ctrl[ctrl[col] == val]
        t = trt[trt[col] == val]
        z_s, p_s, d_s, ci_s = two_proportion_ztest(
            c["converted"].sum(), len(c), t["converted"].sum(), len(t)
        )
        segments.append(
            {"segment": f"{col}={val}", "control": c["converted"].mean(),
             "treatment": t["converted"].mean(), "lift_pp": d_s * 100,
             "ci_lo": ci_s[0] * 100, "ci_hi": ci_s[1] * 100, "p_raw": p_s}
        )
        seg_p.append(p_s)

adj = holm_correction(np.array(seg_p))
for s, a in zip(segments, adj):
    s["p_holm"] = a
summary["segments"] = [
    {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}
    for s in segments
]
print("[Segments] (Holm-adjusted p-values)")
for s in segments:
    print(f"  {s['segment']:<22} lift={s['lift_pp']:+.2f} pp "
          f"[{s['ci_lo']:.2f}, {s['ci_hi']:.2f}], p_holm={s['p_holm']:.4f}")

# 7. Novelty check -------------------------------------------------------------
daily = (
    df.groupby(["date", "group"])["converted"].mean().unstack()
)
daily["lift_pp"] = (daily["treatment"] - daily["control"]) * 100
first_week = daily["lift_pp"].iloc[:7].mean()
second_week = daily["lift_pp"].iloc[7:].mean()
summary["novelty"] = {
    "week1_avg_lift_pp": round(first_week, 2),
    "week2_avg_lift_pp": round(second_week, 2),
}
print(f"\n[Novelty check] avg daily lift week 1 = {first_week:+.2f} pp, "
      f"week 2 = {second_week:+.2f} pp -> "
      f"{'stable effect, no novelty decay' if abs(first_week - second_week) < 0.5 else 'possible novelty effect'}")

# 8. Business impact -----------------------------------------------------------
ANNUAL_CHECKOUT_VISITORS = 2_500_000
extra_orders = ANNUAL_CHECKOUT_VISITORS * diff
extra_orders_lo = ANNUAL_CHECKOUT_VISITORS * ci[0]
extra_orders_hi = ANNUAL_CHECKOUT_VISITORS * ci[1]
blended_aov = df["order_value"].dropna().mean()
summary["business_impact"] = {
    "assumed_annual_checkout_visitors": ANNUAL_CHECKOUT_VISITORS,
    "blended_aov": round(blended_aov, 2),
    "extra_annual_orders": int(extra_orders),
    "extra_annual_revenue": int(extra_orders * blended_aov),
    "revenue_ci95": [int(extra_orders_lo * blended_aov),
                     int(extra_orders_hi * blended_aov)],
}
print(f"\n[Impact] At {ANNUAL_CHECKOUT_VISITORS:,} checkout visitors/yr and "
      f"AOV ${blended_aov:.2f}:")
print(f"  ~{extra_orders:,.0f} extra orders/yr -> "
      f"~${extra_orders * blended_aov:,.0f} incremental revenue "
      f"(95% CI ${extra_orders_lo * blended_aov:,.0f} – "
      f"${extra_orders_hi * blended_aov:,.0f})")

# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

# Fig 1: conversion by group with 95% CI
fig, ax = plt.subplots(figsize=(6, 4))
rates = [x1 / n1, x2 / n2]
errs = [Z.inv_cdf(0.975) * np.sqrt(r * (1 - r) / n) for r, n in
        zip(rates, [n1, n2])]
bars = ax.bar(["Control\n(3-step checkout)", "Treatment\n(one-page checkout)"],
              [r * 100 for r in rates], yerr=[e * 100 for e in errs],
              capsize=6, color=[C_CONTROL, C_TREAT], width=0.55)
for b, r in zip(bars, rates):
    ax.text(b.get_x() + b.get_width() / 2, r * 100 + 0.25, f"{r:.2%}",
            ha="center", fontweight="bold")
ax.set_ylabel("Conversion rate (%)")
ax.set_title(f"Checkout conversion: +{diff*100:.2f} pp lift "
             f"({rel:+.1%} relative), p={p_val:.1e}")
fig.tight_layout()
fig.savefig(FIG / "fig1_conversion_by_group.png")
plt.close(fig)

# Fig 2: daily conversion + lift over time
fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1]})
axes[0].plot(daily.index, daily["control"] * 100, marker="o", ms=3.5,
             color=C_CONTROL, label="Control")
axes[0].plot(daily.index, daily["treatment"] * 100, marker="o", ms=3.5,
             color=C_TREAT, label="Treatment")
axes[0].set_ylabel("Daily conversion (%)")
axes[0].legend(frameon=False)
axes[0].set_title("Daily conversion by group — treatment leads consistently "
                  "(no novelty decay)")
axes[1].bar(daily.index, daily["lift_pp"], color=C_TREAT, alpha=0.75)
axes[1].axhline(0, color="black", lw=0.8)
axes[1].axhline(diff * 100, color="crimson", ls="--", lw=1,
                label=f"Overall lift +{diff*100:.2f} pp")
axes[1].set_ylabel("Daily lift (pp)")
axes[1].legend(frameon=False)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIG / "fig2_daily_lift.png")
plt.close(fig)

# Fig 3: segment forest plot
fig, ax = plt.subplots(figsize=(7, 4))
ys = np.arange(len(segments))[::-1]
for y, s in zip(ys, segments):
    sig = s["p_holm"] < ALPHA
    color = C_TREAT if sig else C_CONTROL
    ax.errorbar(s["lift_pp"], y,
                xerr=[[s["lift_pp"] - s["ci_lo"]], [s["ci_hi"] - s["lift_pp"]]],
                fmt="o", color=color, capsize=4, ms=6)
    ax.text(s["ci_hi"] + 0.06, y, f"p={s['p_holm']:.3f}", va="center",
            fontsize=8.5)
ax.axvline(0, color="black", lw=0.8)
ax.axvline(diff * 100, color="crimson", ls="--", lw=1,
           label=f"Overall +{diff*100:.2f} pp")
ax.set_yticks(ys)
ax.set_yticklabels([s["segment"] for s in segments])
ax.set_xlabel("Conversion lift (pp) with 95% CI")
ax.set_title("Treatment effect by segment (Holm-corrected)")
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "fig3_segment_lift.png")
plt.close(fig)

# Fig 4: order value distributions
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(0, 300, 60)
ax.hist(aov_c, bins=bins, alpha=0.55, density=True, color=C_CONTROL,
        label=f"Control (mean ${aov_c.mean():.2f})")
ax.hist(aov_t, bins=bins, alpha=0.55, density=True, color=C_TREAT,
        label=f"Treatment (mean ${aov_t.mean():.2f})")
ax.set_xlabel("Order value ($)")
ax.set_ylabel("Density")
ax.set_title(f"Order value unchanged (p={p_aov:.2f}) — "
             "lift comes from more orders, not bigger orders")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "fig4_order_value.png")
plt.close(fig)

with open(RES / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved 4 figures -> {FIG}")
print(f"Saved metrics  -> {RES / 'summary.json'}")
