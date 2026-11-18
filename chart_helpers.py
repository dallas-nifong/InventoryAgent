"""
Chart Data Calculations and Altair Chart Builders for Inventory Decision Models.

Provides pure functions for calculating plot data from deterministic calculation tool results,
and builds theme-aware Altair charts (compatible with both Streamlit light and dark modes)
with titles, axis labels with units, and one-sentence captions.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

try:
    import altair as alt
except ImportError:
    alt = None

try:
    import pandas as pd
except ImportError:
    pd = None


# =============================================================================
# 1. EOQ CHART DATA & BUILDER
# =============================================================================

def get_eoq_chart_data(tool_result: Any, num_points: int = 100) -> Dict[str, Any]:
    """
    Pure function: calculates EOQ cost curves for holding, ordering, and total cost
    against order quantity from ~20% to ~250% of Q*.
    Guarantees that the total cost curve achieves its global minimum at Q*.
    """
    global pd
    if pd is None:
        import pandas as pd

    res = getattr(tool_result, "calculation_result", tool_result)
    q_opt = float(getattr(res, "optimal_order_quantity", getattr(res, "eoq", 0.0)))
    d = float(getattr(res, "annual_demand", getattr(res, "demand", 0.0)))
    if d == 0.0 and hasattr(res, "inputs_extracted"):
        d = float(res.inputs_extracted.get("demand", 0.0))
    s = float(getattr(res, "order_cost", 0.0))
    h = float(getattr(res, "holding_cost", 0.0))
    min_cost = float(getattr(res, "annual_inventory_cost", 0.0))
    if min_cost == 0.0 and q_opt > 0:
        min_cost = (d / q_opt) * s + (q_opt / 2.0) * h

    q_min = max(1.0, 0.20 * q_opt)
    q_max = 2.50 * q_opt

    # Generate points across the 20% to 250% range, explicitly including Q*
    step = (q_max - q_min) / max(1, num_points - 1)
    q_vals = [q_min + i * step for i in range(num_points)]
    if not any(abs(q - q_opt) < 1e-4 for q in q_vals):
        q_vals.append(q_opt)
    q_vals = sorted(q_vals)

    plot_rows = []
    curve_points = []
    for q in q_vals:
        ordering_cost = (d / q) * s
        holding_cost = (q / 2.0) * h
        total_cost = ordering_cost + holding_cost
        curve_points.append({
            "order_quantity": q,
            "annual_ordering_cost": ordering_cost,
            "annual_holding_cost": holding_cost,
            "total_cost": total_cost,
        })
        plot_rows.append({"Order Quantity (units)": q, "Annual Cost ($/year)": ordering_cost, "Cost Component": "Annual Ordering Cost"})
        plot_rows.append({"Order Quantity (units)": q, "Annual Cost ($/year)": holding_cost, "Cost Component": "Annual Holding Cost"})
        plot_rows.append({"Order Quantity (units)": q, "Annual Cost ($/year)": total_cost, "Cost Component": "Total Annual Cost"})

    df = pd.DataFrame(plot_rows)
    q_display = getattr(res, "rounded_eoq", getattr(res, "rounded_order_quantity", round(q_opt)))
    caption = (
        f"Total annual inventory cost is minimized where annual ordering cost equals annual holding cost at "
        f"Q* = {q_display:,} units (${min_cost:,.2f}/year)."
    )
    return {
        "df": df,
        "curve_points": curve_points,
        "q_opt": q_opt,
        "q_display": q_display,
        "min_cost": min_cost,
        "title": "EOQ Cost Trade-off: Ordering Cost vs. Holding Cost",
        "x_label": "Order Quantity (units)",
        "y_label": "Annual Cost ($/year)",
        "caption": caption,
    }


def build_eoq_chart(data: Dict[str, Any]) -> Any:
    """Build theme-aware Altair chart for EOQ cost trade-off curves."""
    global alt, pd
    if alt is None:
        import altair as alt
    if pd is None:
        import pandas as pd

    df = data["df"]
    q_opt = data["q_opt"]
    min_cost = data["min_cost"]

    # Curves for AOC, AHC, TIC
    lines = (
        alt.Chart(df)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("Order Quantity (units):Q", title=data["x_label"]),
            y=alt.Y("Annual Cost ($/year):Q", title=data["y_label"]),
            color=alt.Color(
                "Cost Component:N",
                scale=alt.Scale(
                    domain=["Total Annual Cost", "Annual Ordering Cost", "Annual Holding Cost"],
                    range=["#1f77b4", "#ff7f0e", "#2ca02c"],
                ),
                title="Cost Component",
            ),
            tooltip=[
                alt.Tooltip("Order Quantity (units):Q", format=",.1f"),
                alt.Tooltip("Cost Component:N"),
                alt.Tooltip("Annual Cost ($/year):Q", format="$,.2f"),
            ],
        )
    )

    q_display = data.get("q_display", round(q_opt))
    # Point at Q*
    opt_df = pd.DataFrame([{
        "Order Quantity (units)": q_opt,
        "Annual Cost ($/year)": min_cost,
        "Label": f"Q* = {q_display:,} units (${min_cost:,.2f})",
    }])

    point = (
        alt.Chart(opt_df)
        .mark_circle(size=130, color="#d62728")
        .encode(
            x="Order Quantity (units):Q",
            y="Annual Cost ($/year):Q",
            tooltip=[alt.Tooltip("Label:N", title="Optimum")],
        )
    )

    text = (
        alt.Chart(opt_df)
        .mark_text(dy=-16, fontSize=11, fontWeight="bold", color="#d62728")
        .encode(
            x="Order Quantity (units):Q",
            y="Annual Cost ($/year):Q",
            text="Label:N",
        )
    )

    rule = (
        alt.Chart(opt_df)
        .mark_rule(strokeDash=[4, 4], color="#d62728", opacity=0.7)
        .encode(x="Order Quantity (units):Q")
    )

    return ((lines + rule + point + text)
            .properties(title=data["title"], height=380)
            .interactive())


# =============================================================================
# 2. QUANTITY DISCOUNT CHART DATA & BUILDER
# =============================================================================

def get_quantity_discount_chart_data(tool_result: Any) -> Dict[str, Any]:
    """
    Pure function: extracts total annual cost at evaluated quantity for each price tier,
    highlighting the lowest-cost tier, and builds a summary comparison table.
    """
    global pd
    if pd is None:
        import pandas as pd

    res = getattr(tool_result, "calculation_result", tool_result)
    tier_evals = getattr(res, "tier_evaluations", [])
    opt_idx = getattr(res, "optimal_tier_index", 0)
    min_tc = getattr(res, "min_total_annual_cost", 0.0)
    opt_q = getattr(res, "optimal_order_quantity", 0.0)

    plot_rows = []
    table_rows = []
    for t in tier_evals:
        idx = getattr(t, "tier_index", 0)
        tier_label = f"Tier {idx + 1}"
        q_val = round(float(getattr(t, "feasible_order_quantity", 0.0)))
        p_val = float(getattr(t, "unit_price", 0.0))
        tc_val = float(getattr(t, "total_annual_cost", 0.0))
        is_opt = (idx == opt_idx)
        highlight = "Optimal (Lowest Cost)" if is_opt else "Suboptimal"

        plot_rows.append({
            "Price Tier": tier_label,
            "Total Annual Cost ($/year)": tc_val,
            "Evaluated Quantity (units)": q_val,
            "Unit Price ($/unit)": p_val,
            "Tier Status": highlight,
        })
        table_rows.append({
            "Tier": tier_label,
            "Quantity": f"{q_val:,} units",
            "Unit Price": f"${p_val:.2f}",
            "Total Annual Cost": f"${tc_val:,.2f}",
            "Status": "🏆 Optimal" if is_opt else "Suboptimal",
        })

    df = pd.DataFrame(plot_rows)
    table_df = pd.DataFrame(table_rows)
    q_display = getattr(res, "rounded_optimal_quantity", round(opt_q))
    caption = (
        f"Tier {opt_idx + 1} achieves the minimum total annual cost of ${min_tc:,.2f} "
        f"at an evaluated order quantity of {q_display:,} units."
    )
    return {
        "df": df,
        "table_df": table_df,
        "optimal_tier_index": opt_idx,
        "optimal_tier_label": f"Tier {opt_idx + 1}",
        "opt_q": opt_q,
        "q_display": q_display,
        "title": "Total Annual Cost by Price Tier",
        "x_label": "Price Tier",
        "y_label": "Total Annual Cost ($/year)",
        "caption": caption,
    }


def build_quantity_discount_chart(data: Dict[str, Any]) -> Any:
    """Build theme-aware Altair bar chart comparing price tiers with lowest cost highlighted."""
    global alt
    if alt is None:
        import altair as alt

    df = data["df"]
    bars = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Price Tier:N", sort=None, title=data["x_label"]),
            y=alt.Y("Total Annual Cost ($/year):Q", title=data["y_label"], scale=alt.Scale(nice=True, padding=25)),
            color=alt.Color(
                "Tier Status:N",
                scale=alt.Scale(
                    domain=["Optimal (Lowest Cost)", "Suboptimal"],
                    range=["#2ca02c", "#aec7e8"],
                ),
                title="Tier Status",
            ),
            tooltip=[
                alt.Tooltip("Price Tier:N"),
                alt.Tooltip("Evaluated Quantity (units):Q", format=","),
                alt.Tooltip("Unit Price ($/unit):Q", format="$,.2f"),
                alt.Tooltip("Total Annual Cost ($/year):Q", format="$,.2f"),
                alt.Tooltip("Tier Status:N"),
            ],
        )
    )

    text = (
        alt.Chart(df)
        .mark_text(dy=-10, fontSize=11, fontWeight="bold", color="#1f77b4")
        .encode(
            x=alt.X("Price Tier:N", sort=None),
            y=alt.Y("Total Annual Cost ($/year):Q"),
            text=alt.Text("Total Annual Cost ($/year):Q", format="$,.0f"),
        )
    )

    return (bars + text).properties(title=data["title"], height=350)


# =============================================================================
# 3. SAFETY STOCK & ROP CHART DATA & BUILDER
# =============================================================================

def get_safety_stock_chart_data(tool_result: Any, num_points: int = 150) -> Optional[Dict[str, Any]]:
    """
    Pure function: calculates normal curve of lead-time demand, highlighting expected demand,
    reorder point (ROP), safety stock gap, and shaded stockout-risk tail.
    Returns None if there is no lead-time demand variability (std_lead_time_demand <= 0).
    """
    global pd
    if pd is None:
        import pandas as pd

    res = getattr(tool_result, "calculation_result", tool_result)
    sigma_lt = float(getattr(res, "std_lead_time_demand", 0.0))
    if sigma_lt <= 0.0:
        return None

    mu_dl = float(getattr(res, "expected_lead_time_demand", 0.0))
    rop = float(getattr(res, "reorder_point", 0.0))
    ss = float(getattr(res, "safety_stock", 0.0))
    sl = float(getattr(res, "service_level", 0.95)) if getattr(res, "service_level", None) is not None else 0.95

    x_min = max(0.0, mu_dl - 3.5 * sigma_lt)
    x_max = max(rop + 1.5 * sigma_lt, mu_dl + 3.5 * sigma_lt)

    step = (x_max - x_min) / max(1, num_points - 1)
    points = []
    inv_sigma_sqrt_2pi = 1.0 / (sigma_lt * math.sqrt(2 * math.pi))
    two_var = 2.0 * (sigma_lt ** 2)

    for i in range(num_points):
        x = x_min + i * step
        density = inv_sigma_sqrt_2pi * math.exp(-((x - mu_dl) ** 2) / two_var)
        is_tail = (x >= rop)
        region = "Stockout Risk Tail" if is_tail else "In-Stock Probability"
        points.append({
            "Lead-Time Demand (units)": x,
            "Probability Density": density,
            "Region": region,
        })

    df = pd.DataFrame(points)
    ss_display = getattr(res, "rounded_safety_stock", round(ss))
    rop_display = getattr(res, "rounded_reorder_point", round(rop))
    caption = (
        f"Normal distribution of lead-time demand (mean={round(mu_dl):,}, std={sigma_lt:.1f}) showing "
        f"expected lead-time demand ({round(mu_dl):,} units), reorder point ({rop_display:,} units), and "
        f"safety stock buffer ({ss_display:,} units) protecting the {sl * 100:.1f}% service level."
    )
    return {
        "df": df,
        "mu_dl": mu_dl,
        "rop": rop,
        "ss": ss,
        "ss_display": ss_display,
        "rop_display": rop_display,
        "sl": sl,
        "sigma_lt": sigma_lt,
        "title": "Lead-Time Demand Distribution & Safety Stock Buffer",
        "x_label": "Lead-Time Demand (units)",
        "y_label": "Probability Density",
        "caption": caption,
    }


def build_safety_stock_chart(data: Dict[str, Any]) -> Any:
    """Build theme-aware Altair density chart with shaded stockout tail and labeled safety stock gap."""
    global alt, pd
    if alt is None:
        import altair as alt
    if pd is None:
        import pandas as pd

    df = data["df"]
    mu_dl = data["mu_dl"]
    rop = data["rop"]
    ss = data["ss"]
    ss_display = data.get("ss_display", round(ss))
    rop_display = data.get("rop_display", round(rop))

    # Shaded density area with stockout tail highlighted
    area = (
        alt.Chart(df)
        .mark_area(opacity=0.6)
        .encode(
            x=alt.X("Lead-Time Demand (units):Q", title=data["x_label"]),
            y=alt.Y("Probability Density:Q", title=data["y_label"]),
            color=alt.Color(
                "Region:N",
                scale=alt.Scale(
                    domain=["In-Stock Probability", "Stockout Risk Tail"],
                    range=["#4c78a8", "#e45756"],
                ),
                title="Service Level",
            ),
            tooltip=[
                alt.Tooltip("Lead-Time Demand (units):Q", format=",.1f"),
                alt.Tooltip("Probability Density:Q", format=".6f"),
                alt.Tooltip("Region:N"),
            ],
        )
    )

    # Vertical line at Expected Lead-Time Demand (mu_DL)
    mu_df = pd.DataFrame([{"x": mu_dl, "Label": f"Expected Demand: {round(mu_dl):,} units"}])
    rule_mu = (
        alt.Chart(mu_df)
        .mark_rule(color="#1f77b4", strokeWidth=2, strokeDash=[4, 4])
        .encode(x="x:Q")
    )
    text_mu = (
        alt.Chart(mu_df)
        .mark_text(align="right", dx=-6, dy=0, fontSize=11, fontWeight="bold", color="#1f77b4")
        .encode(x="x:Q", y=alt.value(14), text="Label:N")
    )

    # Vertical line at Reorder Point (ROP)
    rop_df = pd.DataFrame([{"x": rop, "Label": f"Reorder Point: {rop_display:,} units"}])
    rule_rop = (
        alt.Chart(rop_df)
        .mark_rule(color="#d62728", strokeWidth=2.5)
        .encode(x="x:Q")
    )
    text_rop = (
        alt.Chart(rop_df)
        .mark_text(align="left", dx=6, dy=0, fontSize=11, fontWeight="bold", color="#d62728")
        .encode(x="x:Q", y=alt.value(14), text="Label:N")
    )

    # Label for Safety Stock gap between mu_dl and rop
    gap_x = (mu_dl + rop) / 2.0
    gap_df = pd.DataFrame([{"x": gap_x, "Label": f"Safety Stock (SS) = {ss_display:,} units"}])
    text_gap = (
        alt.Chart(gap_df)
        .mark_text(align="center", dx=0, dy=0, fontSize=11, fontWeight="bold", color="#2ca02c")
        .encode(x="x:Q", y=alt.value(34), text="Label:N")
    )

    return ((area + rule_mu + text_mu + rule_rop + text_rop + text_gap)
            .properties(title=data["title"], height=360)
            .interactive())


# =============================================================================
# 4. FIXED-PERIOD REVIEW CHART DATA & BUILDER
# =============================================================================

def get_periodic_review_chart_data(tool_result: Any) -> Dict[str, Any]:
    """
    Pure function: builds stacked bar data showing:
    Target Inventory (S) = Expected Protection Demand + Safety Stock
    vs Current Inventory Position (IP) with the amount to order (q = S - IP) labeled.
    """
    global pd
    if pd is None:
        import pandas as pd

    res = getattr(tool_result, "calculation_result", tool_result)
    demand_prot = float(getattr(res, "demand_during_protection", getattr(res, "expected_protection_demand", 0.0)))
    ss = float(getattr(res, "safety_stock", 0.0))
    target_s = float(getattr(res, "target_inventory", demand_prot + ss))
    ip = float(getattr(res, "inventory_position", 0.0))
    q = float(getattr(res, "order_quantity", max(0.0, target_s - ip)))

    rows = [
        {"Inventory Metric": "Target Inventory (S)", "Component": "Expected Demand (T+L)", "Units (units)": demand_prot},
        {"Inventory Metric": "Target Inventory (S)", "Component": "Safety Stock (SS)", "Units (units)": ss},
        {"Inventory Metric": "Current Position (IP)", "Component": "Inventory Position (IP)", "Units (units)": ip},
    ]
    df = pd.DataFrame(rows)
    q_display = getattr(res, "rounded_order_quantity", round(q))
    target_s_display = getattr(res, "rounded_target_inventory", round(target_s))
    ss_display = getattr(res, "rounded_safety_stock", round(ss))
    caption = (
        f"Target inventory of {target_s_display:,} units comprises expected protection-period demand ({round(demand_prot):,} units) "
        f"plus safety stock buffer ({ss_display:,} units); ordering {q_display:,} units raises current inventory position ({round(ip):,} units) to target stock."
    )
    return {
        "df": df,
        "demand_during_protection": demand_prot,
        "safety_stock": ss,
        "target_inventory": target_s,
        "inventory_position": ip,
        "order_quantity": q,
        "q_display": q_display,
        "target_s_display": target_s_display,
        "ss_display": ss_display,
        "title": "Periodic Review Policy: Target Inventory Level vs. Current Position",
        "x_label": "Inventory Policy Metric",
        "y_label": "Inventory Units (units)",
        "caption": caption,
    }


def build_periodic_review_chart(data: Dict[str, Any]) -> Any:
    """Build theme-aware Altair stacked bar chart comparing Target Inventory and Inventory Position."""
    global alt, pd
    if alt is None:
        import altair as alt
    if pd is None:
        import pandas as pd

    df = data["df"]
    q_display = data.get("q_display", round(data.get("order_quantity", 0)))

    bars = (
        alt.Chart(df)
        .mark_bar(width=65, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Inventory Metric:N", sort=None, title=data["x_label"]),
            y=alt.Y("Units (units):Q", stack="zero", title=data["y_label"], scale=alt.Scale(nice=True, padding=25)),
            color=alt.Color(
                "Component:N",
                scale=alt.Scale(
                    domain=["Expected Demand (T+L)", "Safety Stock (SS)", "Inventory Position (IP)"],
                    range=["#4c78a8", "#f58518", "#72b7b2"],
                ),
                title="Component",
            ),
            tooltip=[
                alt.Tooltip("Inventory Metric:N"),
                alt.Tooltip("Component:N"),
                alt.Tooltip("Units (units):Q", format=",.1f"),
            ],
        )
    )

    # Text label for amount to order anchored inside the plot area
    annot_df = pd.DataFrame([{
        "Inventory Metric": "Current Position (IP)",
        "Label": f"Amount to Order: q = {q_display:,} units",
    }])
    text_annot = (
        alt.Chart(annot_df)
        .mark_text(align="center", dy=0, fontSize=11, fontWeight="bold", color="#d62728")
        .encode(
            x="Inventory Metric:N",
            y=alt.value(14),
            text="Label:N",
        )
    )

    return (bars + text_annot).properties(title=data["title"], height=360)


# =============================================================================
# 5. NEWSVENDOR CHART DATA & BUILDER
# =============================================================================

def get_newsvendor_chart_data(tool_result: Any, num_points: int = 150) -> Dict[str, Any]:
    """
    Pure function:
    - Discrete mode: cumulative probability by demand level with horizontal line at critical ratio
      and the chosen quantity highlighted. Guarantees chosen quantity is the first demand level where CP >= CR.
    - Normal mode: normal demand density curve with vertical line at optimal order quantity.
    """
    global pd
    if pd is None:
        import pandas as pd

    res = getattr(tool_result, "calculation_result", tool_result)
    cr = float(getattr(res, "critical_ratio", 0.0))
    q_opt = float(getattr(res, "optimal_order_quantity", getattr(res, "recommended_order_quantity", 0.0)))
    tbl = getattr(res, "cumulative_probability_table", None)
    if tbl is None and hasattr(res, "intermediate_calculations"):
        tbl = res.intermediate_calculations.get("cumulative_probability_table")

    if tbl and len(tbl) > 0:
        plot_rows = []
        for row in tbl:
            d_val = float(row.get("demand", 0.0))
            p_val = float(row.get("probability", 0.0))
            cp_val = float(row.get("cumulative_probability", 0.0))
            is_chosen = (d_val == q_opt)
            highlight = "Optimal Order Quantity Q*" if is_chosen else "Demand Level"
            plot_rows.append({
                "Demand (units)": d_val,
                "Probability": p_val,
                "Cumulative Probability P(D ≤ d)": cp_val,
                "Status": highlight,
            })
        df = pd.DataFrame(plot_rows)
        q_display = getattr(res, "rounded_optimal_quantity", getattr(res, "rounded_order_quantity", round(q_opt)))
        caption = (
            f"Recommended order quantity ({q_display:,} units) is the smallest demand level whose "
            f"cumulative probability satisfies the {cr * 100:.1f}% critical ratio."
        )
        return {
            "mode": "discrete",
            "df": df,
            "critical_ratio": cr,
            "chosen_quantity": q_opt,
            "q_display": q_display,
            "title": "Discrete Demand Cumulative Distribution & Critical Ratio",
            "x_label": "Demand (units)",
            "y_label": "Cumulative Probability P(D ≤ d)",
            "caption": caption,
        }
    else:
        mu = float(getattr(res, "mean_demand", 0.0))
        sigma = float(getattr(res, "std_demand", 0.0))
        x_min = max(0.0, mu - 3.5 * sigma)
        x_max = max(q_opt + 1.5 * sigma, mu + 3.5 * sigma)
        step = (x_max - x_min) / max(1, num_points - 1)

        inv_sigma_sqrt_2pi = 1.0 / (sigma * math.sqrt(2 * math.pi)) if sigma > 0 else 1.0
        two_var = 2.0 * (sigma ** 2) if sigma > 0 else 1.0

        points = []
        for i in range(num_points):
            x = x_min + i * step
            density = inv_sigma_sqrt_2pi * math.exp(-((x - mu) ** 2) / two_var) if sigma > 0 else 0.0
            is_in_stock = (x <= q_opt)
            region = "In-Stock Coverage (CR)" if is_in_stock else "Uncovered Tail"
            points.append({
                "Demand (units)": x,
                "Probability Density": density,
                "Region": region,
            })
        df = pd.DataFrame(points)
        q_display = getattr(res, "rounded_optimal_quantity", getattr(res, "rounded_order_quantity", round(q_opt)))
        caption = (
            f"Normal demand distribution (mean={round(mu):,}, std={sigma:.1f}) with optimal order quantity "
            f"Q* = {q_display:,} units satisfying the {cr * 100:.1f}% critical ratio fractile."
        )
        return {
            "mode": "normal",
            "df": df,
            "mean": mu,
            "std": sigma,
            "critical_ratio": cr,
            "q_opt": q_opt,
            "q_display": q_display,
            "title": "Newsvendor Demand Distribution & Optimal Order Quantity",
            "x_label": "Demand (units)",
            "y_label": "Probability Density",
            "caption": caption,
        }


def build_newsvendor_chart(data: Dict[str, Any]) -> Any:
    """Build theme-aware Altair chart for discrete or normal newsvendor distribution."""
    global alt, pd
    if alt is None:
        import altair as alt
    if pd is None:
        import pandas as pd

    if data["mode"] == "discrete":
        df = data["df"]
        cr = data["critical_ratio"]
        bars = (
            alt.Chart(df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Demand (units):O", title=data["x_label"]),
                y=alt.Y("Cumulative Probability P(D ≤ d):Q", title=data["y_label"], scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "Status:N",
                    scale=alt.Scale(
                        domain=["Optimal Order Quantity Q*", "Demand Level"],
                        range=["#2ca02c", "#aec7e8"],
                    ),
                    title="Selection",
                ),
                tooltip=[
                    alt.Tooltip("Demand (units):O"),
                    alt.Tooltip("Probability:Q", format=".2%"),
                    alt.Tooltip("Cumulative Probability P(D ≤ d):Q", format=".4f"),
                    alt.Tooltip("Status:N"),
                ],
            )
        )
        cr_df = pd.DataFrame([{"y": cr, "Label": f"Critical Ratio = {cr * 100:.1f}%"}])
        rule_cr = (
            alt.Chart(cr_df)
            .mark_rule(color="#d62728", strokeWidth=2, strokeDash=[4, 4])
            .encode(y="y:Q")
        )
        text_cr = (
            alt.Chart(cr_df)
            .mark_text(align="right", dx=-10, dy=-8, fontSize=11, fontWeight="bold", color="#d62728")
            .encode(y="y:Q", text="Label:N")
        )
        return (bars + rule_cr + text_cr).properties(title=data["title"], height=350)
    else:
        df = data["df"]
        q_opt = data["q_opt"]
        area = (
            alt.Chart(df)
            .mark_area(opacity=0.6)
            .encode(
                x=alt.X("Demand (units):Q", title=data["x_label"]),
                y=alt.Y("Probability Density:Q", title=data["y_label"]),
                color=alt.Color(
                    "Region:N",
                    scale=alt.Scale(
                        domain=["In-Stock Coverage (CR)", "Uncovered Tail"],
                        range=["#2ca02c", "#aec7e8"],
                    ),
                    title="Coverage",
                ),
                tooltip=[
                    alt.Tooltip("Demand (units):Q", format=",.1f"),
                    alt.Tooltip("Probability Density:Q", format=".6f"),
                ],
            )
        )
        q_display = data.get("q_display", round(q_opt))
        q_df = pd.DataFrame([{"x": q_opt, "Label": f"Q* = {q_display:,} units"}])
        rule_q = (
            alt.Chart(q_df)
            .mark_rule(color="#d62728", strokeWidth=2.5)
            .encode(x="x:Q")
        )
        text_q = (
            alt.Chart(q_df)
            .mark_text(align="left", dx=6, dy=0, fontSize=11, fontWeight="bold", color="#d62728")
            .encode(x="x:Q", y=alt.value(14), text="Label:N")
        )
        return (area + rule_q + text_q).properties(title=data["title"], height=350).interactive()


# =============================================================================
# 6. UNIFIED MODEL DISPATCHER
# =============================================================================

def build_model_chart(resp_or_result: Any, model_name: Optional[str] = None) -> Optional[Tuple[Any, str, Optional[Any]]]:
    """
    Convenience orchestrator that returns:
    (altair_chart, caption_string, optional_price_break_table_dataframe)
    or None if model is unsupported or has no chart (e.g. ROP with no variability).
    """
    if resp_or_result is None:
        return None

    if hasattr(resp_or_result, "calculation_result"):
        res = resp_or_result.calculation_result
        m = resp_or_result.model_name
    else:
        res = resp_or_result
        m = model_name or getattr(res, "model_name", "")

    if res is None:
        return None

    if m == "EOQ":
        data = get_eoq_chart_data(res)
        chart = build_eoq_chart(data)
        return chart, data["caption"], None

    elif m == "QUANTITY_DISCOUNT":
        data = get_quantity_discount_chart_data(res)
        chart = build_quantity_discount_chart(data)
        return chart, data["caption"], data["table_df"]

    elif m == "SAFETY_STOCK_ROP":
        data = get_safety_stock_chart_data(res)
        if data is None:
            return None
        chart = build_safety_stock_chart(data)
        return chart, data["caption"], None

    elif m == "FIXED_PERIOD_REVIEW":
        data = get_periodic_review_chart_data(res)
        chart = build_periodic_review_chart(data)
        return chart, data["caption"], None

    elif m == "NEWSVENDOR":
        data = get_newsvendor_chart_data(res)
        chart = build_newsvendor_chart(data)
        return chart, data["caption"], None

    return None
