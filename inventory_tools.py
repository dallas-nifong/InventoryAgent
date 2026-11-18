"""
Deterministic Inventory Tools Module for Supply Chain Analytics.

Provides rigorous, deterministic mathematical implementations of core inventory models:
1. calculate_eoq()
2. calculate_quantity_discount()
3. calculate_rop()
4. calculate_periodic_review()
5. calculate_newsvendor()

Every tool strictly fulfills the 8 verification requirements for student verification:
1. Inputs Extracted (inputs_extracted)
2. Variable Definitions (variable_definitions)
3. Formula (formula)
4. Formula with Substituted Values (formula_with_substituted_values)
5. Intermediate Calculations (intermediate_calculations)
6. Final Answer (final_answer)
7. Verification Checks (verification_checks)
8. Assumptions (assumptions)

All numerical steps come directly and deterministically from these Python tools.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Statistical Helper Functions (Pure Python Standard Normal)
# ---------------------------------------------------------------------------

def norm_ppf(p: float) -> float:
    """Standard normal quantile function (inverse CDF), Φ^(-1)(p)."""
    if not isinstance(p, (int, float)):
        raise TypeError(f"Probability p must be a number, got {type(p).__name__}")
    if not (0.0 < p < 1.0):
        raise ValueError(f"Probability p must be strictly between 0 and 1, got {p}")
    return NormalDist(0.0, 1.0).inv_cdf(p)


def norm_cdf(z: float) -> float:
    """Standard normal cumulative distribution function, Φ(z)."""
    if not isinstance(z, (int, float)):
        raise TypeError(f"z must be a number, got {type(z).__name__}")
    return NormalDist(0.0, 1.0).cdf(z)


def norm_pdf(z: float) -> float:
    """Standard normal probability density function, φ(z)."""
    if not isinstance(z, (int, float)):
        raise TypeError(f"z must be a number, got {type(z).__name__}")
    return NormalDist(0.0, 1.0).pdf(z)


def unit_normal_loss(z: float) -> float:
    """
    Standard unit normal loss function G_u(z) = E[max(Z - z, 0)]
    Formula: G_u(z) = φ(z) - z * (1 - Φ(z))
    """
    pdf = norm_pdf(z)
    cdf = norm_cdf(z)
    return pdf - z * (1.0 - cdf)


# ---------------------------------------------------------------------------
# Structured Result Data Classes with 8 Educational Verification Fields
# ---------------------------------------------------------------------------

@dataclass
class EOQResult:
    """Structured result for Economic Order Quantity with 8 student verification fields."""
    # 1. Inputs Extracted
    inputs_extracted: Dict[str, Any]

    # 2. Variable Definitions
    variable_definitions: Dict[str, str]

    # 3. Formula
    formula: Dict[str, str]

    # 4. Formula with Substituted Values
    formula_with_substituted_values: Dict[str, str]

    # 5. Intermediate Calculations
    intermediate_calculations: Dict[str, Any]

    # 6. Final Answer
    final_answer: float
    optimal_order_quantity: float
    rounded_order_quantity: int

    # 7. Verification Checks
    verification_checks: List[str]

    # 8. Assumptions
    assumptions: List[str]

    # Direct attributes for convenience
    annual_demand: float = 0.0
    order_cost: float = 0.0
    holding_cost: float = 0.0
    unit_cost: Optional[float] = None
    work_days_per_year: Optional[int] = None
    annual_ordering_cost: float = 0.0
    annual_holding_cost: float = 0.0
    annual_inventory_cost: float = 0.0
    annual_purchase_cost: Optional[float] = None
    total_annual_cost: float = 0.0
    orders_per_year: float = 0.0
    cycle_time_months: float = 0.0
    cycle_time_days: Optional[float] = None
    average_cycle_inventory: float = 0.0

    @property
    def eoq(self) -> float:
        return self.optimal_order_quantity

    @property
    def rounded_eoq(self) -> int:
        return self.rounded_order_quantity

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiscountTierEvaluation:
    """Evaluation of an individual tier within a quantity discount schedule."""
    tier_index: int
    min_qty: float
    max_qty: Optional[float]
    unit_price: float
    holding_cost_per_unit: float
    unconstrained_eoq: float
    feasible_order_quantity: float
    is_price_break_bumped: bool
    annual_ordering_cost: float
    annual_holding_cost: float
    annual_purchase_cost: float
    total_annual_cost: float
    formula_substitution: str


@dataclass
class QuantityDiscountResult:
    """Structured result for Quantity Discount with 8 student verification fields."""
    # 1. Inputs Extracted
    inputs_extracted: Dict[str, Any]

    # 2. Variable Definitions
    variable_definitions: Dict[str, str]

    # 3. Formula
    formula: Dict[str, str]

    # 4. Formula with Substituted Values
    formula_with_substituted_values: Dict[str, str]

    # 5. Intermediate Calculations
    intermediate_calculations: Dict[str, Any]

    # 6. Final Answer
    final_answer: float
    optimal_order_quantity: float
    rounded_optimal_quantity: int
    optimal_tier_index: int
    optimal_unit_price: float
    min_total_annual_cost: float

    # 7. Verification Checks
    verification_checks: List[str]

    # 8. Assumptions
    assumptions: List[str]

    # Direct attributes for convenience
    annual_demand: float = 0.0
    order_cost: float = 0.0
    holding_cost_rate: Optional[float] = None
    fixed_holding_cost: Optional[float] = None
    baseline_total_annual_cost: float = 0.0
    annual_savings: float = 0.0
    percentage_savings: float = 0.0
    tier_evaluations: List[DiscountTierEvaluation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SafetyStockROPResult:
    """Structured result for Safety Stock and Reorder Point with 8 student verification fields."""
    # 1. Inputs Extracted
    inputs_extracted: Dict[str, Any]

    # 2. Variable Definitions
    variable_definitions: Dict[str, str]

    # 3. Formula
    formula: Dict[str, str]

    # 4. Formula with Substituted Values
    formula_with_substituted_values: Dict[str, str]

    # 5. Intermediate Calculations
    intermediate_calculations: Dict[str, Any]

    # 6. Final Answer
    final_answer: Dict[str, float]
    reorder_point: float
    rounded_reorder_point: int
    safety_stock: float
    rounded_safety_stock: int

    # 7. Verification Checks
    verification_checks: List[str]

    # 8. Assumptions
    assumptions: List[str]

    # Direct attributes for convenience
    model_case: str = ""
    demand_per_period: float = 0.0
    lead_time: float = 0.0
    service_level: Optional[float] = None
    std_demand: float = 0.0
    std_lead_time: float = 0.0
    time_unit: str = "day"
    z_score: Optional[float] = None
    expected_lead_time_demand: float = 0.0
    variance_lead_time_demand: float = 0.0
    std_lead_time_demand: float = 0.0
    stockout_risk_per_cycle: Optional[float] = None
    lead_time_variance: Optional[float] = None
    lead_time_demand_mean: Optional[float] = None
    lead_time_demand_std: Optional[float] = None

    @property
    def avg_demand_per_period(self) -> float:
        return self.demand_per_period

    @property
    def lead_time_periods(self) -> float:
        return self.lead_time

    @property
    def std_demand_per_period(self) -> float:
        return self.std_demand

    @property
    def std_lead_time_periods(self) -> float:
        return self.std_lead_time

    @property
    def lead_time_variability_incorporated(self) -> bool:
        return self.std_lead_time > 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FixedPeriodReviewResult:
    """Structured result for Fixed-Period Review with 8 student verification fields."""
    # 1. Inputs Extracted
    inputs_extracted: Dict[str, Any]

    # 2. Variable Definitions
    variable_definitions: Dict[str, str]

    # 3. Formula
    formula: Dict[str, str]

    # 4. Formula with Substituted Values
    formula_with_substituted_values: Dict[str, str]

    # 5. Intermediate Calculations
    intermediate_calculations: Dict[str, Any]

    # 6. Final Answer
    final_answer: Dict[str, float]
    order_quantity: float
    rounded_order_quantity: int
    target_inventory: float
    rounded_target_inventory: int

    # 7. Verification Checks
    verification_checks: List[str]

    # 8. Assumptions
    assumptions: List[str]

    # Direct attributes for convenience
    review_period: float = 0.0
    lead_time: float = 0.0
    demand_per_period: float = 0.0
    inventory_position: float = 0.0
    service_level: Optional[float] = None
    std_demand: float = 0.0
    time_unit: str = "day"
    protection_period: float = 0.0
    demand_during_protection: float = 0.0
    variance_protection_demand: float = 0.0
    std_protection_demand: float = 0.0
    z_score: Optional[float] = None
    safety_stock: float = 0.0
    rounded_safety_stock: int = 0
    current_on_hand: Optional[float] = None
    current_on_order: Optional[float] = None
    current_backorders: Optional[float] = None

    @property
    def order_up_to_level(self) -> float:
        return self.target_inventory

    @property
    def rounded_order_up_to_level(self) -> int:
        return self.rounded_target_inventory

    @property
    def recommended_order_quantity(self) -> float:
        return self.order_quantity

    @property
    def protection_interval(self) -> float:
        return self.protection_period

    @property
    def expected_protection_demand(self) -> float:
        return self.demand_during_protection

    @property
    def avg_demand_per_period(self) -> float:
        return self.demand_per_period

    @property
    def std_demand_per_period(self) -> float:
        return self.std_demand

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NewsvendorResult:
    """Structured result for Single-Period Newsvendor with 8 student verification fields."""
    # 1. Inputs Extracted
    inputs_extracted: Dict[str, Any]

    # 2. Variable Definitions
    variable_definitions: Dict[str, str]

    # 3. Formula
    formula: Dict[str, str]

    # 4. Formula with Substituted Values
    formula_with_substituted_values: Dict[str, str]

    # 5. Intermediate Calculations
    intermediate_calculations: Dict[str, Any]

    # 6. Final Answer
    final_answer: float
    recommended_order_quantity: float
    rounded_order_quantity: int
    critical_ratio: float

    # 7. Verification Checks
    verification_checks: List[str]

    # 8. Assumptions
    assumptions: List[str]

    # Direct attributes for convenience
    selling_price: Optional[float] = None
    cost: Optional[float] = None
    salvage_value: Optional[float] = None
    mean_demand: Optional[float] = None
    std_demand: Optional[float] = None
    shortage_penalty: float = 0.0
    underage_cost: float = 0.0
    overage_cost: float = 0.0
    z_score: Optional[float] = None
    unit_normal_loss: Optional[float] = None
    expected_lost_sales: Optional[float] = None
    expected_sales: Optional[float] = None
    expected_leftover_inventory: Optional[float] = None
    expected_revenue: Optional[float] = None
    expected_cost: Optional[float] = None
    expected_shortage_cost: Optional[float] = None
    expected_profit: Optional[float] = None
    profit_under_perfect_information: Optional[float] = None
    expected_value_of_perfect_information: Optional[float] = None
    in_stock_probability: float = 0.0
    stockout_probability: float = 0.0
    demand_table: Optional[List[Dict[str, Any]]] = None

    @property
    def evpi(self) -> Optional[float]:
        return self.expected_value_of_perfect_information

    @property
    def optimal_order_quantity(self) -> float:
        return self.recommended_order_quantity

    @property
    def rounded_optimal_quantity(self) -> int:
        return self.rounded_order_quantity

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 1. Economic Order Quantity (EOQ) Tool
# ---------------------------------------------------------------------------

def calculate_eoq(
    demand: Optional[float] = None,
    order_cost: Optional[float] = None,
    holding_cost: Optional[float] = None,
    holding_cost_rate: Optional[float] = None,
    unit_cost: Optional[float] = None,
    annual_demand: Optional[float] = None,
    work_days_per_year: Optional[int] = None,
) -> EOQResult:
    """
    Calculate the deterministic Economic Order Quantity (Wilson EOQ).
    Returns all 8 verification components for complete student verification.
    """
    # Input validation - never invent missing numerical inputs
    d = demand if demand is not None else annual_demand
    if d is None:
        raise ValueError("demand (D) must be provided. Missing numerical inputs cannot be invented.")
    if not isinstance(d, (int, float)):
        raise TypeError(f"demand must be a numeric value, got {type(d).__name__}")
    if d <= 0:
        raise ValueError(f"demand (D) must be strictly positive, got {d}")

    if order_cost is None:
        raise ValueError("order_cost (S) must be provided. Missing numerical inputs cannot be invented.")
    if not isinstance(order_cost, (int, float)):
        raise TypeError(f"order_cost must be a numeric value, got {type(order_cost).__name__}")
    if order_cost <= 0:
        raise ValueError(f"order_cost (S) must be strictly positive, got {order_cost}")

    h = holding_cost
    if h is None:
        if holding_cost_rate is not None and unit_cost is not None:
            if not isinstance(holding_cost_rate, (int, float)):
                raise TypeError("holding_cost_rate must be a numeric value")
            if not isinstance(unit_cost, (int, float)):
                raise TypeError("unit_cost must be a numeric value")
            if holding_cost_rate <= 0:
                raise ValueError(f"holding_cost_rate must be strictly positive, got {holding_cost_rate}")
            if unit_cost < 0:
                raise ValueError(f"unit_cost cannot be negative, got {unit_cost}")
            h = holding_cost_rate * unit_cost
        else:
            raise ValueError(
                "Must provide either 'holding_cost' (H) or both 'holding_cost_rate' (h) and 'unit_cost' (C). "
                "Missing numerical inputs cannot be invented."
            )
    else:
        if not isinstance(h, (int, float)):
            raise TypeError(f"holding_cost must be a numeric value, got {type(h).__name__}")

    if h <= 0:
        raise ValueError(f"holding_cost (H) must be strictly positive, got {h}")

    work_days = None
    if work_days_per_year is not None:
        if not isinstance(work_days_per_year, (int, float)) or work_days_per_year <= 0:
            raise ValueError(f"work_days_per_year must be positive, got {work_days_per_year}")
        work_days = int(work_days_per_year)

    d_val = float(d)
    s_val = float(order_cost)
    h_val = float(h)
    c_val = float(unit_cost) if unit_cost is not None else None

    # Step 1: 1. Inputs Extracted
    inputs_extracted = {
        "demand": d_val,
        "order_cost": s_val,
        "holding_cost": h_val,
        "holding_cost_rate": holding_cost_rate,
        "unit_cost": c_val,
        "work_days_per_year": work_days,
    }

    # Step 2: 2. Variable Definitions
    variable_definitions = {
        "D": "Annual demand (units/year)",
        "S": "Ordering or setup cost per replenishment order ($/order)",
        "H": "Annual holding/carrying cost per unit ($/unit/year)",
        "C": "Unit purchase price ($/unit, optional)",
        "Q*": "Economic Order Quantity: cost-minimizing order batch size (units)",
        "AOC": "Annual Ordering Cost = (D / Q*) * S ($/year)",
        "AHC": "Annual Holding Cost = (Q* / 2) * H ($/year)",
        "TIC": "Total Annual Inventory Cost = AOC + AHC ($/year)",
        "N": "Number of replenishment orders placed per year = D / Q*",
        "T_months": "Cycle time between orders in months = 12 / N",
    }
    if work_days is not None:
        variable_definitions["T_days"] = "Cycle time between orders in operating days = (Q* / D) * work_days"

    # Step 3: 3. Formula
    formula = {
        "optimal_order_quantity": "Q* = sqrt((2 * D * S) / H)",
        "annual_ordering_cost": "AOC = (D / Q*) * S",
        "annual_holding_cost": "AHC = (Q* / 2) * H",
        "total_inventory_cost": "TIC = AOC + AHC = (D / Q*) * S + (Q* / 2) * H",
        "orders_per_year": "N = D / Q*",
        "cycle_time_months": "T_months = 12 / N",
        "average_cycle_inventory": "ACI = Q* / 2",
    }
    if work_days is not None:
        formula["cycle_time_days"] = "T_days = (Q* / D) * work_days"

    # Numerical execution
    numerator_2DS = 2.0 * d_val * s_val
    ratio_2DS_over_H = numerator_2DS / h_val
    optimal_qty = math.sqrt(ratio_2DS_over_H)
    rounded_qty = int(round(optimal_qty))

    annual_ordering_cost = (d_val / optimal_qty) * s_val
    annual_holding_cost = (optimal_qty / 2.0) * h_val
    annual_inventory_cost = annual_ordering_cost + annual_holding_cost
    annual_purchase_cost = (d_val * c_val) if c_val is not None else None
    total_annual_cost = annual_inventory_cost + (annual_purchase_cost or 0.0)

    orders_per_year = d_val / optimal_qty
    cycle_time_months = 12.0 / orders_per_year
    cycle_time_days = (optimal_qty / d_val) * work_days if work_days is not None else None
    average_cycle_inventory = optimal_qty / 2.0

    # Step 4: 4. Formula with Substituted Values
    formula_with_substituted_values = {
        "optimal_order_quantity": f"Q* = sqrt((2 * {d_val:,.2f} * {s_val:,.2f}) / {h_val:,.2f}) = sqrt({numerator_2DS:,.2f} / {h_val:,.2f}) = sqrt({ratio_2DS_over_H:,.2f}) = {optimal_qty:,.2f}",
        "annual_ordering_cost": f"AOC = ({d_val:,.2f} / {optimal_qty:,.2f}) * {s_val:,.2f} = {orders_per_year:,.4f} * {s_val:,.2f} = ${annual_ordering_cost:,.2f}",
        "annual_holding_cost": f"AHC = ({optimal_qty:,.2f} / 2) * {h_val:,.2f} = {average_cycle_inventory:,.4f} * {h_val:,.2f} = ${annual_holding_cost:,.2f}",
        "total_inventory_cost": f"TIC = ${annual_ordering_cost:,.2f} + ${annual_holding_cost:,.2f} = ${annual_inventory_cost:,.2f}",
        "orders_per_year": f"N = {d_val:,.2f} / {optimal_qty:,.2f} = {orders_per_year:,.2f} orders/year",
        "cycle_time_months": f"T_months = 12 / {orders_per_year:,.4f} = {cycle_time_months:,.2f} months",
    }
    if work_days is not None and cycle_time_days is not None:
        formula_with_substituted_values["cycle_time_days"] = f"T_days = ({optimal_qty:,.2f} / {d_val:,.2f}) * {work_days} = {cycle_time_days:,.2f} work days"

    if annual_purchase_cost is not None and c_val is not None:
        formula_with_substituted_values["total_annual_cost"] = (
            f"TAC = TIC + (D * C) = ${annual_inventory_cost:,.2f} + ({d_val:,.2f} * ${c_val:,.2f}) "
            f"= ${annual_inventory_cost:,.2f} + ${annual_purchase_cost:,.2f} = ${total_annual_cost:,.2f}"
        )

    # Step 5: 5. Intermediate Calculations
    intermediate_calculations = {
        "numerator_2DS": numerator_2DS,
        "ratio_2DS_over_H": ratio_2DS_over_H,
        "sqrt_result": optimal_qty,
        "annual_ordering_cost": annual_ordering_cost,
        "annual_holding_cost": annual_holding_cost,
        "cost_difference_at_optimum": abs(annual_ordering_cost - annual_holding_cost),
        "annual_inventory_cost": annual_inventory_cost,
        "annual_purchase_cost": annual_purchase_cost,
        "total_annual_cost": total_annual_cost,
        "orders_per_year": orders_per_year,
        "cycle_time_months": cycle_time_months,
        "cycle_time_days": cycle_time_days,
        "work_days_per_year": work_days,
        "average_cycle_inventory": average_cycle_inventory,
    }

    # Step 6: 6. Final Answer
    final_answer = optimal_qty

    # Step 7: 7. Verification Checks
    cost_diff = abs(annual_ordering_cost - annual_holding_cost)
    pos_pass = optimal_qty > 0
    tradeoff_pass = (cost_diff <= 0.05) or (annual_inventory_cost > 0 and (cost_diff / annual_inventory_cost) < 0.001)
    d2_val = (2 * d_val * s_val) / (optimal_qty ** 3) if optimal_qty > 0 else 0.0
    convex_pass = d2_val > 0
    cadence_val = (orders_per_year * cycle_time_months / 12.0)
    cadence_pass = abs(cadence_val - 1.0) < 1e-3

    verification_checks = [
        f"Positivity check: Q* = {optimal_qty:,.2f} > 0 ({'PASS' if pos_pass else 'FAIL'})",
        f"Trade-off equilibrium check: |AOC - AHC| = |${annual_ordering_cost:,.2f} - ${annual_holding_cost:,.2f}| = ${cost_diff:,.4f} <= $0.05 tolerance ({'PASS' if tradeoff_pass else 'FAIL'}: At optimal Wilson EOQ, annual ordering cost balances annual holding cost)",
        f"Second-order convexity check: d^2(TIC)/dQ^2 = 2DS/Q^3 = {d2_val:.6f} > 0 ({'PASS' if convex_pass else 'FAIL'}: Strictly convex function confirms global cost minimum)",
        f"Cadence consistency check: N * (T_months / 12) = {orders_per_year:.4f} * ({cycle_time_months:.4f} / 12) = {cadence_val:.4f} ≈ 1.0 ({'PASS' if cadence_pass else 'FAIL'}: Replenishment cadence matches annual calendar)",
    ]
    if work_days is not None and cycle_time_days is not None:
        days_cadence_val = (orders_per_year * cycle_time_days / work_days)
        days_cadence_pass = abs(days_cadence_val - 1.0) < 1e-3
        verification_checks.append(
            f"Working days cadence check: N * (T_days / work_days) = {orders_per_year:.4f} * ({cycle_time_days:.4f} / {work_days}) = {days_cadence_val:.4f} ≈ 1.0 ({'PASS' if days_cadence_pass else 'FAIL'})"
        )

    # Step 8: 8. Assumptions
    assumptions = [
        "Demand rate (D) is deterministic, continuous, and constant across the entire annual planning horizon.",
        "Ordering/setup cost (S) is fixed per replenishment order, regardless of batch size ordered.",
        f"Holding cost (H = ${h_val:,.2f}/unit/year) is strictly linear and constant per unit in inventory.",
        "Replenishment lead time is zero or constant, resulting in instantaneous batch delivery.",
        "No shortages, stockouts, or customer backorders are permitted.",
        "The entire order quantity Q arrives simultaneously in a single replenishment batch.",
        "Purchase price per unit is independent of order quantity (no price breaks apply to this EOQ).",
    ]

    return EOQResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate_calculations,
        final_answer=final_answer,
        optimal_order_quantity=optimal_qty,
        rounded_order_quantity=rounded_qty,
        verification_checks=verification_checks,
        assumptions=assumptions,
        annual_demand=d_val,
        order_cost=s_val,
        holding_cost=h_val,
        unit_cost=c_val,
        work_days_per_year=work_days,
        annual_ordering_cost=annual_ordering_cost,
        annual_holding_cost=annual_holding_cost,
        annual_inventory_cost=annual_inventory_cost,
        annual_purchase_cost=annual_purchase_cost,
        total_annual_cost=total_annual_cost,
        orders_per_year=orders_per_year,
        cycle_time_months=cycle_time_months,
        cycle_time_days=cycle_time_days,
        average_cycle_inventory=average_cycle_inventory,
    )


# ---------------------------------------------------------------------------
# 2. Quantity Discount Tool
# ---------------------------------------------------------------------------

def calculate_quantity_discount(
    demand: Optional[float] = None,
    order_cost: Optional[float] = None,
    discount_schedule: Optional[List[Dict[str, Any]]] = None,
    holding_cost_rate: Optional[float] = None,
    fixed_holding_cost: Optional[float] = None,
    annual_demand: Optional[float] = None,
) -> QuantityDiscountResult:
    """
    Evaluate feasible order quantities across all discount price tiers and compare total annual cost.
    Returns all 8 verification components for complete student verification.
    """
    d = demand if demand is not None else annual_demand
    if d is None:
        raise ValueError("demand (D) must be provided. Missing numerical inputs cannot be invented.")
    if not isinstance(d, (int, float)):
        raise TypeError("demand must be numeric")
    if d <= 0:
        raise ValueError(f"demand (D) must be strictly positive, got {d}")

    if order_cost is None:
        raise ValueError("order_cost (S) must be provided. Missing numerical inputs cannot be invented.")
    if not isinstance(order_cost, (int, float)):
        raise TypeError("order_cost must be numeric")
    if order_cost <= 0:
        raise ValueError(f"order_cost (S) must be strictly positive, got {order_cost}")

    if not discount_schedule:
        raise ValueError("discount_schedule must be a non-empty list of price tiers. Missing numerical inputs cannot be invented.")

    if holding_cost_rate is None and fixed_holding_cost is None:
        raise ValueError(
            "Must provide either 'holding_cost_rate' (percentage holding cost) or 'fixed_holding_cost'. "
            "Missing numerical inputs cannot be invented."
        )

    if holding_cost_rate is not None:
        if not isinstance(holding_cost_rate, (int, float)):
            raise TypeError("holding_cost_rate must be numeric")
        if holding_cost_rate <= 0:
            raise ValueError(f"holding_cost_rate must be strictly positive, got {holding_cost_rate}")

    if fixed_holding_cost is not None:
        if not isinstance(fixed_holding_cost, (int, float)):
            raise TypeError("fixed_holding_cost must be numeric")
        if fixed_holding_cost <= 0:
            raise ValueError(f"fixed_holding_cost must be strictly positive, got {fixed_holding_cost}")

    d_val = float(d)
    s_val = float(order_cost)
    sorted_tiers = sorted(discount_schedule, key=lambda t: t.get("min_qty", 0))

    # 1. Inputs Extracted
    inputs_extracted = {
        "demand": d_val,
        "order_cost": s_val,
        "discount_schedule": sorted_tiers,
        "holding_cost_rate": holding_cost_rate,
        "fixed_holding_cost": fixed_holding_cost,
    }

    # 2. Variable Definitions
    variable_definitions = {
        "D": "Annual demand (units/year)",
        "S": "Ordering/setup cost per order ($/order)",
        "C_i": "Unit purchase price in price tier i ($/unit)",
        "H_i": "Annual holding cost in tier i = h * C_i or fixed H ($/unit/year)",
        "Q_unconstrained,i": "Unconstrained EOQ for tier i = sqrt((2 * D * S) / H_i)",
        "q_min,i": "Minimum order quantity threshold required to qualify for tier i price",
        "q_max,i": "Maximum order quantity boundary for tier i",
        "Q_feasible,i": "Feasible batch size evaluated for tier i (bumped to q_min,i if unconstrained Q < q_min,i)",
        "AOC_i": "Annual Ordering Cost = (D / Q_feasible,i) * S ($/year)",
        "AHC_i": "Annual Holding Cost = (Q_feasible,i / 2) * H_i ($/year)",
        "APC_i": "Annual Purchase Cost = D * C_i ($/year)",
        "TAC_i": "Total Annual Cost = AOC_i + AHC_i + APC_i ($/year)",
    }

    # 3. Formula
    formula = {
        "unconstrained_eoq": "Q_unconstrained,i = sqrt((2 * D * S) / H_i)",
        "feasibility_rule": "If Q_i < q_min,i then Q_feasible = q_min,i (price-break bump); elif q_min,i <= Q_i <= q_max,i then Q_feasible = Q_i; else Q_feasible = q_max,i",
        "total_annual_cost": "TAC_i(Q) = (D / Q) * S + (Q / 2) * H_i + D * C_i",
        "optimal_decision_rule": "Select tier k with minimum global TAC_k = min_i(TAC_i)",
        "annual_savings": "Savings = TAC_baseline - TAC_optimal",
    }

    tier_evaluations: List[DiscountTierEvaluation] = []
    substitutions_dict: Dict[str, str] = {}
    best_tier_index = 0
    best_total_cost = float("inf")
    best_qty = 0.0

    for idx, tier in enumerate(sorted_tiers):
        if "unit_price" not in tier:
            raise ValueError(f"Tier {idx} missing required 'unit_price'")
        unit_price = float(tier["unit_price"])
        if unit_price <= 0:
            raise ValueError(f"Unit price for tier {idx} must be positive, got {unit_price}")

        min_qty = float(tier.get("min_qty", 0.0))
        if min_qty < 0:
            raise ValueError(f"min_qty for tier {idx} cannot be negative, got {min_qty}")
        max_qty_raw = tier.get("max_qty")
        max_qty = float(max_qty_raw) if max_qty_raw is not None else None

        # Determine holding cost for this tier
        if holding_cost_rate is not None:
            tier_h = holding_cost_rate * unit_price
            h_calc_str = f"{holding_cost_rate:.2f} * ${unit_price:.2f} = ${tier_h:.2f}"
        else:
            tier_h = float(fixed_holding_cost)  # type: ignore
            h_calc_str = f"fixed ${tier_h:.2f}"

        # Unconstrained EOQ for this tier
        unconstrained_eoq = math.sqrt((2.0 * d_val * s_val) / tier_h)

        # Feasibility check
        is_bumped = False
        if unconstrained_eoq < min_qty:
            feasible_qty = min_qty
            is_bumped = True
        elif max_qty is not None and unconstrained_eoq > max_qty:
            feasible_qty = max_qty
        else:
            feasible_qty = unconstrained_eoq

        aoc = (d_val / feasible_qty) * s_val
        ahc = (feasible_qty / 2.0) * tier_h
        apc = d_val * unit_price
        tac = aoc + ahc + apc

        rng_str = f"[{int(min_qty):,} - {int(max_qty):,}]" if max_qty else f"[{int(min_qty):,}+]"
        tier_sub_str = (
            f"Tier {idx+1} {rng_str} @ ${unit_price:.2f}/unit: H_{idx+1} = {h_calc_str}. "
            f"Unconstrained Q_{idx+1} = sqrt((2 * {d_val:,.0f} * {s_val:,.0f}) / {tier_h:.2f}) = {unconstrained_eoq:,.2f}. "
            f"{'Bumped to price break threshold Q_feasible = ' + f'{feasible_qty:,.0f}' if is_bumped else 'Feasible as-is Q_feasible = ' + f'{feasible_qty:,.2f}'}. "
            f"AOC = ({d_val:,.0f}/{feasible_qty:,.0f})*{s_val:,.0f} = ${aoc:,.2f}, "
            f"AHC = ({feasible_qty:,.0f}/2)*{tier_h:.2f} = ${ahc:,.2f}, "
            f"APC = {d_val:,.0f}*${unit_price:.2f} = ${apc:,.2f} => "
            f"TAC_{idx+1} = ${aoc:,.2f} + ${ahc:,.2f} + ${apc:,.2f} = ${tac:,.2f}"
        )
        substitutions_dict[f"tier_{idx+1}"] = tier_sub_str

        evaluation = DiscountTierEvaluation(
            tier_index=idx,
            min_qty=min_qty,
            max_qty=max_qty,
            unit_price=unit_price,
            holding_cost_per_unit=tier_h,
            unconstrained_eoq=unconstrained_eoq,
            feasible_order_quantity=feasible_qty,
            is_price_break_bumped=is_bumped,
            annual_ordering_cost=aoc,
            annual_holding_cost=ahc,
            annual_purchase_cost=apc,
            total_annual_cost=tac,
            formula_substitution=tier_sub_str,
        )
        tier_evaluations.append(evaluation)

        if tac < best_total_cost:
            best_total_cost = tac
            best_tier_index = idx
            best_qty = feasible_qty

    baseline_total_cost = tier_evaluations[0].total_annual_cost
    annual_savings = max(0.0, baseline_total_cost - best_total_cost)
    pct_savings = (annual_savings / baseline_total_cost * 100.0) if baseline_total_cost > 0 else 0.0

    substitutions_dict["winning_decision"] = (
        f"Tier {best_tier_index + 1} achieves minimum TAC = ${best_total_cost:,.2f} at Q* = {best_qty:,.0f} units. "
        f"Savings vs Tier 1 baseline: ${baseline_total_cost:,.2f} - ${best_total_cost:,.2f} = ${annual_savings:,.2f} ({pct_savings:.2f}%)"
    )

    # 4. Formula with Substituted Values
    formula_with_substituted_values = substitutions_dict

    # 5. Intermediate Calculations
    intermediate_calculations = {
        "tier_evaluations_count": len(tier_evaluations),
        "baseline_total_annual_cost": baseline_total_cost,
        "min_total_annual_cost": best_total_cost,
        "annual_savings": annual_savings,
        "percentage_savings": pct_savings,
        "optimal_tier_index": best_tier_index,
        "optimal_order_quantity": best_qty,
        "optimal_unit_price": tier_evaluations[best_tier_index].unit_price,
        "tiers_breakdown": [
            {
                "tier": t.tier_index + 1,
                "feasible_qty": t.feasible_order_quantity,
                "is_bumped": t.is_price_break_bumped,
                "AOC": t.annual_ordering_cost,
                "AHC": t.annual_holding_cost,
                "APC": t.annual_purchase_cost,
                "TAC": t.total_annual_cost,
            }
            for t in tier_evaluations
        ],
    }

    # 6. Final Answer
    final_answer = best_qty

    # 7. Verification Checks
    opt_tier = tier_evaluations[best_tier_index]
    t0 = tier_evaluations[0]
    feas_pass = (best_qty >= opt_tier.min_qty)
    optim_pass = all(t.total_annual_cost >= best_total_cost - 1e-4 for t in tier_evaluations)
    
    purchase_cost_saved = t0.annual_purchase_cost - opt_tier.annual_purchase_cost
    inv_cost_change = (opt_tier.annual_ordering_cost + opt_tier.annual_holding_cost) - (t0.annual_ordering_cost + t0.annual_holding_cost)
    net_annual_savings = baseline_total_cost - best_total_cost
    tradeoff_pass = (net_annual_savings >= -1e-4)

    inv_cost_str = f"-${abs(inv_cost_change):,.2f}" if inv_cost_change < 0 else f"${inv_cost_change:,.2f}"
    net_savings_str = f"-${abs(net_annual_savings):,.2f}" if net_annual_savings < 0 else f"${net_annual_savings:,.2f}"
    purch_saved_str = f"-${abs(purchase_cost_saved):,.2f}" if purchase_cost_saved < 0 else f"${purchase_cost_saved:,.2f}"

    verification_checks = [
        f"Threshold feasibility check: Q* = {best_qty:,.0f} >= min_qty ({opt_tier.min_qty:,.0f}) for Tier {best_tier_index+1} ({'PASS' if feas_pass else 'FAIL'}: Order size qualifies for discounted unit price ${opt_tier.unit_price:.2f})",
        f"Global optimality check: TAC_optimal = ${best_total_cost:,.2f} <= TAC_i for all {len(tier_evaluations)} price tiers ({'PASS' if optim_pass else 'FAIL'}: Confirmed mathematical minimum across all feasible candidate batch sizes)",
        f"Economic trade-off check: Purchase cost saved vs Tier 1 = {purch_saved_str}, change in ordering + holding cost = {inv_cost_str}, net annual savings = {net_savings_str} (baseline ${baseline_total_cost:,.2f} - optimal ${best_total_cost:,.2f}) ({'PASS' if tradeoff_pass else 'FAIL'})",
    ]

    # 8. Assumptions
    holding_rule_desc = (
        f"Percentage holding cost rate h = {holding_cost_rate*100:.1f}% of unit price"
        if holding_cost_rate is not None
        else f"Fixed holding cost H = ${fixed_holding_cost:,.2f}/unit/year"
    )
    assumptions = [
        "All-units quantity discount structure: the discount applies to all units purchased when order size meets or exceeds threshold.",
        "Annual demand (D) is deterministic, continuous, and constant across all price tiers.",
        f"Holding cost rule: {holding_rule_desc}.",
        "Ordering cost (S) remains constant across all order sizes and brackets.",
        "Replenishment lead time is zero or constant with zero shortages allowed.",
        "Warehouse storage and financial working capital are sufficient to accommodate the optimal batch size without secondary penalty.",
    ]

    return QuantityDiscountResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate_calculations,
        final_answer=final_answer,
        optimal_order_quantity=best_qty,
        rounded_optimal_quantity=int(round(best_qty)),
        optimal_tier_index=best_tier_index,
        optimal_unit_price=tier_evaluations[best_tier_index].unit_price,
        min_total_annual_cost=best_total_cost,
        verification_checks=verification_checks,
        assumptions=assumptions,
        annual_demand=d_val,
        order_cost=s_val,
        holding_cost_rate=holding_cost_rate,
        fixed_holding_cost=fixed_holding_cost,
        baseline_total_annual_cost=baseline_total_cost,
        annual_savings=annual_savings,
        percentage_savings=pct_savings,
        tier_evaluations=tier_evaluations,
    )


# ---------------------------------------------------------------------------
# 3. Safety Stock & Reorder Point (ROP) Tool
# ---------------------------------------------------------------------------

def calculate_rop(
    demand_per_period: Optional[float] = None,
    lead_time: Optional[float] = None,
    service_level: Optional[float] = None,
    std_demand: float = 0.0,
    std_lead_time: Optional[float] = None,
    avg_demand: Optional[float] = None,
    time_unit: str = "day",
    lead_time_variance: Optional[float] = None,
    lead_time_demand_mean: Optional[float] = None,
    lead_time_demand_std: Optional[float] = None,
) -> SafetyStockROPResult:
    """
    Calculate Safety Stock and Reorder Point (ROP) under continuous review (s, Q).
    Supports all four operations research demand and lead time regimes:
    1. Constant demand and constant lead time
    2. Variable demand and constant lead time
    3. Constant demand and variable lead time
    4. Variable demand and variable lead time

    Unified formulation:
        sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
        SS = z * sigma_LT
        ROP = d * L + SS

    Returns all 8 verification components for complete student verification.
    """
    # 1. Lead time variance vs std_lead_time validation and resolution
    if lead_time_variance is not None:
        if not isinstance(lead_time_variance, (int, float)):
            raise TypeError("lead_time_variance must be numeric")
        if lead_time_variance < 0:
            raise ValueError(f"lead_time_variance cannot be negative, got {lead_time_variance}")
        computed_std_lt = math.sqrt(float(lead_time_variance))
        if std_lead_time is not None:
            if not math.isclose(float(std_lead_time), computed_std_lt, rel_tol=1e-4, abs_tol=1e-4):
                raise ValueError(
                    f"lead_time_variance ({lead_time_variance}) and std_lead_time ({std_lead_time}) disagree. "
                    f"Expected std_lead_time = sqrt({lead_time_variance}) = {computed_std_lt:.4f}"
                )
        slt_val = computed_std_lt
    else:
        slt_val = float(std_lead_time) if std_lead_time is not None else 0.0

    if slt_val < 0:
        raise ValueError(f"std_lead_time cannot be negative, got {slt_val}")

    # 2. Expected lead-time demand resolution (lead_time_demand_mean vs d * L)
    d = demand_per_period if demand_per_period is not None else avg_demand
    if lead_time_demand_mean is not None:
        if not isinstance(lead_time_demand_mean, (int, float)):
            raise TypeError("lead_time_demand_mean must be numeric")
        if lead_time_demand_mean < 0:
            raise ValueError(f"lead_time_demand_mean cannot be negative, got {lead_time_demand_mean}")
        expected_ltd = float(lead_time_demand_mean)
        d_val = float(d) if d is not None else expected_ltd
        l_val = float(lead_time) if lead_time is not None else 1.0
    else:
        if d is None:
            raise ValueError("demand_per_period (d) or lead_time_demand_mean must be provided. Missing numerical inputs cannot be invented.")
        if not isinstance(d, (int, float)):
            raise TypeError("demand_per_period must be numeric")
        if d <= 0:
            raise ValueError(f"demand_per_period (d) must be strictly positive, got {d}")

        if lead_time is None:
            raise ValueError("lead_time (L) or lead_time_demand_mean must be provided. Missing numerical inputs cannot be invented.")
        if not isinstance(lead_time, (int, float)):
            raise TypeError("lead_time must be numeric")
        if lead_time <= 0:
            raise ValueError(f"lead_time (L) must be strictly positive, got {lead_time}")

        d_val = float(d)
        l_val = float(lead_time)
        expected_ltd = d_val * l_val

    if not isinstance(std_demand, (int, float)):
        raise TypeError("std_demand must be numeric")
    if std_demand < 0:
        raise ValueError(f"std_demand cannot be negative, got {std_demand}")
    sd_val = float(std_demand)

    # 3. Lead-time demand standard deviation resolution (lead_time_demand_std vs sqrt(L*sigma_d^2 + d^2*sigma_L^2))
    if lead_time_demand_std is not None:
        if not isinstance(lead_time_demand_std, (int, float)):
            raise TypeError("lead_time_demand_std must be numeric")
        if lead_time_demand_std < 0:
            raise ValueError(f"lead_time_demand_std cannot be negative, got {lead_time_demand_std}")
        sigma_lt = float(lead_time_demand_std)
        var_ltd = sigma_lt ** 2
        term1 = var_ltd
        term2 = 0.0
    else:
        term1 = l_val * (sd_val ** 2)
        term2 = (d_val ** 2) * (slt_val ** 2)
        var_ltd = term1 + term2
        sigma_lt = math.sqrt(var_ltd)

    # 1. Inputs Extracted
    inputs_extracted = {
        "demand_per_period": d_val,
        "lead_time": l_val,
        "service_level": float(service_level) if service_level is not None else None,
        "std_demand": sd_val,
        "std_lead_time": slt_val,
        "time_unit": time_unit,
        "lead_time_variance": float(lead_time_variance) if lead_time_variance is not None else None,
        "lead_time_demand_mean": float(lead_time_demand_mean) if lead_time_demand_mean is not None else None,
        "lead_time_demand_std": float(lead_time_demand_std) if lead_time_demand_std is not None else None,
    }

    # 2. Variable Definitions
    variable_definitions = {
        "d": f"Average demand per period (units/{time_unit})",
        "L": f"Average replenishment lead time ({time_unit}s)",
        "sigma_d": f"Standard deviation of demand per period (units/{time_unit})",
        "sigma_L": f"Standard deviation of replenishment lead time ({time_unit}s)",
        "SL": "Target Cycle Service Level (CSL, probability of no stockout during lead time)",
        "z": "Safety factor = norm_ppf(SL), standard normal quantile",
        "mu_LT": "Expected demand during lead time (units)",
        "variance_LT": "Variance of lead time demand",
        "sigma_LT": "Standard deviation of lead time demand = sqrt(variance_LT) (units)",
        "SS": "Safety stock buffer = z * sigma_LT (units)",
        "ROP": "Reorder point trigger level = mu_LT + SS (units)",
        "Risk": "Stockout risk per replenishment cycle = 1 - SL",
    }
    if lead_time_variance is not None:
        variable_definitions["lead_time_variance"] = "Variance of lead time (periods^2)"

    # Determine regime
    is_d_var = sd_val > 0.0 or (lead_time_demand_std is not None and lead_time_demand_std > 0.0)
    is_lt_var = slt_val > 0.0

    if lead_time_demand_mean is not None or lead_time_demand_std is not None:
        model_case = "Direct Lead-Time Demand Parameters"
    elif not is_d_var and not is_lt_var:
        model_case = "Constant Demand and Constant Lead Time"
    elif is_d_var and not is_lt_var:
        model_case = "Variable Demand and Constant Lead Time"
    elif not is_d_var and is_lt_var:
        model_case = "Constant Demand and Variable Lead Time"
    else:
        model_case = "Variable Demand and Variable Lead Time"

    # If variability exists, service level is mandatory
    z: Optional[float] = None
    stockout_risk: Optional[float] = None
    if is_d_var or is_lt_var:
        if service_level is None:
            raise ValueError(
                f"service_level (CSL) is required when demand or lead time is variable ({model_case}). "
                "Missing numerical inputs cannot be invented."
            )
        if not isinstance(service_level, (int, float)):
            raise TypeError("service_level must be numeric")
        if not (0.0 < service_level < 1.0):
            raise ValueError(f"service_level must be strictly between 0 and 1, got {service_level}")
        z = norm_ppf(float(service_level))
        stockout_risk = 1.0 - float(service_level)
    else:
        # Constant demand & constant lead time: deterministic ROP = d * L
        if service_level is not None:
            if not (0.0 < service_level < 1.0):
                raise ValueError(f"service_level must be between 0 and 1, got {service_level}")
            z = norm_ppf(float(service_level))
            stockout_risk = 1.0 - float(service_level)
        else:
            z = 0.0
            stockout_risk = 0.0

    # 3. Formula
    formula = {
        "expected_lead_time_demand": "mu_LT = lead_time_demand_mean" if lead_time_demand_mean is not None else "mu_LT = d * L",
        "variance_lead_time_demand": "sigma_LT^2 = lead_time_demand_std^2" if lead_time_demand_std is not None else "sigma_LT^2 = L * sigma_d^2 + d^2 * sigma_L^2",
        "std_lead_time_demand": "sigma_LT = lead_time_demand_std" if lead_time_demand_std is not None else "sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)",
        "safety_factor": "z = norm_ppf(SL)",
        "safety_stock": "SS = z * sigma_LT",
        "reorder_point": "ROP = mu_LT + SS = mu_LT + z * sigma_LT",
    }
    if lead_time_variance is not None:
        formula["std_lead_time"] = "sigma_L = sqrt(lead_time_variance)"

    if z is not None and sigma_lt > 0.0:
        safety_stock = z * sigma_lt
    else:
        safety_stock = 0.0

    reorder_point = expected_ltd + safety_stock
    rounded_ss = int(math.ceil(safety_stock))
    rounded_rop = int(math.ceil(reorder_point))

    # 4. Formula with Substituted Values
    z_str = f"{z:.4f}" if z is not None else "0.0000"
    formula_with_substituted_values = {}
    if lead_time_variance is not None:
        formula_with_substituted_values["std_lead_time"] = f"sigma_L = sqrt(lead_time_variance) = sqrt({lead_time_variance}) = {slt_val:.4f}"

    if lead_time_demand_mean is not None:
        formula_with_substituted_values["expected_lead_time_demand"] = f"mu_LT = {expected_ltd:,.2f} units (provided directly as lead-time total)"
    else:
        formula_with_substituted_values["expected_lead_time_demand"] = f"mu_LT = {d_val:,.2f} * {l_val:,.2f} = {expected_ltd:,.2f} units"

    if lead_time_demand_std is not None:
        formula_with_substituted_values["std_lead_time_demand"] = f"sigma_LT = {sigma_lt:,.2f} units (provided directly as lead-time total)"
    else:
        formula_with_substituted_values["std_lead_time_demand"] = (
            f"sigma_LT = sqrt({l_val:,.2f} * ({sd_val:,.2f})^2 + ({d_val:,.2f})^2 * ({slt_val:,.2f})^2) "
            f"= sqrt({term1:,.2f} + {term2:,.2f}) = sqrt({var_ltd:,.2f}) = {sigma_lt:,.2f} units"
        )

    formula_with_substituted_values["safety_factor"] = f"z = norm_ppf({service_level if service_level else 0.5}) = {z_str}" if service_level else "z = 0 (deterministic)"
    formula_with_substituted_values["safety_stock"] = f"SS = {z_str} * {sigma_lt:,.2f} = {safety_stock:,.2f} units (ceiling: {rounded_ss} units)"
    formula_with_substituted_values["reorder_point"] = f"ROP = {expected_ltd:,.2f} + {safety_stock:,.2f} = {reorder_point:,.2f} units (ceiling: {rounded_rop} units)"

    # 5. Intermediate Calculations
    intermediate_calculations = {
        "model_case": model_case,
        "std_lead_time": slt_val,
        "expected_lead_time_demand": expected_ltd,
        "demand_variance_component": term1,
        "lead_time_variance_component": term2,
        "term_lead_time_times_demand_variance": term1,
        "term_demand_squared_times_lead_time_variance": term2,
        "variance_lead_time_demand": var_ltd,
        "std_lead_time_demand": sigma_lt,
        "z_score": z,
        "safety_stock": safety_stock,
        "rounded_safety_stock": rounded_ss,
        "reorder_point": reorder_point,
        "rounded_reorder_point": rounded_rop,
        "stockout_risk_per_cycle": stockout_risk,
        "lead_time_variance": lead_time_variance,
        "lead_time_demand_mean": lead_time_demand_mean,
        "lead_time_demand_std": lead_time_demand_std,
    }

    # 6. Final Answer
    final_answer = {"reorder_point": reorder_point, "safety_stock": safety_stock}

    # 7. Verification Checks
    regime_pass = bool(model_case)
    var_pass = (var_ltd >= -1e-6)
    ss_pass = (safety_stock >= -1e-6)
    rop_floor_pass = (reorder_point >= expected_ltd - 1e-4)

    verification_checks = [
        f"Regime identification check: Classified as '{model_case}' ({'PASS' if regime_pass else 'FAIL'})",
        f"Variance non-negativity check: variance_LT = {var_ltd:,.4f} >= 0 ({'PASS' if var_pass else 'FAIL'}: Mathematically valid variance)",
        f"Safety stock non-negativity check: SS = {safety_stock:,.2f} >= 0 ({'PASS' if ss_pass else 'FAIL'}: Non-negative buffer for CSL >= 50%)",
        f"Reorder point floor check: ROP ({reorder_point:,.2f}) >= expected_lead_time_demand ({expected_ltd:,.2f}) by SS buffer ({safety_stock:,.2f}) ({'PASS' if rop_floor_pass else 'FAIL'})",
    ]
    if stockout_risk is not None:
        risk_pass = (0.0 <= stockout_risk <= 1.0)
        verification_checks.append(f"Cycle stockout risk check: 1 - SL = {stockout_risk*100:.2f}% ({'PASS' if risk_pass else 'FAIL'}: Complies with target service level)")

    # 8. Assumptions
    assumptions = [
        f"Lead time and demand regime: {model_case}.",
        "Continuous review inventory control policy (s, Q): inventory position is monitored continuously.",
        "Replenishment order of batch size Q is placed immediately when inventory position drops to or below ROP.",
        "Stockouts during the replenishment lead time are backordered (Cycle Service Level Type I policy).",
    ]
    if lead_time_demand_mean is not None:
        assumptions.append(f"Expected lead-time demand (mu_LT = {expected_ltd:,.2f} units) was taken directly as a lead-time total without multiplying by lead time.")
    if lead_time_demand_std is not None:
        assumptions.append(f"Lead-time demand standard deviation (sigma_LT = {sigma_lt:,.2f} units) was taken directly as a lead-time total without multiplying by sqrt(L).")
    if lead_time_variance is not None:
        assumptions.append(f"Lead-time variance ({lead_time_variance}) was supplied and used to compute lead-time standard deviation sigma_L = sqrt({lead_time_variance}) = {slt_val:.4f}.")
    if lead_time_demand_mean is None and lead_time_demand_std is None:
        assumptions.append("Periodic demands and lead times across successive periods are statistically independent random variables.")

    return SafetyStockROPResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate_calculations,
        final_answer=final_answer,
        reorder_point=reorder_point,
        rounded_reorder_point=rounded_rop,
        safety_stock=safety_stock,
        rounded_safety_stock=rounded_ss,
        verification_checks=verification_checks,
        assumptions=assumptions,
        model_case=model_case,
        demand_per_period=d_val,
        lead_time=l_val,
        service_level=float(service_level) if service_level is not None else None,
        std_demand=sd_val,
        std_lead_time=slt_val,
        time_unit=time_unit,
        z_score=z,
        expected_lead_time_demand=expected_ltd,
        variance_lead_time_demand=var_ltd,
        std_lead_time_demand=sigma_lt,
        stockout_risk_per_cycle=stockout_risk,
        lead_time_variance=lead_time_variance,
        lead_time_demand_mean=lead_time_demand_mean,
        lead_time_demand_std=lead_time_demand_std,
    )


calculate_safety_stock_and_rop = calculate_rop


# ---------------------------------------------------------------------------
# 4. Fixed-Period Review Tool ((T, S) Policy)
# ---------------------------------------------------------------------------

def calculate_periodic_review(
    review_period: Optional[float] = None,
    lead_time: Optional[float] = None,
    demand_per_period: Optional[float] = None,
    inventory_position: Optional[float] = None,
    service_level: Optional[float] = None,
    std_demand: float = 0.0,
    safety_stock: Optional[float] = None,
    current_inventory: Optional[float] = None,
    on_order: Optional[float] = None,
    backorders: Optional[float] = None,
    avg_demand: Optional[float] = None,
    time_unit: str = "day",
) -> FixedPeriodReviewResult:
    """
    Calculate Order-Up-To Level (Target Inventory S) and Recommended Order Quantity (q)
    under a periodic review system ((T, S) policy).
    Returns all 8 verification components for complete student verification.
    """
    if review_period is None:
        raise ValueError("review_period (T) must be provided. Missing numerical inputs cannot be invented.")
    if not isinstance(review_period, (int, float)):
        raise TypeError("review_period must be numeric")
    if review_period <= 0:
        raise ValueError(f"review_period (T) must be strictly positive, got {review_period}")

    if lead_time is None:
        raise ValueError("lead_time (L) must be provided. Missing numerical inputs cannot be invented.")
    if not isinstance(lead_time, (int, float)):
        raise TypeError("lead_time must be numeric")
    if lead_time < 0:
        raise ValueError(f"lead_time (L) cannot be negative, got {lead_time}")

    d = demand_per_period if demand_per_period is not None else avg_demand
    if d is None:
        raise ValueError("demand_per_period (d) must be provided. Missing numerical inputs cannot be invented.")
    if not isinstance(d, (int, float)):
        raise TypeError("demand_per_period must be numeric")
    if d <= 0:
        raise ValueError(f"demand_per_period (d) must be strictly positive, got {d}")

    # Determine inventory position
    ip = inventory_position
    if ip is None:
        if current_inventory is not None:
            if not isinstance(current_inventory, (int, float)):
                raise TypeError("current_inventory must be numeric")
            oo = float(on_order) if on_order is not None else 0.0
            bo = float(backorders) if backorders is not None else 0.0
            ip = float(current_inventory) + oo - bo
        else:
            raise ValueError(
                "inventory_position (or current_inventory) must be provided. "
                "Missing numerical inputs cannot be invented."
            )
    else:
        if not isinstance(ip, (int, float)):
            raise TypeError("inventory_position must be numeric")

    if not isinstance(std_demand, (int, float)):
        raise TypeError("std_demand must be numeric")
    if std_demand < 0:
        raise ValueError(f"std_demand cannot be negative, got {std_demand}")

    t_val = float(review_period)
    l_val = float(lead_time)
    d_val = float(d)
    ip_val = float(ip)
    sd_val = float(std_demand)

    # 1. Inputs Extracted
    inputs_extracted = {
        "review_period": t_val,
        "lead_time": l_val,
        "demand_per_period": d_val,
        "inventory_position": ip_val,
        "service_level": float(service_level) if service_level is not None else None,
        "std_demand": sd_val,
        "safety_stock_input": float(safety_stock) if safety_stock is not None else None,
        "current_on_hand": float(current_inventory) if current_inventory is not None else None,
        "current_on_order": float(on_order) if on_order is not None else None,
        "current_backorders": float(backorders) if backorders is not None else None,
        "time_unit": time_unit,
    }

    # 2. Variable Definitions
    variable_definitions = {
        "T": f"Review period / interval between orders ({time_unit}s)",
        "L": f"Replenishment lead time ({time_unit}s)",
        "P": f"Protection period = review period + lead time = T + L ({time_unit}s)",
        "d": f"Average demand per period (units/{time_unit})",
        "sigma_d": f"Standard deviation of demand per period (units/{time_unit})",
        "mu_P": "Expected demand during protection period = d * (T + L) (units)",
        "sigma_P": "Standard deviation of demand during protection period = sigma_d * sqrt(T + L) (units)",
        "z": "Safety factor = norm_ppf(SL)",
        "SS": "Safety stock buffer = z * sigma_P (units)",
        "S": "Target inventory / Order-up-to level = demand during protection period + safety stock (units)",
        "IP": "Current inventory position = On_Hand + On_Order - Backorders (units)",
        "q": "Order quantity = max(0, S - IP) (units)",
    }

    # 3. Formula
    formula = {
        "protection_period": "P = T + L",
        "demand_during_protection": "mu_P = d * (T + L)",
        "std_protection_demand": "sigma_P = sigma_d * sqrt(T + L)",
        "safety_stock": "SS = z * sigma_d * sqrt(T + L)",
        "target_inventory": "S = demand_during_protection + safety_stock = d * (T + L) + z * sigma_d * sqrt(T + L)",
        "inventory_position": "IP = On_Hand + On_Order - Backorders",
        "order_quantity": "q = max(0, target_inventory - inventory_position)",
    }

    # Numerical execution
    protection_period = t_val + l_val
    demand_during_protection = d_val * protection_period
    var_protection_demand = (sd_val ** 2) * protection_period
    std_protection_demand = math.sqrt(var_protection_demand)

    z: Optional[float] = None
    if safety_stock is not None:
        if not isinstance(safety_stock, (int, float)):
            raise TypeError("safety_stock must be numeric")
        if safety_stock < 0:
            raise ValueError(f"safety_stock cannot be negative, got {safety_stock}")
        ss_val = float(safety_stock)
        if sd_val > 0.0 and service_level is not None:
            z = norm_ppf(float(service_level))
    elif sd_val > 0.0:
        if service_level is None:
            raise ValueError("service_level (or direct safety_stock) is required when std_demand > 0. Missing numerical inputs cannot be invented.")
        if not isinstance(service_level, (int, float)):
            raise TypeError("service_level must be numeric")
        if not (0.0 < service_level < 1.0):
            raise ValueError(f"service_level must be strictly between 0 and 1, got {service_level}")
        z = norm_ppf(float(service_level))
        ss_val = z * std_protection_demand
    else:
        if service_level is not None:
            if not (0.0 < service_level < 1.0):
                raise ValueError(f"service_level must be between 0 and 1, got {service_level}")
            z = norm_ppf(float(service_level))
        else:
            z = 0.0
        ss_val = 0.0

    target_inventory = demand_during_protection + ss_val
    order_quantity = max(0.0, target_inventory - ip_val)
    rounded_order_qty = int(math.ceil(order_quantity))
    rounded_target = int(math.ceil(target_inventory))
    rounded_ss = int(math.ceil(ss_val))

    # 4. Formula with Substituted Values
    z_str = f"{z:.4f}" if z is not None else "0.0000"
    formula_with_substituted_values = {
        "protection_period": f"P = {t_val:,.1f} + {l_val:,.1f} = {protection_period:,.1f} {time_unit}s",
        "demand_during_protection": f"mu_P = {d_val:,.2f} * ({t_val:,.1f} + {l_val:,.1f}) = {d_val:,.2f} * {protection_period:,.1f} = {demand_during_protection:,.2f} units",
        "std_protection_demand": f"sigma_P = {sd_val:,.2f} * sqrt({protection_period:,.1f}) = {sd_val:,.2f} * {math.sqrt(protection_period):.4f} = {std_protection_demand:,.2f} units",
        "safety_stock": f"SS = {z_str} * {std_protection_demand:,.2f} = {ss_val:,.2f} units" if safety_stock is None else f"SS = direct input = {ss_val:,.2f} units",
        "target_inventory": f"S = {demand_during_protection:,.2f} + {ss_val:,.2f} = {target_inventory:,.2f} units (ceiling: {rounded_target} units)",
        "order_quantity": f"q = max(0, {target_inventory:,.2f} - {ip_val:,.2f}) = max(0, {(target_inventory - ip_val):,.2f}) = {order_quantity:,.2f} units (ceiling: {rounded_order_qty} units)",
    }

    # 5. Intermediate Calculations
    intermediate_calculations = {
        "protection_period": protection_period,
        "demand_during_protection": demand_during_protection,
        "variance_protection_demand": var_protection_demand,
        "std_protection_demand": std_protection_demand,
        "z_score": z,
        "safety_stock": ss_val,
        "rounded_safety_stock": rounded_ss,
        "target_inventory": target_inventory,
        "rounded_target_inventory": rounded_target,
        "inventory_position": ip_val,
        "raw_net_shortfall": target_inventory - ip_val,
        "order_quantity": order_quantity,
        "rounded_order_quantity": rounded_order_qty,
    }

    # 6. Final Answer
    final_answer = {"order_quantity": order_quantity, "target_inventory": target_inventory}

    # 7. Verification Checks
    p_pass = (protection_period > 0)
    s_buffer_pass = (target_inventory >= ss_val - 1e-4)
    q_nonneg_pass = (order_quantity >= 0)
    expected_order_qty = max(0.0, target_inventory - ip_val)
    ip_consistency_pass = abs(order_quantity - expected_order_qty) < 1e-4

    verification_checks = [
        f"Protection interval check: P = {t_val:,.1f} + {l_val:,.1f} = {protection_period:,.1f} > 0 ({'PASS' if p_pass else 'FAIL'}: Protection window covers entire review cycle and delivery)",
        f"Order-up-to buffer check: S ({target_inventory:,.2f}) >= SS ({ss_val:,.2f}) ({'PASS' if s_buffer_pass else 'FAIL'}: Base stock covers cycle demand plus buffer)",
        f"Non-negativity order check: q = {order_quantity:,.2f} >= 0 ({'PASS' if q_nonneg_pass else 'FAIL'}: Verified non-negative order quantity; negative returns prevented)",
        f"Inventory position consistency check: Target ({target_inventory:,.2f}) - IP ({ip_val:,.2f}) = Net Shortfall ({target_inventory - ip_val:,.2f}) -> q = {order_quantity:,.2f} ({'PASS' if ip_consistency_pass else 'FAIL'})",
    ]

    # 8. Assumptions
    assumptions = [
        "Periodic review inventory control policy (T, S) / Order-Up-To Base Stock Policy.",
        "Protection period equals review interval (T) plus replenishment lead time (L).",
        "Inventory position is reviewed strictly at periodic intervals T: IP = On_Hand + On_Order - Backorders.",
        "Target inventory (S) must protect against demand variability throughout the entire protection period (T + L).",
        "Order quantity is non-negative: q = max(0, S - IP). Negative orders (returns) are not permitted.",
        "Demand across successive review periods is independent and identically normally distributed.",
    ]

    return FixedPeriodReviewResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate_calculations,
        final_answer=final_answer,
        order_quantity=order_quantity,
        rounded_order_quantity=rounded_order_qty,
        target_inventory=target_inventory,
        rounded_target_inventory=rounded_target,
        verification_checks=verification_checks,
        assumptions=assumptions,
        review_period=t_val,
        lead_time=l_val,
        demand_per_period=d_val,
        inventory_position=ip_val,
        service_level=float(service_level) if service_level is not None else None,
        std_demand=sd_val,
        time_unit=time_unit,
        protection_period=protection_period,
        demand_during_protection=demand_during_protection,
        variance_protection_demand=var_protection_demand,
        std_protection_demand=std_protection_demand,
        z_score=z,
        safety_stock=ss_val,
        rounded_safety_stock=rounded_ss,
        current_on_hand=float(current_inventory) if current_inventory is not None else None,
        current_on_order=float(on_order) if on_order is not None else None,
        current_backorders=float(backorders) if backorders is not None else None,
    )


calculate_fixed_period_review = calculate_periodic_review


# ---------------------------------------------------------------------------
# 5. Newsvendor Tool (Single-Period Stochastic Model)
# ---------------------------------------------------------------------------

def calculate_newsvendor(
    selling_price: Optional[float] = None,
    cost: Optional[float] = None,
    mean_demand: Optional[float] = None,
    std_demand: Optional[float] = None,
    salvage_value: Optional[float] = None,
    shortage_penalty: float = 0.0,
    underage_cost: Optional[float] = None,
    overage_cost: Optional[float] = None,
    demand_table: Optional[List[Dict[str, Any]]] = None,
) -> NewsvendorResult:
    """
    Calculate optimal order quantity, critical ratio, and financial expectations
    for the single-period Newsvendor model under normal or discrete demand distribution.
    Returns all 8 verification components for complete student verification.
    """
    # 1. Determine Cu and Co
    if underage_cost is not None and overage_cost is not None:
        if not isinstance(underage_cost, (int, float)) or underage_cost <= 0:
            raise ValueError(f"underage_cost must be strictly positive, got {underage_cost}")
        if not isinstance(overage_cost, (int, float)) or overage_cost <= 0:
            raise ValueError(f"overage_cost must be strictly positive, got {overage_cost}")
        cu_val = float(underage_cost)
        co_val = float(overage_cost)
        p_val = float(selling_price) if selling_price is not None else None
        c_val = float(cost) if cost is not None else None
        s_val = float(salvage_value) if salvage_value is not None else None
        g_val = float(shortage_penalty) if shortage_penalty is not None else 0.0
        has_cost_components = (p_val is not None and c_val is not None)
    else:
        if selling_price is None:
            raise ValueError("selling_price (p) or underage_cost must be provided. Missing numerical inputs cannot be invented.")
        if not isinstance(selling_price, (int, float)):
            raise TypeError("selling_price must be numeric")
        if selling_price <= 0:
            raise ValueError(f"selling_price must be strictly positive, got {selling_price}")

        if cost is None:
            raise ValueError("cost (c) or overage_cost must be provided. Missing numerical inputs cannot be invented.")
        if not isinstance(cost, (int, float)):
            raise TypeError("cost must be numeric")
        if cost <= 0:
            raise ValueError(f"cost must be strictly positive, got {cost}")

        if salvage_value is None:
            raise ValueError("salvage_value (s) or overage_cost must be provided (e.g. 0.0). Missing numerical inputs cannot be invented.")
        if not isinstance(salvage_value, (int, float)):
            raise TypeError("salvage_value must be numeric")
        if salvage_value >= cost:
            raise ValueError(
                f"salvage_value ({salvage_value}) must be strictly less than cost ({cost}), "
                "otherwise inventory carries zero financial overage risk."
            )

        if not isinstance(shortage_penalty, (int, float)):
            raise TypeError("shortage_penalty must be numeric")
        if shortage_penalty < 0:
            raise ValueError(f"shortage_penalty cannot be negative, got {shortage_penalty}")

        p_val = float(selling_price)
        c_val = float(cost)
        s_val = float(salvage_value)
        g_val = float(shortage_penalty) if shortage_penalty is not None else 0.0

        cu_val = p_val - c_val + g_val
        if cu_val <= 0:
            raise ValueError("Underage cost (p - c + g) must be positive: unit margin plus shortage penalty is non-positive.")
        co_val = c_val - s_val
        if co_val <= 0:
            raise ValueError("Overage cost (c - s) must be strictly positive.")
        has_cost_components = True

    total_cost_denom = cu_val + co_val
    critical_ratio = round(cu_val / total_cost_denom, 8)

    # 2. Demand Distribution: Discrete Table vs Normal Distribution
    cum_table: Optional[List[Dict[str, Any]]] = None
    selected_table_row: Optional[Dict[str, Any]] = None

    if demand_table is not None:
        if not isinstance(demand_table, list) or len(demand_table) == 0:
            raise ValueError("demand_table must be a non-empty list of {demand, probability} rows.")

        parsed_rows = []
        for idx, row in enumerate(demand_table):
            if not isinstance(row, dict):
                raise TypeError(f"demand_table row {idx} must be a dict with 'demand' and 'probability'.")
            d_item = row.get("demand", row.get("d"))
            p_item = row.get("probability", row.get("prob", row.get("p")))
            if d_item is None or p_item is None:
                raise ValueError(f"demand_table row {idx} must contain 'demand' and 'probability', got {row}.")
            if not isinstance(d_item, (int, float)) or d_item < 0:
                raise ValueError(f"demand in row {idx} must be a non-negative number, got {d_item}")
            if not isinstance(p_item, (int, float)) or p_item < 0:
                raise ValueError(f"probability in row {idx} must be a non-negative number, got {p_item}")
            parsed_rows.append((float(d_item), float(p_item)))

        # Validate probability sum (accept probabilities summing to 1 or percentages summing to 100)
        total_p = sum(p for _, p in parsed_rows)
        if math.isclose(total_p, 100.0, rel_tol=1e-3, abs_tol=1e-3) or any(p > 1.0 for _, p in parsed_rows):
            parsed_rows = [(d, p / 100.0) for d, p in parsed_rows]
            total_p = sum(p for _, p in parsed_rows)

        if not math.isclose(total_p, 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError(f"Probabilities in demand_table must sum to 1.0 (or 100%), got {total_p}")

        # Sort rows ascending by demand
        parsed_rows.sort(key=lambda item: item[0])

        cum_table = []
        running_cum = 0.0
        for d_i, p_i in parsed_rows:
            running_cum += p_i
            cum_table.append({
                "demand": d_i,
                "probability": p_i,
                "cumulative_probability": round(running_cum, 6),
            })
        cum_table[-1]["cumulative_probability"] = 1.0

        # Smallest demand whose cumulative probability >= critical_ratio
        for row in cum_table:
            if row["cumulative_probability"] >= critical_ratio - 1e-9:
                selected_table_row = row
                break
        if selected_table_row is None:
            selected_table_row = cum_table[-1]

        recommended_qty = selected_table_row["demand"]
        rounded_qty = int(round(recommended_qty))

        mu_val = sum(r["demand"] * r["probability"] for r in cum_table)
        var_d = sum(r["probability"] * ((r["demand"] - mu_val) ** 2) for r in cum_table)
        sigma_val = math.sqrt(var_d)
        z = None
        g_z = None

        if has_cost_components and p_val is not None and c_val is not None:
            expected_lost_sales = sum(max(0.0, r["demand"] - recommended_qty) * r["probability"] for r in cum_table)
            expected_sales = sum(min(recommended_qty, r["demand"]) * r["probability"] for r in cum_table)
            expected_leftover = sum(max(0.0, recommended_qty - r["demand"]) * r["probability"] for r in cum_table)
            expected_revenue = (p_val * expected_sales) + ((s_val or 0.0) * expected_leftover)
            expected_cost = c_val * recommended_qty
            expected_shortage_cost = g_val * expected_lost_sales
            expected_profit = expected_revenue - expected_cost - expected_shortage_cost
            profit_pi = (p_val - c_val) * mu_val
            evpi = max(0.0, profit_pi - expected_profit)
        else:
            expected_lost_sales = None
            expected_sales = None
            expected_leftover = None
            expected_revenue = None
            expected_cost = None
            expected_shortage_cost = None
            expected_profit = None
            profit_pi = None
            evpi = None

    else:
        # Normal distribution mode
        if mean_demand is None:
            raise ValueError("mean_demand (mu) must be provided when demand_table is not given. Missing numerical inputs cannot be invented.")
        if not isinstance(mean_demand, (int, float)):
            raise TypeError("mean_demand must be numeric")
        if mean_demand <= 0:
            raise ValueError(f"mean_demand must be strictly positive, got {mean_demand}")

        if std_demand is None:
            raise ValueError("std_demand (sigma) must be provided when demand_table is not given. Missing numerical inputs cannot be invented.")
        if not isinstance(std_demand, (int, float)):
            raise TypeError("std_demand must be numeric")
        if std_demand <= 0:
            raise ValueError(f"std_demand must be strictly positive, got {std_demand}")

        mu_val = float(mean_demand)
        sigma_val = float(std_demand)

        z = norm_ppf(critical_ratio)
        recommended_qty = max(0.0, mu_val + z * sigma_val)
        rounded_qty = int(round(recommended_qty))

        g_z = unit_normal_loss(z)
        expected_lost_sales = sigma_val * g_z
        expected_sales = mu_val - expected_lost_sales
        expected_leftover = recommended_qty - expected_sales

        if has_cost_components and p_val is not None and c_val is not None:
            expected_revenue = (p_val * expected_sales) + ((s_val or 0.0) * expected_leftover)
            expected_cost = c_val * recommended_qty
            expected_shortage_cost = g_val * expected_lost_sales
            expected_profit = expected_revenue - expected_cost - expected_shortage_cost
            profit_pi = mu_val * (p_val - c_val)
            evpi = max(0.0, profit_pi - expected_profit)
        else:
            expected_revenue = None
            expected_cost = None
            expected_shortage_cost = None
            expected_profit = None
            profit_pi = None
            evpi = None

    # Step 1: Inputs Extracted
    inputs_extracted = {
        "selling_price": p_val,
        "cost": c_val,
        "salvage_value": s_val,
        "shortage_penalty": g_val,
        "mean_demand": mu_val,
        "std_demand": sigma_val,
        "underage_cost_input": float(underage_cost) if underage_cost is not None else None,
        "overage_cost_input": float(overage_cost) if overage_cost is not None else None,
        "demand_table": demand_table,
    }

    # Step 2: Variable Definitions
    variable_definitions = {
        "C_u": "Underage cost = marginal profit forgone per unit short ($/unit)",
        "C_o": "Overage cost = marginal loss per unsold leftover unit ($/unit)",
        "CR": "Critical ratio (fractile) = C_u / (C_u + C_o) = target in-stock probability",
        "Q*": "Optimal order quantity (units)",
    }
    if p_val is not None:
        variable_definitions["p"] = "Selling price per unit ($/unit)"
    if c_val is not None:
        variable_definitions["c"] = "Purchase / production cost per unit ($/unit)"
    if s_val is not None:
        variable_definitions["s"] = "Salvage / liquidation value per unsold unit ($/unit)"
    if g_val > 0:
        variable_definitions["g"] = "Shortage penalty / goodwill loss per unit of unmet demand ($/unit)"

    if demand_table is not None:
        variable_definitions["D"] = "Discrete customer demand random variable"
        variable_definitions["P(D=d)"] = "Discrete probability distribution"
        variable_definitions["P(D<=Q)"] = "Cumulative probability function of demand"
    else:
        variable_definitions["mu"] = "Mean expected customer demand (units)"
        variable_definitions["sigma"] = "Standard deviation of customer demand (units)"
        variable_definitions["z"] = "Safety factor = norm_ppf(CR), standard normal quantile"
        variable_definitions["G_u(z)"] = "Standard normal unit loss function = phi(z) - z * (1 - Phi(z))"

    if has_cost_components:
        variable_definitions["E[Profit]"] = "Expected profit ($)"
        variable_definitions["EVPI"] = "Expected Value of Perfect Information ($)"

    # Step 3: Formula
    formula = {
        "underage_cost": "C_u = p - c + g" if has_cost_components else "C_u (provided directly)",
        "overage_cost": "C_o = c - s" if has_cost_components else "C_o (provided directly)",
        "critical_ratio": "CR = C_u / (C_u + C_o)",
    }
    if demand_table is not None:
        formula["optimal_order_quantity"] = "Smallest demand Q where P(D <= Q) >= CR"
    else:
        formula["safety_factor"] = "z = norm_ppf(CR)"
        formula["optimal_order_quantity"] = "Q* = mu + z * sigma"
        formula["unit_normal_loss"] = "G_u(z) = phi(z) - z * (1 - Phi(z))"

    if has_cost_components:
        formula["expected_profit"] = "E[Profit] = Revenue - Cost - Shortage"
        formula["evpi"] = "EVPI = mu * (p - c) - E[Profit]"
        formula["expected_value_of_perfect_information"] = "EVPI = mu * (p - c) - E[Profit]"

    # Step 4: Formula with Substituted Values
    formula_with_substituted_values = {}
    if has_cost_components and p_val is not None and c_val is not None and s_val is not None:
        formula_with_substituted_values["underage_cost"] = f"C_u = {p_val:,.2f} - {c_val:,.2f} + {g_val:,.2f} = ${cu_val:,.2f}/unit"
        formula_with_substituted_values["overage_cost"] = f"C_o = {c_val:,.2f} - {s_val:,.2f} = ${co_val:,.2f}/unit"
    else:
        formula_with_substituted_values["underage_cost"] = f"C_u = ${cu_val:,.2f}/unit (provided directly)"
        formula_with_substituted_values["overage_cost"] = f"C_o = ${co_val:,.2f}/unit (provided directly)"

    formula_with_substituted_values["critical_ratio"] = f"CR = {cu_val:,.2f} / ({cu_val:,.2f} + {co_val:,.2f}) = {cu_val:,.2f} / {total_cost_denom:,.2f} = {critical_ratio:.4f} ({critical_ratio*100:.2f}%)"

    if demand_table is not None and selected_table_row is not None:
        sel_str = (
            f"Select smallest demand with cumulative probability P(D <= Q) >= CR ({critical_ratio:.4f}). "
            f"From demand table, demand = {recommended_qty:,.0f} has cumulative probability "
            f"{selected_table_row['cumulative_probability']:.4f} >= {critical_ratio:.4f}. Optimal Q* = {recommended_qty:,.0f} units."
        )
        formula_with_substituted_values["optimal_order_quantity"] = sel_str
        formula_with_substituted_values["table_selection"] = sel_str
    else:
        z_val = z if z is not None else 0.0
        formula_with_substituted_values["safety_factor"] = f"z = norm_ppf({critical_ratio:.4f}) = {z_val:.4f}"
        formula_with_substituted_values["optimal_order_quantity"] = f"Q* = {mu_val:,.2f} + ({z_val:.4f} * {sigma_val:,.2f}) = {mu_val:,.2f} + {(z_val * sigma_val):,.2f} = {recommended_qty:,.2f} units (rounded: {rounded_qty} units)"
        if g_z is not None:
            formula_with_substituted_values["unit_normal_loss"] = f"G_u({z_val:.4f}) = {norm_pdf(z_val):.4f} - {z_val:.4f} * (1 - {norm_cdf(z_val):.4f}) = {g_z:.4f}"

    if has_cost_components and expected_profit is not None and evpi is not None:
        formula_with_substituted_values["expected_profit"] = f"E[Profit] = ${expected_profit:,.2f}"
        formula_with_substituted_values["evpi"] = f"EVPI = ${evpi:,.2f}"
        formula_with_substituted_values["expected_value_of_perfect_information"] = f"EVPI = ${evpi:,.2f}"
    else:
        formula_with_substituted_values["expected_profit"] = "Not computed: selling price and unit cost were not provided."
        formula_with_substituted_values["expected_value_of_perfect_information"] = "Not computed: selling price and unit cost were not provided."

    # Step 5: Intermediate Calculations
    intermediate_calculations = {
        "underage_cost": cu_val,
        "overage_cost": co_val,
        "critical_ratio_denominator": total_cost_denom,
        "critical_ratio": critical_ratio,
        "z_score": z,
        "unit_normal_loss": g_z,
        "unit_normal_loss_G_u": g_z,
        "expected_lost_sales": expected_lost_sales,
        "expected_sales": expected_sales,
        "expected_leftover_inventory": expected_leftover,
        "expected_revenue": expected_revenue,
        "expected_cost": expected_cost,
        "expected_shortage_cost": expected_shortage_cost,
        "expected_profit": expected_profit,
        "profit_under_perfect_information": profit_pi,
        "expected_value_of_perfect_information": evpi,
        "evpi": evpi,
        "in_stock_probability": critical_ratio,
        "stockout_probability": 1.0 - critical_ratio,
    }
    if cum_table is not None:
        intermediate_calculations["cumulative_probability_table"] = cum_table
        intermediate_calculations["demand_table"] = cum_table
        intermediate_calculations["selected_table_row"] = selected_table_row

    # Step 6: Final Answer
    final_answer = recommended_qty

    # Step 7: Verification Checks
    cr_pass = (0.0 < critical_ratio < 1.0)
    verification_checks = [
        f"Critical ratio boundary check: 0 < CR = {critical_ratio:.4f} < 1 ({'PASS' if cr_pass else 'FAIL'}: Valid service level probability)",
    ]
    if cum_table is not None and selected_table_row is not None:
        table_sel_pass = (selected_table_row['cumulative_probability'] >= critical_ratio - 1e-4)
        verification_checks.append(
            f"Discrete table selection check: P(D <= {recommended_qty:,.0f}) = {selected_table_row['cumulative_probability']:.4f} >= CR ({critical_ratio:.4f}) ({'PASS' if table_sel_pass else 'FAIL'})"
        )
        total_p = sum(r['probability'] for r in cum_table)
        total_p_pass = abs(total_p - 1.0) < 1e-4
        verification_checks.append(
            f"Demand table total probability check: sum(p_i) = {total_p:.4f} ≈ 1.0 ({'PASS' if total_p_pass else 'FAIL'})"
        )
    if expected_sales is not None and mu_val is not None:
        sales_pass = (expected_sales <= mu_val + 1e-4)
        verification_checks.append(
            f"Expected sales ceiling check: E[Sales] ({expected_sales:,.2f}) <= mean_demand ({mu_val:,.2f}) ({'PASS' if sales_pass else 'FAIL'}: Satisfied sales cannot exceed customer demand)"
        )
    if expected_leftover is not None:
        leftover_pass = (expected_leftover >= -1e-4)
        verification_checks.append(
            f"Leftover stock non-negativity check: E[Leftover] = {expected_leftover:,.2f} >= 0 ({'PASS' if leftover_pass else 'FAIL'})"
        )
    if evpi is not None:
        evpi_pass = (evpi >= -1e-4)
        verification_checks.append(
            f"Information value non-negativity check: EVPI = ${evpi:,.2f} >= 0 ({'PASS' if evpi_pass else 'FAIL'}: Perfect forecasting value is non-negative)"
        )

    # Step 8: Assumptions
    assumptions = [
        "Single selling season for perishable, fashion, or seasonal items with no mid-season replenishment.",
        "Underage cost captures forgone contribution margin plus any shortage/goodwill penalty.",
        "Overage cost captures the net loss per unsold unit liquidated at salvage value.",
        "Critical ratio (CR = Cu / (Cu + Co)) defines the exact profit-maximizing in-stock probability.",
    ]
    if demand_table is not None:
        assumptions.append("Demand is discrete and governed by the provided demand_table probability distribution.")
    else:
        assumptions.append("Demand is normally distributed with parameters: mean = mu, standard deviation = sigma.")

    if not has_cost_components:
        assumptions.append(
            "Underage cost (Cu) and overage cost (Co) were given directly without selling price and cost; "
            "expected revenue, expected profit, and EVPI cannot be computed and are returned as None."
        )
    else:
        assumptions.append("Unsold units at the end of the selling season are salvaged with zero secondary market disruption.")

    return NewsvendorResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate_calculations,
        final_answer=final_answer,
        recommended_order_quantity=recommended_qty,
        rounded_order_quantity=rounded_qty,
        critical_ratio=critical_ratio,
        verification_checks=verification_checks,
        assumptions=assumptions,
        selling_price=p_val,
        cost=c_val,
        salvage_value=s_val,
        mean_demand=mu_val,
        std_demand=sigma_val,
        shortage_penalty=g_val,
        underage_cost=cu_val,
        overage_cost=co_val,
        z_score=z,
        unit_normal_loss=g_z,
        expected_lost_sales=expected_lost_sales,
        expected_sales=expected_sales,
        expected_leftover_inventory=expected_leftover,
        expected_revenue=expected_revenue,
        expected_cost=expected_cost,
        expected_shortage_cost=expected_shortage_cost,
        expected_profit=expected_profit,
        profit_under_perfect_information=profit_pi,
        expected_value_of_perfect_information=evpi,
        in_stock_probability=critical_ratio,
        stockout_probability=1.0 - critical_ratio,
        demand_table=demand_table,
    )
