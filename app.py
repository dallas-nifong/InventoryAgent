"""
Supply Chain Analytics: Inventory Decision Agent Application.

Provides two operational modes:
1. Interactive Streamlit Web Interface (when launched via `streamlit run app.py`)
2. Command-Line Interface (CLI) (when executed via `python3 app.py [options]`)

Implements the 6-stage architecture:
Customer Question -> Identify Model -> Extract Inputs -> Select Tool -> Calculate -> Verify -> Explain
"""

from __future__ import annotations

import argparse
from datetime import datetime
import io
import json
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import pandas as pd
except ImportError:
    pd = None

from inventory_agent import AgentResponse, InventoryAgent
from chart_helpers import build_model_chart
from inventory_tools import calculate_multicriteria_inventory_classification


# =============================================================================
# CLI IMPLEMENTATION
# =============================================================================

BENCHMARK_SCENARIOS = [
    {
        "name": "1. Economic Order Quantity (EOQ)",
        "question": (
            "Our annual demand is 10,000 units, ordering cost is $50 per order, "
            "and unit holding cost is $2.00 per year. What is our optimal order quantity and annual inventory cost?"
        ),
    },
    {
        "name": "2. Quantity Discount Model",
        "question": (
            "Annual demand is 5,000 units and setup cost is $40 per order. Our carrying cost rate is 20% of unit price. "
            "Supplier price breaks are: 1-499 units at $30, 500-999 units at $28, and 1000+ units at $26. Which quantity should we order?"
        ),
    },
    {
        "name": "3. Safety Stock & Reorder Point (ROP)",
        "question": (
            "Average daily demand is 100 units with standard deviation of 15 units. "
            "Supplier lead time is 4 days. If our target cycle service level is 95%, what should our safety stock and reorder point be?"
        ),
    },
    {
        "name": "4. Fixed-Period Review ((T, S) Policy)",
        "question": (
            "We review inventory every 7 days and supplier lead time is 3 days. Average demand is 20 units per day "
            "with a standard deviation of 5 units. At a 95% service level, with 50 units on hand and 30 units on order, what is our order quantity?"
        ),
    },
    {
        "name": "5. Newsvendor Single-Period Model",
        "question": (
            "A seasonal designer ski jacket sells for $150 and costs $60 wholesale. Unsold jackets are liquidated for $20 at the end of winter. "
            "Forecasted demand is normally distributed with a mean of 1,000 units and a standard deviation of 200. What is the optimal order quantity?"
        ),
    },
]

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "EOQ": "EOQ",
    "QUANTITY_DISCOUNT": "Quantity Discount",
    "SAFETY_STOCK_ROP": "Safety Stock & ROP",
    "FIXED_PERIOD_REVIEW": "Fixed-Period Review",
    "NEWSVENDOR": "Newsvendor",
}


def run_cli_single(agent: InventoryAgent, question: str):
    """Run agent on a single question and display structured analytical output."""
    print("=" * 80)
    print("SUPPLY CHAIN ANALYTICS INVENTORY DECISION AGENT")
    print("=" * 80)
    print(f"\n[Customer Question]:\n  {question}\n")

    resp: AgentResponse = agent.process_question(question)

    print("-" * 80)
    print("EXECUTION PIPELINE TRACE:")
    print("-" * 80)
    print(f"1. Identified Model   : {resp.model_name} (Confidence: {resp.confidence * 100:.1f}%)")
    print(f"2. Extracted Inputs   : {json.dumps(resp.extracted_inputs, indent=2)}")
    if resp.missing_inputs:
        print(f"   Missing Inputs     : {resp.missing_inputs}")
    print(f"3. Python Tool Selected: {resp.tool_name}")
    print(f"4. Verification Passed : {'✅ YES' if resp.verification.passed else '❌ NO'}")
    for chk in resp.verification.checks:
        print(f"   • {chk}")

    print("-" * 80)
    print("EXECUTIVE DECISION MEMO:")
    print("-" * 80)
    print(resp.explanation)
    print("=" * 80 + "\n")


def run_cli_interactive(agent: InventoryAgent):
    """Run an interactive terminal session."""
    print("=" * 80)
    print("SUPPLY CHAIN ANALYTICS: INVENTORY DECISION AGENT (CLI Mode)")
    print("Type your inventory question below, 'demo' for benchmark examples, or 'exit' to quit.")
    print("=" * 80)

    while True:
        try:
            user_input = input("\n📝 Inventory Question > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting Supply Chain Decision Agent. Goodbye!")
                break
            if user_input.lower() == "demo":
                for item in BENCHMARK_SCENARIOS:
                    print(f"\n--- {item['name']} ---")
                    run_cli_single(agent, item["question"])
                continue

            run_cli_single(agent, user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nSession terminated.")
            break


# =============================================================================
# CSV DATA UPLOAD & INVENTORY ANALYTICS UTILITIES
# =============================================================================

REQUIRED_CSV_COLUMNS = ["Product", "Category", "Units", "Unit Cost"]
CLASSIFICATION_REQUIRED_COLUMNS = ["Item Number", "Item Name", "UoM", "Sales ($)", "Hits"]


def process_uploaded_csv(file_or_buffer: Any) -> Tuple[Optional[Any], List[str]]:
    """
    Read an uploaded CSV file into a pandas DataFrame and validate required columns.

    Supports two schemas:
    1. Inventory Classification Schema:
       - Item Number
       - Item Name
       - UoM
       - Sales ($)
       - Hits
    2. Legacy Product Inventory Schema:
       - Product
       - Category
       - Units
       - Unit Cost

    Returns:
        Tuple of (dataframe, missing_columns_list)
    """
    global pd
    if pd is None:
        import pandas as pd

    try:
        df = pd.read_csv(file_or_buffer)
    except Exception as e:
        return None, [f"Error reading CSV: {e}"]

    # Clean whitespace in column names if any
    df.columns = [str(col).strip() for col in df.columns]

    # Normalize common column name aliases for classification
    col_map = {}
    for c in df.columns:
        c_clean = str(c).strip()
        c_lower = c_clean.lower()
        if c_clean not in CLASSIFICATION_REQUIRED_COLUMNS:
            if c_lower in ["sales", "revenue", "annual sales", "sales_usd"]:
                col_map[c] = "Sales ($)"
            elif c_lower in ["hit", "hit count", "picks", "frequency", "orders"]:
                col_map[c] = "Hits"
            elif c_lower in ["item no", "item_number", "item #", "sku", "part number", "item_no"]:
                col_map[c] = "Item Number"
            elif c_lower in ["item description", "description", "product name"]:
                col_map[c] = "Item Name"
            elif c_lower in ["uom", "unit of measure", "unit"]:
                col_map[c] = "UoM"
    if col_map:
        df = df.rename(columns=col_map)

    # Detect schema based on column matches
    cls_matches = sum(1 for c in CLASSIFICATION_REQUIRED_COLUMNS if c in df.columns)
    inv_matches = sum(1 for c in REQUIRED_CSV_COLUMNS if c in df.columns)

    if cls_matches >= 2 and cls_matches >= inv_matches:
        schema = "classification"
        missing = [col for col in CLASSIFICATION_REQUIRED_COLUMNS if col not in df.columns]
    else:
        schema = "inventory"
        missing = [col for col in REQUIRED_CSV_COLUMNS if col not in df.columns]

    df.attrs["dataset_type"] = schema
    if missing:
        return df, missing

    if schema == "classification":
        # Coerce numeric columns for classification
        if "Sales ($)" in df.columns:
            if not pd.api.types.is_numeric_dtype(df["Sales ($)"]):
                df["Sales ($)"] = pd.to_numeric(
                    df["Sales ($)"]
                    .astype(str)
                    .str.replace("$", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.strip(),
                    errors="coerce",
                ).fillna(0.0)
            else:
                df["Sales ($)"] = pd.to_numeric(df["Sales ($)"], errors="coerce").fillna(0.0)

        if "Hits" in df.columns:
            if not pd.api.types.is_numeric_dtype(df["Hits"]):
                df["Hits"] = pd.to_numeric(
                    df["Hits"]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .str.strip(),
                    errors="coerce",
                ).fillna(0.0)
            else:
                df["Hits"] = pd.to_numeric(df["Hits"], errors="coerce").fillna(0.0)
    else:
        # Coerce numeric columns cleanly without dropping valid rows for legacy inventory
        if "Units" in df.columns:
            if not pd.api.types.is_numeric_dtype(df["Units"]):
                df["Units"] = pd.to_numeric(
                    df["Units"].astype(str).str.replace(",", "", regex=False).str.strip(),
                    errors="coerce",
                )
            else:
                df["Units"] = pd.to_numeric(df["Units"], errors="coerce")

        if "Unit Cost" in df.columns:
            if not pd.api.types.is_numeric_dtype(df["Unit Cost"]):
                df["Unit Cost"] = pd.to_numeric(
                    df["Unit Cost"]
                    .astype(str)
                    .str.replace("$", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.strip(),
                    errors="coerce",
                )
            else:
                df["Unit Cost"] = pd.to_numeric(df["Unit Cost"], errors="coerce")

    return df, []


def calculate_inventory_dataframe_metrics(df: Any) -> Dict[str, Any]:
    """
    Perform pandas-based data calculations on the uploaded inventory dataframe:
    1. Calculate total Units.
    2. Calculate average Unit Cost.
    3. Identify the Product with the highest Units.
    4. Group the data by Category and calculate total Units for each Category.
    5. Create a new calculated column: Inventory Value = Units × Unit Cost.
    6. Calculate total Inventory Value across all products.

    Returns:
        Dictionary with analytical results and the enhanced dataframe.
    """
    global pd
    if pd is None:
        import pandas as pd

    df_calc = df.copy()

    # 1. Total Units
    total_units = float(df_calc["Units"].sum())

    # 2. Average Unit Cost
    avg_unit_cost = float(df_calc["Unit Cost"].mean())

    # 3. Product with highest Units
    if not df_calc.empty and df_calc["Units"].notna().any():
        max_idx = df_calc["Units"].idxmax()
        highest_units_product = str(df_calc.loc[max_idx, "Product"])
        highest_units_count = float(df_calc.loc[max_idx, "Units"])
    else:
        highest_units_product = "N/A"
        highest_units_count = 0.0

    # 4. Group by Category and calculate total Units for each Category
    category_units_df = (
        df_calc.groupby("Category")["Units"]
        .sum()
        .reset_index()
        .sort_values(by="Units", ascending=False)
    )

    # 5. Create new calculated column: Inventory Value = Units * Unit Cost
    df_calc["Inventory Value"] = df_calc["Units"] * df_calc["Unit Cost"]

    # 6. Total Inventory Value across all products
    total_inventory_value = float(df_calc["Inventory Value"].sum())

    return {
        "total_units": total_units,
        "avg_unit_cost": avg_unit_cost,
        "highest_units_product": highest_units_product,
        "highest_units_count": highest_units_count,
        "category_units_df": category_units_df,
        "total_inventory_value": total_inventory_value,
        "dataframe_with_inventory_value": df_calc,
    }


# =============================================================================
# STREAMLIT DISPLAY & FORMATTING HELPERS
# =============================================================================

def remove_null_values(obj: Any) -> Any:
    """Recursively remove keys or list items with None / null values."""
    if isinstance(obj, dict):
        return {k: remove_null_values(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [remove_null_values(item) for item in obj if item is not None]
    return obj


def escape_dollars(text: str) -> str:
    """
    Escape dollar signs at display time so every dollar sign outside inline code
    shows as plain text instead of being interpreted as LaTeX math by Streamlit / Markdown.
    Includes currency amounts ($707.11) and unit labels like ($/order).
    Leaves inline code spans (`...` or ```...```) untouched.
    """
    if not isinstance(text, str):
        return text
    # Split text by inline code spans (`...` or ```...```)
    # Even tokens are outside code, odd tokens are inside code
    tokens = re.split(r"(`+[\s\S]*?`+)", text)
    out = []
    for i, token in enumerate(tokens):
        if i % 2 == 1:
            # Inside inline code: preserve as-is
            out.append(token)
        else:
            # Outside inline code: escape any unescaped dollar sign
            out.append(re.sub(r"(?<!\\)\$", r"\$", token))
    return "".join(out)


def get_model_metric_cards(resp_or_result: Any, model_name: Optional[str] = None) -> List[Tuple[str, str]]:
    """
    Returns a list of (label, formatted_value) tuples for the 4 key metric cards:
    - EOQ: order quantity, orders per year, months between orders, total annual inventory cost
    - Quantity Discount: order quantity, unit price, total annual cost, annual savings
    - Safety Stock / ROP: safety stock, reorder point, service level, z value
    - Fixed-Period Review: amount to order, target inventory, protection period, safety stock
    - Newsvendor: order quantity, critical ratio, underage cost, overage cost
    """
    if resp_or_result is None:
        return []
    if hasattr(resp_or_result, "calculation_result"):
        res = resp_or_result.calculation_result
        m = resp_or_result.model_name
    else:
        res = resp_or_result
        m = model_name or getattr(res, "model_name", "")

    if res is None:
        return []

    if m == "EOQ":
        qty = getattr(res, "rounded_eoq", getattr(res, "rounded_order_quantity", 0))
        opy = getattr(res, "orders_per_year", 0.0)
        ct_m = getattr(res, "cycle_time_months", None)
        months_str = f"{ct_m:.2f} months" if ct_m is not None else "N/A"
        cost = getattr(res, "annual_inventory_cost", 0.0)
        return [
            ("Order Quantity", f"{qty:,} units"),
            ("Orders Per Year", f"{opy:.2f}"),
            ("Months Between Orders", months_str),
            ("Total Annual Cost", f"${cost:,.2f}"),
        ]
    elif m == "QUANTITY_DISCOUNT":
        qty = getattr(res, "rounded_optimal_quantity", 0)
        p = getattr(res, "optimal_unit_price", 0.0)
        tc = getattr(res, "min_total_annual_cost", 0.0)
        sav = getattr(res, "annual_savings", 0.0)
        return [
            ("Order Quantity", f"{qty:,} units"),
            ("Unit Price", f"${p:,.2f}"),
            ("Total Annual Cost", f"${tc:,.2f}"),
            ("Annual Savings", f"${sav:,.2f}"),
        ]
    elif m == "SAFETY_STOCK_ROP":
        ss = getattr(res, "rounded_safety_stock", 0)
        rop = getattr(res, "rounded_reorder_point", 0)
        sl = getattr(res, "service_level", None)
        sl_str = f"{sl * 100:.1f}%" if sl is not None else "N/A"
        z = getattr(res, "z_score", None)
        z_str = f"{z:.2f}" if z is not None else "N/A"
        return [
            ("Safety Stock", f"{ss:,} units"),
            ("Reorder Point", f"{rop:,} units"),
            ("Service Level", sl_str),
            ("z Value", z_str),
        ]
    elif m == "FIXED_PERIOD_REVIEW":
        q = getattr(res, "rounded_order_quantity", 0)
        s = getattr(res, "rounded_target_inventory", 0)
        t_unit = getattr(res, "time_unit", "day")
        pp = getattr(res, "protection_period", 0.0)
        ss = getattr(res, "rounded_safety_stock", 0)
        return [
            ("Amount to Order", f"{q:,} units"),
            ("Target Inventory", f"{s:,} units"),
            ("Protection Period", f"{pp:.1f} {t_unit}s"),
            ("Safety Stock", f"{ss:,} units"),
        ]
    elif m == "NEWSVENDOR":
        q = getattr(res, "rounded_optimal_quantity", 0)
        cr = getattr(res, "critical_ratio", 0.0)
        cu = getattr(res, "underage_cost", 0.0)
        co = getattr(res, "overage_cost", 0.0)
        return [
            ("Order Quantity", f"{q:,} units"),
            ("Critical Ratio", f"{cr * 100:.1f}%"),
            ("Underage Cost", f"${cu:,.2f}"),
            ("Overage Cost", f"${co:,.2f}"),
        ]
    return []


def get_plain_language_recommendation(resp_or_result: Any, model_name: Optional[str] = None) -> str:
    """
    Returns a concise two or three sentence plain-language operational recommendation.
    """
    if resp_or_result is None:
        return ""
    if hasattr(resp_or_result, "calculation_result"):
        res = resp_or_result.calculation_result
        m = resp_or_result.model_name
    else:
        res = resp_or_result
        m = model_name or getattr(res, "model_name", "")

    if res is None:
        return ""

    if m == "EOQ":
        ct_m = getattr(res, "cycle_time_months", None)
        cadence = f"{ct_m:.2f} months" if ct_m is not None else f"{getattr(res, 'cycle_time_days', 0):.0f} work days"
        qty = getattr(res, "rounded_eoq", getattr(res, "rounded_order_quantity", 0))
        opy = getattr(res, "orders_per_year", 0.0)
        cost = getattr(res, "annual_inventory_cost", 0.0)
        return (
            f"Place replenishment orders of **{qty:,} units** approximately every **{cadence}** "
            f"({opy:.1f} orders per year). This order quantity balances setup costs against carrying charges, "
            f"achieving a minimum total annual inventory cost of **${cost:,.2f}**."
        )
    elif m == "QUANTITY_DISCOUNT":
        qty = getattr(res, "rounded_optimal_quantity", 0)
        p = getattr(res, "optimal_unit_price", 0.0)
        tier_idx = getattr(res, "optimal_tier_index", 0) + 1
        tc = getattr(res, "min_total_annual_cost", 0.0)
        sav = getattr(res, "annual_savings", 0.0)
        savings_clause = f" with net annual savings of **${sav:,.2f}** versus the baseline tier" if sav > 0 else " as the baseline price structure remains the most cost-effective"
        return (
            f"Order **{qty:,} units** at **${p:,.2f}** per unit to qualify for Tier {tier_idx}. "
            f"This batch size balances purchasing discounts against additional holding charges, "
            f"securing a minimum total annual cost of **${tc:,.2f}**{savings_clause}."
        )
    elif m == "SAFETY_STOCK_ROP":
        rop = getattr(res, "rounded_reorder_point", 0)
        ss = getattr(res, "rounded_safety_stock", 0)
        sl = getattr(res, "service_level", None)
        sl_pct = f"{sl * 100:.1f}%" if sl is not None else "prescribed"
        return (
            f"Set your reorder point at **{rop:,} units**, including a buffer of **{ss:,} units** for safety stock. "
            f"Trigger a replenishment order whenever inventory drops to or below this threshold to protect against demand and lead-time variability. "
            f"This operational policy ensures your target **{sl_pct}** cycle service level is maintained."
        )
    elif m == "FIXED_PERIOD_REVIEW":
        q = getattr(res, "rounded_order_quantity", 0)
        s = getattr(res, "rounded_target_inventory", 0)
        t_unit = getattr(res, "time_unit", "day")
        t = getattr(res, "review_period", 0)
        ss = getattr(res, "rounded_safety_stock", 0)
        pp = getattr(res, "protection_period", 0)
        return (
            f"Order **{q:,} units** to return your inventory position to the order-up-to target of **{s:,} units**. "
            f"Maintain a strict review interval of every **{t:,.0f} {t_unit}s**, holding **{ss:,} units** in buffer stock. "
            f"This policy provides continuous coverage across the entire {pp:.0f}-{t_unit} protection period."
        )
    elif m == "NEWSVENDOR":
        q = getattr(res, "rounded_optimal_quantity", 0)
        cr = getattr(res, "critical_ratio", 0.0)
        cu = getattr(res, "underage_cost", 0.0)
        co = getattr(res, "overage_cost", 0.0)
        ep = getattr(res, "expected_profit", None)
        profit_sentence = f" Your expected seasonal profit under this policy is **${ep:,.2f}**." if ep is not None else " This minimizes expected total overage and underage costs."
        return (
            f"Procure **{q:,} units** for the selling season to satisfy your optimal critical fractile of **{cr * 100:.1f}%**. "
            f"This quantity balances the underage risk of unsatisfied demand (${cu:,.2f} margin loss/unit) against the overage risk of excess inventory (${co:,.2f} markdown/unit)."
            f"{profit_sentence}"
        )
    return ""


# =============================================================================
# STREAMLIT WEB UI IMPLEMENTATION
# =============================================================================

def run_streamlit_app():
    """Streamlit web application interface."""
    import streamlit as st

    st.set_page_config(
        page_title="Supply Chain Inventory Decision Agent",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    agent = InventoryAgent()

    # Theme-aware custom styling (uses Streamlit theme variables, no fixed hex colors)
    st.markdown("""
        <style>
        .pipeline-card {
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
        }
        .metric-badge {
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: 600;
            border: 1px solid var(--text-color);
        }
        /* Prevent truncation in metric cards: wrap text across all metric cards */
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {
            white-space: normal !important;
            word-break: break-word !important;
            overflow-wrap: anywhere !important;
            text-overflow: clip !important;
        }
        /* Apply smaller responsive font (1.15rem) only to the four cards in Execution Pipeline Inspection row */
        .st-key-inspection_pipeline [data-testid="stMetricValue"],
        .st-key-inspection_pipeline [data-testid="stMetricValue"] > div {
            font-size: 1.15rem !important;
            line-height: 1.3 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Native Streamlit headers (theme-aware)
    st.title("📦 Supply Chain Analytics: Inventory Decision Agent")
    st.caption("Automated Decision Support System for Operations Research & Inventory Policy Optimization")

    # Session State Management for running behavior
    if "current_question" not in st.session_state:
        st.session_state["current_question"] = BENCHMARK_SCENARIOS[0]["question"]
    if "query_text_area" not in st.session_state:
        st.session_state["query_text_area"] = BENCHMARK_SCENARIOS[0]["question"]
    if "latest_response" not in st.session_state:
        st.session_state["latest_response"] = None
    if "prev_scenario" not in st.session_state:
        st.session_state["prev_scenario"] = "(Custom Question)"

    # Sidebar: Ordered as Benchmark Scenarios, Data Upload, Supported Models, Architecture Pipeline
    with st.sidebar:
        st.header("⚙️ Agent Controls")

        # 1. Benchmark Scenarios
        st.subheader("📚 Benchmark Scenarios")
        selected_scenario = st.selectbox(
            "Load a pre-configured industry question:",
            ["(Custom Question)"] + [s["name"] for s in BENCHMARK_SCENARIOS],
            index=0,
            key="benchmark_select",
        )

        benchmark_selected = False
        if selected_scenario != st.session_state["prev_scenario"]:
            st.session_state["prev_scenario"] = selected_scenario
            if selected_scenario != "(Custom Question)":
                for s in BENCHMARK_SCENARIOS:
                    if s["name"] == selected_scenario:
                        st.session_state["current_question"] = s["question"]
                        st.session_state["query_text_area"] = s["question"]
                        benchmark_selected = True
                        break

        # 2. Data Upload
        st.divider()
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            help="Upload a CSV file containing: Item Number, Item Name, UoM, Sales ($), Hits (for classification) or Product, Category, Units, Unit Cost (for inventory analysis)",
        )

        # 3. Supported Models (Collapsed expander)
        st.divider()
        with st.expander("🛠️ Supported Models", expanded=False):
            st.markdown("""
            1. **EOQ**: Wilson classic lot sizing
            2. **Quantity Discount**: All-units price break analysis
            3. **Safety Stock & ROP**: Continuous review (s, Q)
            4. **Fixed-Period Review**: Periodic (T, S) base stock
            5. **Newsvendor**: Single-period stochastic inventory
            """)

        # 4. Architecture Pipeline (Collapsed expander)
        with st.expander("⚙️ Architecture Pipeline", expanded=False):
            st.markdown("""
            1. 🔍 **Identify Model**
            2. 📥 **Extract Inputs**
            3. 🐍 **Select Python Tool**
            4. 🧮 **Calculate**
            5. 🛡️ **Verify**
            6. 📢 **Explain**
            """)

    # Main Area
    # -------------------------------------------------------------------------
    # CSV Data Upload & Inventory Analytics Display
    # -------------------------------------------------------------------------
    if uploaded_file is not None:
        st.markdown("---")
        st.header("📁 Uploaded Inventory Data & Analysis")

        df_raw, missing_cols = process_uploaded_csv(uploaded_file)

        if missing_cols:
            st.error(
                f"❌ **Validation Error**: Missing required column(s): **{', '.join(missing_cols)}**.\n\n"
                f"Please ensure your CSV matches either:\n"
                f"- **Inventory Classification**: {', '.join(CLASSIFICATION_REQUIRED_COLUMNS)}\n"
                f"- **Product Inventory**: {', '.join(REQUIRED_CSV_COLUMNS)}"
            )
        elif df_raw is None or df_raw.empty:
            st.warning("⚠️ The uploaded CSV file is empty. Please upload a CSV containing inventory records.")
        else:
            dataset_type = getattr(df_raw, "attrs", {}).get("dataset_type", "inventory")
            if dataset_type == "classification" or ("Hits" in df_raw.columns and "Sales ($)" in df_raw.columns):
                # -------------------------------------------------------------
                # MULTICRITERIA INVENTORY CLASSIFICATION (SALES & HITS)
                # -------------------------------------------------------------
                st.subheader("📋 Dataset Overview")
                col_r, col_c, col_names = st.columns([1, 1, 2])
                with col_r:
                    st.metric("Number of Items", f"{len(df_raw):,}")
                with col_c:
                    st.metric("Number of Columns", f"{len(df_raw.columns):,}")
                with col_names:
                    st.markdown("**Column Names:**")
                    st.write(", ".join(list(df_raw.columns)))

                # Perform multicriteria classification
                cls_results = calculate_multicriteria_inventory_classification(df_raw)
                classified_df = cls_results["dataframe"]
                class_summary_df = cls_results["class_summary_df"]
                total_sales = cls_results["total_sales"]
                total_hits = cls_results["total_hits"]
                total_items = cls_results["total_items"]

                st.subheader("🏷️ Multicriteria Inventory Classification Results (A, B, C, D)")
                st.caption(
                    "Classified using **Sales ($)** (Monetary Value / Revenue Velocity) and "
                    "**Hits** (Picking Frequency / Transaction Velocity)."
                )

                # Summary Metrics
                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.metric("Total Items", f"{total_items:,}")
                with k2:
                    st.metric("Total Sales", f"${total_sales:,.2f}")
                with k3:
                    st.metric("Total Activity (Hits)", f"{total_hits:,.0f} picks")
                with k4:
                    cls_a_sub = class_summary_df[class_summary_df["Final Class"] == "A"]
                    a_items = int(cls_a_sub["Number of Items"].iloc[0]) if not cls_a_sub.empty else 0
                    a_pct = (a_items / total_items * 100.0) if total_items > 0 else 0.0
                    st.metric("Class A Fast Movers", f"{a_items:,} ({a_pct:.1f}%)")

                # Short Description of BAN 517 Methodology
                st.markdown(
                    """
                    **BAN 517 Multicriteria Inventory Classification Methodology:**
                    - **Step 1 (Single-Criterion Pareto Ranking):** Items are sorted from largest to smallest for Sales ($) and Hits separately using a stable sort preserving file order for ties. Each item's class is determined from its exact unrounded cumulative percentage:
                      - **A**: below 60% (Score = 4)
                      - **B**: 60% to below 80% (Score = 3)
                      - **C**: 80% to below 90% (Score = 2)
                      - **D**: 90% to 100% (Score = 1)
                      *(Every item receives a class on both criteria; zero sales or hits are not special cases)*
                    - **Step 2 (Combined Weighted Score):** $\\text{Combined Weighted Score} = 0.65 \\times \\text{Sales Score} + 0.35 \\times \\text{Hits Score}$
                    - **Step 3 (Final Inventory Class):** The combined score is rounded to 2 decimal places before banding:
                      - **A**: 3.25 to 4.00
                      - **B**: 2.50 to 3.24
                      - **C**: 1.75 to 2.49
                      - **D**: below 1.75
                    """
                )

                # Class Breakdown Table
                st.markdown("### 📊 Class Summary Breakdown")
                summary_display_cols = ["Final Class", "Number of Items", "Total Sales ($)"]
                st.dataframe(
                    class_summary_df[summary_display_cols],
                    column_config={
                        "Final Class": st.column_config.TextColumn("Final Class"),
                        "Number of Items": st.column_config.NumberColumn("Number of Items", format="%d"),
                        "Total Sales ($)": st.column_config.NumberColumn("Total Sales ($)", format="$%,.2f"),
                    },
                    hide_index=True,
                    use_container_width=True,
                )

                # Classified Inventory Items Table
                st.markdown("### 📋 Classified Inventory Dataframe")
                col_filt, col_info = st.columns([1, 2])
                with col_filt:
                    filter_class = st.selectbox(
                        "Filter by Final Class:",
                        ["All Items", "Class A", "Class B", "Class C", "Class D"],
                        index=0,
                        key="filter_class_select",
                    )

                display_df = classified_df.copy()
                if filter_class != "All Items":
                    target_cls = filter_class.replace("Class ", "")
                    display_df = display_df[display_df["Final Class"] == target_cls]

                # Select and order columns: original columns + added columns
                standard_cols = [
                    "Item Number",
                    "Item Name",
                    "UoM",
                    "Sales ($)",
                    "Sales Cumulative %",
                    "Sales Class",
                    "Sales Score",
                    "Hits",
                    "Hits Cumulative %",
                    "Hits Class",
                    "Hits Score",
                    "Combined Weighted Score",
                    "Final Class",
                ]
                display_cols = [c for c in standard_cols if c in display_df.columns]
                extra_cols = [c for c in display_df.columns if c not in display_cols and c != "Inventory Class"]
                final_col_order = extra_cols + display_cols

                st.dataframe(
                    display_df[final_col_order],
                    column_config={
                        "Sales ($)": st.column_config.NumberColumn("Sales ($)", format="$%,.2f"),
                        "Sales Cumulative %": st.column_config.NumberColumn("Sales Cumulative %", format="%.2f%%"),
                        "Sales Score": st.column_config.NumberColumn("Sales Score", format="%d"),
                        "Hits": st.column_config.NumberColumn("Hits", format="%d"),
                        "Hits Cumulative %": st.column_config.NumberColumn("Hits Cumulative %", format="%.2f%%"),
                        "Hits Score": st.column_config.NumberColumn("Hits Score", format="%d"),
                        "Combined Weighted Score": st.column_config.NumberColumn("Combined Weighted Score", format="%.2f"),
                        "Final Class": st.column_config.TextColumn("Final Class"),
                    },
                    hide_index=True,
                    use_container_width=True,
                )

                # Download Classified CSV
                download_cols = [c for c in display_df.columns if c != "Inventory Class"]
                csv_bytes = classified_df[download_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Classified Inventory CSV",
                    data=csv_bytes,
                    file_name="classified_inventory_results.csv",
                    mime="text/csv",
                    key="download_classified_csv_btn",
                )
            else:
                # -------------------------------------------------------------
                # LEGACY PRODUCT INVENTORY ANALYSIS
                # -------------------------------------------------------------
                # 1. Dataset Dimensions and Column Names
                st.subheader("📋 Dataset Overview")
                col_r, col_c, col_names = st.columns([1, 1, 2])
                with col_r:
                    st.metric("Number of Rows", f"{len(df_raw):,}")
                with col_c:
                    st.metric("Number of Columns", f"{len(df_raw.columns):,}")
                with col_names:
                    st.markdown("**Column Names:**")
                    st.write(", ".join(list(df_raw.columns)))

                # Display uploaded dataframe
                st.markdown("**Uploaded Dataframe:**")
                st.dataframe(df_raw, use_container_width=True, hide_index=True)

                # 2. Data Analysis Calculations
                metrics = calculate_inventory_dataframe_metrics(df_raw)

                st.subheader("📊 Inventory Data Analysis Results")
                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.metric("Total Units", f"{metrics['total_units']:,.0f}")
                with k2:
                    st.metric("Average Unit Cost", f"${metrics['avg_unit_cost']:,.2f}")
                with k3:
                    st.metric("Product with Highest Units", metrics["highest_units_product"])
                    st.caption(f"{metrics['highest_units_count']:,.0f} units")
                with k4:
                    st.metric("Total Inventory Value", f"${metrics['total_inventory_value']:,.2f}")

                # Total Units by Category
                st.markdown("### 🏷️ Total Units by Category")
                category_df = metrics["category_units_df"]
                st.dataframe(
                    category_df,
                    column_config={
                        "Units": st.column_config.NumberColumn("Total Units", format="%d"),
                    },
                    use_container_width=True,
                    hide_index=True,
                )

                # Dataframe including the new Inventory Value column
                st.markdown("### 💰 Inventory Dataframe (with Inventory Value Column)")
                st.caption("New Calculated Column: `Inventory Value = Units × Unit Cost`")
                df_with_val = metrics["dataframe_with_inventory_value"]
                st.dataframe(
                    df_with_val,
                    column_config={
                        "Unit Cost": st.column_config.NumberColumn("Unit Cost", format="$%.2f"),
                        "Inventory Value": st.column_config.NumberColumn("Inventory Value", format="$%.2f"),
                        "Units": st.column_config.NumberColumn("Units", format="%d"),
                    },
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown("---")

    # -------------------------------------------------------------------------
    # Question Input & Trigger Controls
    # -------------------------------------------------------------------------
    query = st.text_area(
        "Enter your natural language inventory question:",
        height=100,
        placeholder="e.g., Annual demand is 5000, setup cost is $40, supplier price breaks are...",
        key="query_text_area",
    )
    st.session_state["current_question"] = query

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        run_btn = st.button("🚀 Analyze & Optimize", type="primary", use_container_width=True)

    # Execute analysis only when requested (button click or new benchmark selected)
    if run_btn or benchmark_selected:
        with st.spinner("Executing 6-Stage Inventory Pipeline..."):
            st.session_state["latest_response"] = agent.process_question(st.session_state["current_question"])

    resp: Optional[AgentResponse] = st.session_state.get("latest_response")

    # -------------------------------------------------------------------------
    # Results Display
    # -------------------------------------------------------------------------
    if resp is not None:
        # Status Message 1: Unused values yellow warning banner above results
        if resp.unused_values:
            unused_items = []
            for u in resp.unused_values:
                if hasattr(u, "context") and u.context:
                    val_repr = getattr(u, "_int_or_float", u)
                    unused_items.append(f"{val_repr} (in '{u.context}')")
                else:
                    unused_items.append(str(u))
            st.warning(f"⚠️ **Warning: these values from your question were not used in the calculation:** {escape_dollars(', '.join(unused_items))}")

        # Status Message 2: Missing inputs error message (no result cards)
        if resp.missing_inputs:
            st.error(
                f"⚠️ **Clarification Needed for {resp.model_name} Model**\n\n"
                f"I identified that your question relates to **{resp.model_name}** "
                f"(Match Strength: {resp.confidence * 100:.0f}%), but the following required parameter(s) are missing:\n\n"
                + "\n".join([f"- **{m}**" for m in resp.missing_inputs]) + "\n\n"
                f"Please provide these values so I can execute the exact calculation."
            )
            if resp.extracted_inputs:
                with st.expander("📥 Parameters Extracted So Far", expanded=True):
                    st.json(resp.extracted_inputs)
            return

        # Status Message 3: Unsupported model info message (no result cards)
        if resp.model_name in ("unsupported", InventoryAgent.MODEL_UNSUPPORTED):
            st.info(
                "❓ **Unsupported Question**\n\n"
                "I could not match your question to any of the 5 supported inventory decision models. "
                "The supported inventory models are:\n\n"
                "1. **Economic Order Quantity (EOQ)**: Classical Wilson lot-sizing for constant demand under ordering/setup and holding cost trade-offs.\n"
                "2. **Quantity Discount**: All-units price-break analysis evaluating feasible batch sizes and total annual acquisition and inventory costs.\n"
                "3. **Safety Stock and Reorder Point (ROP)**: Continuous review (s, Q) replenishment policy under demand and/or lead-time variability.\n"
                "4. **Fixed-Period Review**: Periodic review (T, S) order-up-to base-stock replenishment policy for fixed review intervals.\n"
                "5. **Newsvendor**: Single-period stochastic inventory optimization for seasonal or perishable products with underage and overage risks.\n\n"
                "Please rephrase your question using relevant parameters from one of these models."
            )
            return

        # Pipeline status cards (Match Strength renamed with tooltip)
        st.subheader("🔍 Execution Pipeline Inspection")
        with st.container(key="inspection_pipeline"):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                display_model_name = MODEL_DISPLAY_NAMES.get(resp.model_name, resp.model_name)
                st.metric("Identified Model", display_model_name)
            with c2:
                st.metric(
                    "Match Strength",
                    f"{resp.confidence * 100:.0f}%",
                    help="How strongly the wording matched this model's keywords. Not a probability.",
                )
            with c3:
                st.metric("Python Tool", resp.tool_name)
            with c4:
                status_label = "✅ Verified" if resp.verification.passed else "⚠️ Attention"
                st.metric("Verification", status_label)

        # Four distinct tabs without duplication
        tab_summary, tab_student, tab_checks, tab_inputs = st.tabs([
            "Summary",
            "Step-by-Step Verification",
            "Checks",
            "Inputs",
        ])

        with tab_summary:
            # Answer first: Key results as large metric cards
            st.subheader("🎯 Key Decision Results")
            metric_cards = get_model_metric_cards(resp)
            if metric_cards:
                mc_cols = st.columns(len(metric_cards))
                for idx, (lbl, val) in enumerate(metric_cards):
                    with mc_cols[idx]:
                        st.metric(lbl, escape_dollars(val))

            # Two or three sentence plain-language recommendation
            st.markdown("### 💡 Plain-Language Recommendation")
            recommendation = get_plain_language_recommendation(resp)
            st.info(escape_dollars(recommendation))

            # One chart per model in Summary tab built from calculation tool values
            chart_tuple = build_model_chart(resp)
            if chart_tuple is not None:
                chart_obj, chart_caption, opt_table_df = chart_tuple
                st.markdown("### 📈 Visual Decision Analysis")
                st.altair_chart(chart_obj, use_container_width=True, theme="streamlit")
                st.caption(escape_dollars(chart_caption))
                if opt_table_df is not None:
                    st.markdown("#### 📋 Price Break Tier Comparison Table")
                    st.dataframe(opt_table_df, use_container_width=True, hide_index=True)

            # Download report button: downloads full report as Markdown named after model and date
            st.markdown("---")
            model_slug = resp.model_name.lower().replace(" ", "_").replace("/", "_")
            date_str = datetime.now().strftime("%Y-%m-%d")
            report_filename = f"{model_slug}_report_{date_str}.md"
            st.download_button(
                label="📥 Download Report",
                data=resp.explanation,
                file_name=report_filename,
                mime="text/markdown",
                use_container_width=False,
                key="download_report_btn",
            )

        with tab_student:
            st.markdown("## 🎓 Step-by-Step Verification")
            st.caption("All calculation steps below originate deterministically from Python operations research tools.")
            report_lines = [
                line for line in agent._format_verification_report(resp.model_name, resp.calculation_result, resp.verification)
                if "Student Verification Report" not in line
            ]
            report_md = "\n".join(report_lines)
            st.markdown(escape_dollars(report_md))

        with tab_checks:
            st.subheader("🛡️ Verification Checks & Mathematical Validation")
            checks_list = getattr(resp.calculation_result, "verification_checks", []) or resp.verification.checks
            for chk in checks_list:
                chk_str = escape_dollars(str(chk))
                if "FAIL" in str(chk):
                    st.error(f"❌ {chk_str}")
                else:
                    st.success(f"✅ {chk_str}")

            if resp.verification.notes:
                st.markdown("### ℹ️ Analytical Notes")
                for note in resp.verification.notes:
                    st.info(escape_dollars(str(note)))

            st.markdown("### 📋 Model Assumptions")
            for idx, asm in enumerate(resp.assumptions, start=1):
                st.markdown(f"{idx}. {escape_dollars(str(asm))}")

        with tab_inputs:
            st.subheader("📥 Extracted Parameters")
            filtered_inputs = remove_null_values(resp.inputs_extracted)
            st.json(filtered_inputs)

            conv_notes = [n for n in getattr(resp.verification, "notes", []) if "converted" in str(n).lower() or "risk" in str(n).lower()]
            if conv_notes:
                st.markdown("### 🔄 Parameter / Time-Unit Conversions")
                for cn in conv_notes:
                    st.info(escape_dollars(cn))


# =============================================================================
# MAIN ENTRYPOINT (Dispatches to Streamlit or CLI)
# =============================================================================

def is_running_under_streamlit() -> bool:
    """Detect if running inside streamlit CLI (`streamlit run app.py`)."""
    try:
        from streamlit import runtime
        return runtime.exists()
    except ImportError:
        return False


def main():
    agent = InventoryAgent()

    parser = argparse.ArgumentParser(description="Supply Chain Analytics: Inventory Decision Agent")
    parser.add_argument("--cli", action="store_true", help="Launch interactive CLI prompt")
    parser.add_argument("--question", type=str, help="Single natural language question to process")
    parser.add_argument("--demo", action="store_true", help="Run all 5 benchmark model questions")

    args, unknown = parser.parse_known_args()

    # If running with `streamlit run app.py`, launch Streamlit Web UI
    if is_running_under_streamlit():
        run_streamlit_app()
        return

    # If explicit CLI flags or questions passed
    if args.question:
        run_cli_single(agent, args.question)
    elif args.demo:
        for scenario in BENCHMARK_SCENARIOS:
            print(f"\n==================== {scenario['name']} ====================")
            run_cli_single(agent, scenario["question"])
    else:
        # Default terminal behavior when run via `python3 app.py`
        # If user passed no flags, check if streamlit is available, otherwise default to CLI interactive
        try:
            import streamlit  # noqa: F401
            # If user ran `python3 app.py`, provide helpful message and run CLI interactive
            print("\n💡 Tip: To launch the web UI, run: streamlit run app.py\n")
            run_cli_interactive(agent)
        except ImportError:
            run_cli_interactive(agent)


if __name__ == "__main__":
    main()
