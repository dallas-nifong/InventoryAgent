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
import json
import sys
from typing import Any, Dict, List

from inventory_agent import AgentResponse, InventoryAgent


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

    # Custom styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #4B5563;
            margin-bottom: 1.5rem;
        }
        .pipeline-card {
            background-color: #F3F4F6;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
        }
        .metric-badge {
            background-color: #DBEAFE;
            color: #1E40AF;
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">📦 Supply Chain Analytics: Inventory Decision Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Automated Decision Support System for Operations Research & Inventory Policy Optimization</div>',
        unsafe_allow_html=True,
    )

    # Sidebar: Model Reference & Presets
    with st.sidebar:
        st.header("⚙️ Agent Controls")
        st.markdown("**Architecture Pipeline:**")
        st.markdown("""
        1. 🔍 **Identify Model**
        2. 📥 **Extract Inputs**
        3. 🐍 **Select Python Tool**
        4. 🧮 **Calculate**
        5. 🛡️ **Verify**
        6. 📢 **Explain**
        """)

        st.divider()
        st.subheader("📚 Benchmark Scenarios")
        selected_scenario = st.selectbox(
            "Load a pre-configured industry question:",
            ["(Custom Question)"] + [s["name"] for s in BENCHMARK_SCENARIOS],
        )

        preset_text = ""
        if selected_scenario != "(Custom Question)":
            for s in BENCHMARK_SCENARIOS:
                if s["name"] == selected_scenario:
                    preset_text = s["question"]

        st.divider()
        st.markdown("### 🛠️ Supported Inventory Models")
        st.caption("1. **EOQ**: Wilson classic lot sizing")
        st.caption("2. **Quantity Discount**: All-units price break analysis")
        st.caption("3. **Safety Stock & ROP**: Continuous review (s, Q)")
        st.caption("4. **Fixed-Period Review**: Periodic (T, S) base stock")
        st.caption("5. **Newsvendor**: Single-period stochastic inventory")

    # Main Area
    default_val = preset_text if preset_text else "Our annual demand is 10,000 units, ordering cost is $50, and holding cost is $2 per unit per year. What is our EOQ?"
    query = st.text_area(
        "Enter your natural language inventory question:",
        value=default_val,
        height=100,
        placeholder="e.g., Annual demand is 5000, setup cost is $40, supplier price breaks are...",
    )

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        run_btn = st.button("🚀 Analyze & Optimize", type="primary", use_container_width=True)

    if run_btn or query:
        with st.spinner("Executing 6-Stage Inventory Pipeline..."):
            resp: AgentResponse = agent.process_question(query)

        # Pipeline status cards
        st.subheader("🔍 Execution Pipeline Inspection")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Identified Model", resp.model_name)
        with c2:
            st.metric("Classification Confidence", f"{resp.confidence * 100:.0f}%")
        with c3:
            st.metric("Python Tool", resp.tool_name)
        with c4:
            status_label = "✅ Verified" if resp.verification.passed else "⚠️ Attention"
            st.metric("Verification", status_label)

        # Tabbed details: Decision Memo, 8-Step Student Verification, Sanity Checks, Parameters
        tab_memo, tab_student, tab_verif, tab_inputs = st.tabs([
            "📋 Full Decision Memo",
            "🎓 8-Step Student Verification",
            "🛡️ Mathematical Sanity Checks",
            "📥 Extracted Parameters & Intermediates",
        ])

        with tab_memo:
            st.markdown(resp.explanation)

        with tab_student:
            st.markdown("## 🎓 Step-by-Step Student Verification Engine")
            st.caption("All calculation steps below originate deterministically from Python operations research tools.")

            # 1. Inputs Extracted
            st.markdown("### 1. Inputs Extracted")
            st.json(resp.inputs_extracted)

            # 2. Variable Definitions
            st.markdown("### 2. Variable Definitions")
            for var, defn in resp.variable_definitions.items():
                st.markdown(f"- **`{var}`**: {defn}")

            # 3. Formula
            st.markdown("### 3. Formula")
            for fname, fexpr in resp.formula.items():
                st.markdown(f"- **{fname.replace('_', ' ').title()}**: `{fexpr}`")

            # 4. Formula with Substituted Values
            st.markdown("### 4. Formula with Substituted Values")
            for sname, sexpr in resp.formula_with_substituted_values.items():
                st.info(f"**{sname.replace('_', ' ').title()}**:\n\n{sexpr}")

            # 5. Intermediate Calculations
            st.markdown("### 5. Intermediate Calculations")
            st.json(resp.intermediate_calculations)

            # 6. Final Answer
            st.markdown("### 6. Final Answer")
            st.success(f"**Final Answer Output**: `{resp.final_answer}`")

            # 7. Verification Checks
            st.markdown("### 7. Verification Checks")
            for chk in resp.verification_checks:
                st.markdown(f"- [x] {chk}")

            # 8. Assumptions
            st.markdown("### 8. Assumptions")
            for idx, asm in enumerate(resp.assumptions, start=1):
                st.markdown(f"{idx}. {asm}")

        with tab_verif:
            st.markdown("### Validation and Mathematical Sanity Checks")
            if resp.verification.checks:
                for chk in resp.verification.checks:
                    st.success(f"✔ {chk}")
            if resp.verification.notes:
                for note in resp.verification.notes:
                    st.warning(f"ℹ️ {note}")

        with tab_inputs:
            st.markdown("### Model Input Extraction & Intermediates")
            st.json({
                "inputs_extracted": resp.inputs_extracted,
                "intermediate_calculations": resp.intermediate_calculations,
                "final_answer": resp.final_answer,
            })
            if resp.missing_inputs:
                st.error(f"Missing parameters required for full execution: {resp.missing_inputs}")


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
