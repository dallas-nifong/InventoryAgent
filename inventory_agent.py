"""
Supply Chain Analytics Inventory Decision Agent.

Implements the 6-stage architecture:
Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain

Models supported:
1. Economic Order Quantity (EOQ)
2. Quantity Discount
3. Safety Stock & Reorder Point (ROP)
4. Fixed-Period Review ((T, S) Order-Up-To Level)
5. Newsvendor Model (Single-Period Stochastic Inventory)
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import inventory_tools as it


# ---------------------------------------------------------------------------
# Data Classes for Agent Pipeline
# ---------------------------------------------------------------------------

class UnusedNumber(float):
    """
    Represents a numeric value appearing in the question that was not used in the calculation.
    Subclasses float so it transparently supports numeric comparisons (e.g. 3 in unused_values,
    unused_values == [3], unused_values == [3.0]), string matching, dict matching,
    while storing raw text and context snippet around the number.
    """
    def __new__(cls, value: Union[int, float], raw_text: str = "", context: str = ""):
        instance = super().__new__(cls, float(value))
        instance.raw_text = raw_text
        instance.context = context
        instance._int_or_float = int(value) if float(value).is_integer() else float(value)
        return instance

    def __repr__(self) -> str:
        return f"{self._int_or_float} (in '{self.context}')" if self.context else str(self._int_or_float)

    def __str__(self) -> str:
        return str(self._int_or_float)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, (int, float)):
            return float(self) == float(other)
        if isinstance(other, str):
            try:
                return float(self) == float(other)
            except ValueError:
                return str(self._int_or_float) == other or self.raw_text == other or str(self) == other
        if isinstance(other, dict):
            return float(self) == float(other.get("value", 0))
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(float(self))


@dataclass
class VerificationResult:
    """Outcome of domain and sanity checks on calculations."""
    passed: bool
    checks: List[str]
    notes: List[str] = field(default_factory=list)


@dataclass
class AgentResponse:
    """Full end-to-end response from the Inventory Decision Agent."""
    question: str
    model_name: str
    confidence: float
    extracted_inputs: Dict[str, Any]
    missing_inputs: List[str]
    tool_name: str
    calculation_result: Optional[Any]
    verification: VerificationResult
    explanation: str
    unused_values: List[Any] = field(default_factory=list)

    @property
    def inputs_extracted(self) -> Dict[str, Any]:
        """1. Inputs Extracted directly from the deterministic tool calculation."""
        if self.calculation_result and hasattr(self.calculation_result, "inputs_extracted"):
            return self.calculation_result.inputs_extracted
        return self.extracted_inputs

    @property
    def variable_definitions(self) -> Dict[str, str]:
        """2. Variable Definitions directly from the deterministic tool calculation."""
        if self.calculation_result and hasattr(self.calculation_result, "variable_definitions"):
            return self.calculation_result.variable_definitions
        return {}

    @property
    def formula(self) -> Dict[str, str]:
        """3. Formula directly from the deterministic tool calculation."""
        if self.calculation_result and hasattr(self.calculation_result, "formula"):
            return self.calculation_result.formula
        return {}

    @property
    def formula_with_substituted_values(self) -> Dict[str, str]:
        """4. Formula with Substituted Values directly from the deterministic tool calculation."""
        if self.calculation_result and hasattr(self.calculation_result, "formula_with_substituted_values"):
            return self.calculation_result.formula_with_substituted_values
        return {}

    @property
    def intermediate_calculations(self) -> Dict[str, Any]:
        """5. Intermediate Calculations directly from the deterministic tool calculation."""
        if self.calculation_result and hasattr(self.calculation_result, "intermediate_calculations"):
            return self.calculation_result.intermediate_calculations
        return {}

    @property
    def final_answer(self) -> Any:
        """6. Final Answer directly from the deterministic tool calculation."""
        if self.calculation_result and hasattr(self.calculation_result, "final_answer"):
            return self.calculation_result.final_answer
        return None

    @property
    def verification_checks(self) -> List[str]:
        """7. Verification Checks directly from the deterministic tool calculation."""
        if self.calculation_result and hasattr(self.calculation_result, "verification_checks"):
            return self.calculation_result.verification_checks
        return self.verification.checks

    @property
    def assumptions(self) -> List[str]:
        """8. Assumptions directly from the deterministic tool calculation."""
        if self.calculation_result and hasattr(self.calculation_result, "assumptions"):
            return self.calculation_result.assumptions
        return []

    def to_dict(self) -> Dict[str, Any]:
        calc_dict = (
            self.calculation_result.to_dict()
            if hasattr(self.calculation_result, "to_dict")
            else str(self.calculation_result)
        )
        return {
            "question": self.question,
            "model_name": self.model_name,
            "confidence": self.confidence,
            "extracted_inputs": self.extracted_inputs,
            "missing_inputs": self.missing_inputs,
            "tool_name": self.tool_name,
            "calculation_result": calc_dict,
            "verification": asdict(self.verification),
            "explanation": self.explanation,
            "unused_values": self.unused_values,
            "inputs_extracted": self.inputs_extracted,
            "variable_definitions": self.variable_definitions,
            "formula": self.formula,
            "formula_with_substituted_values": self.formula_with_substituted_values,
            "intermediate_calculations": self.intermediate_calculations,
            "final_answer": self.final_answer,
            "verification_checks": self.verification_checks,
            "assumptions": self.assumptions,
        }


# ---------------------------------------------------------------------------
# Core Agent Implementation
# ---------------------------------------------------------------------------

class InventoryAgent:
    """
    Intelligent Agent for Supply Chain Inventory Decision Making.
    Executes the deterministic 6-stage analytical pipeline on natural language queries.
    """

    MODEL_EOQ = "EOQ"
    MODEL_QUANTITY_DISCOUNT = "QUANTITY_DISCOUNT"
    MODEL_SAFETY_STOCK_ROP = "SAFETY_STOCK_ROP"
    MODEL_FIXED_PERIOD_REVIEW = "FIXED_PERIOD_REVIEW"
    MODEL_NEWSVENDOR = "NEWSVENDOR"
    MODEL_UNSUPPORTED = "unsupported"

    SUPPORTED_MODELS = [
        MODEL_EOQ,
        MODEL_QUANTITY_DISCOUNT,
        MODEL_SAFETY_STOCK_ROP,
        MODEL_FIXED_PERIOD_REVIEW,
        MODEL_NEWSVENDOR,
    ]

    def __init__(self):
        pass

    # =========================================================================
    # STAGE 1: Identify Model
    # =========================================================================

    def identify_model(self, question: str) -> Tuple[str, float]:
        """
        Analyze customer question text and classify into one of the 5 inventory models,
        or 'unsupported' if the question does not match any of the five models.

        Returns:
            Tuple of (model_name, confidence_score)
        """
        q = question.lower()

        scores: Dict[str, float] = {
            self.MODEL_EOQ: 0.0,
            self.MODEL_QUANTITY_DISCOUNT: 0.0,
            self.MODEL_SAFETY_STOCK_ROP: 0.0,
            self.MODEL_FIXED_PERIOD_REVIEW: 0.0,
            self.MODEL_NEWSVENDOR: 0.0,
        }

        # 1. Quantity Discount Cues
        qd_keywords = [
            "discount", "price break", "volume pricing", "bulk discount",
            "tier", "tiers", "schedule", "all-units", "price bracket",
            "order over", "order at least", "unit price if", "price per unit for",
            "price schedule", "price breaks", "take the discount",
        ]
        for kw in qd_keywords:
            if kw in q:
                scores[self.MODEL_QUANTITY_DISCOUNT] += 2.5
        # Check for range/price patterns like "0-999: $20", "under 500", or "100 or more cases $16"
        if (
            re.search(r"\b\d+\s*[-–to]+\s*\d+\s*[:@=]?\s*\$?\d+", q)
            or re.search(r"\$\d+\s*for\s*\d+", q)
            or re.search(r"\d+\s+or\s+more\s+(?:cases|units|items|boxes|cartons)?\s*\$?\d+", q)
            or re.search(r"\bunder\s+\d+\s+(?:units|cases|items)?\s+at\s+\$?\d+", q)
        ):
            scores[self.MODEL_QUANTITY_DISCOUNT] += 5.0

        # Gap fix (Item 10): A question mentioning a discount or lower price for buying at least some quantity is QUANTITY_DISCOUNT
        has_qd_price_break = bool(
            re.search(r"\b(?:lower\s+price|drops?\s+the\s+price|discount(?:ed)?\s+price|price\s+drop|take\s+the\s+discount)\b", q)
            or re.search(r"\b(?:if\s+we\s+buy|if\s+we\s+order|when\s+ordering|order\s+at\s+least|buy)\s+\d+\s*(?:or\s+more)?\b", q)
            or ("discount" in q and not any(k in q for k in ["salvage", "liquidat", "unsold", "markdown"]))
        )
        if has_qd_price_break:
            scores[self.MODEL_QUANTITY_DISCOUNT] += 8.0

        # 2. Fixed-Period Review Cues
        # Must win over Safety Stock / Reorder Point whenever reviewing or ordering at a fixed interval
        # Gap fix (Item 10): Stock is counted, reviewed, or ordered every N days or weeks in any word order
        has_review_interval = bool(
            re.search(r"\breview\s+(?:the\s+)?inventory\s+every\s+\d+", q)
            or re.search(r"\binventory\s+is\s+reviewed\s+every\s+\d+", q)
            or re.search(r"\b(?:we|they|company)?\s*review\s+every\s+\d+", q)
            or re.search(r"\b(?:we|they|company)\s+review\s+(?:inventory\s+)?every\b", q)
            or re.search(r"\border\s+every\s+\d+", q)
            or re.search(r"\breview\s+(?:period|interval|frequency|schedule)\b", q)
            or re.search(r"\border\s+(?:interval|frequency)\b", q)
            or re.search(r"\bperiodic\s+review\b", q)
            or re.search(r"\border[- ]up[- ]to\b", q)
            or re.search(r"\breview(?:ed)?\s+every\s+\d+\s*(?:days?|weeks?|months?)", q)
            or re.search(r"\b\(t,\s*s\)\b", q)
            or re.search(r"\breview\b.*\bevery\s+\d+\s*(?:days?|weeks?|months?)", q)
            or re.search(r"\bevery\s+\d+\s*(?:days?|weeks?|months?)\b.*?\b(?:count(?:\s+stock)?|review|order|place\s+an\s+order)\b", q)
            or re.search(r"\b(?:count(?:\s+stock)?|stock\s+is\s+counted|inventory\s+is\s+counted|review|order|place\s+an\s+order)\b.*?\bevery\s+\d+\s*(?:days?|weeks?|months?)\b", q)
            or re.search(r"\bcount\s+stock\b", q)
        )

        # Benchmark / prompt rule: Fixed-Period Review identification
        # A question that states an "order interval" or "review period" together with a lead time,
        # or that gives inventory on hand and asks for the amount to order, is FIXED_PERIOD_REVIEW, not Safety Stock / ROP.
        has_fpr_rule = bool(
            (
                ("order interval" in q or "review period" in q or "review interval" in q)
                and ("lead time" in q or "lead-time" in q or "delivery time" in q)
            )
            or (
                re.search(r"\b(?:inventory\s+on\s+hand|on[- ]hand(?:\s+inventory)?|stock\s+on\s+hand)\b", q)
                and re.search(r"\b(?:amount\s+to\s+order|how\s+much\s+to\s+order|how\s+many\s+(?:units\s+)?to\s+order|order\s+quantity)\b", q)
            )
        )

        fp_keywords = [
            "fixed period", "periodic review", "order up to", "order-up-to",
            "(t, s)", "t,s", "review period", "review interval", "inventory position",
            "base stock", "target inventory", "replenishment interval", "order interval",
        ]
        for kw in fp_keywords:
            if kw in q:
                scores[self.MODEL_FIXED_PERIOD_REVIEW] += 3.5

        if has_review_interval:
            scores[self.MODEL_FIXED_PERIOD_REVIEW] += 8.0
            # If question gives both a review interval and a lead time, it is unambiguously Fixed-Period Review
            if "lead time" in q or "lead-time" in q or "delivery time" in q:
                scores[self.MODEL_FIXED_PERIOD_REVIEW] += 4.0

        if has_fpr_rule:
            scores[self.MODEL_FIXED_PERIOD_REVIEW] += 15.0
            scores[self.MODEL_SAFETY_STOCK_ROP] = 0.0
            scores[self.MODEL_EOQ] = 0.0

        # 3. Newsvendor Cues (recognized with or without the word 'salvage')
        nv_keywords = [
            "newsvendor", "newsboy", "newsstand", "newspaper", "newspapers", "single period", "salvage", "salvages",
            "perishable", "spoilage", "seasonal", "fashion item", "leftover",
            "critical ratio", "critical fractile", "underage", "overage",
            "shortage penalty", "clearance price", "scrap value", "one-time buy",
            "underage cost", "overage cost", "shortage cost", "excess cost",
            "liquidated", "liquidate", "markdown", "marked down", "discarded",
            "donated", "recycled", "recycle", "returned", "return",
        ]
        for kw in nv_keywords:
            if kw in q:
                scores[self.MODEL_NEWSVENDOR] += 3.0

        has_selling_price = bool(
            re.search(r"\bsells?\s+(?:[a-zA-Z]+\s+){0,3}(?:for|at)\s*\$?\d+", q)
            or re.search(r"\bsold\s+(?:for|at)\s*\$?\d+", q)
            or "selling price" in q
            or "retail price" in q
            or "revenue per unit" in q
        )
        has_unit_cost = bool(
            re.search(r"\bcosts?\s*\$?\d+", q)
            or re.search(r"\b(?:buys?|purchases?)\s+(?:[a-zA-Z]+\s+){0,3}(?:for|at)\s*\$?\d+", q)
            or re.search(r"\bcosts?\s*\$?\d+\s+to\s+(?:make|produce|manufacture|buy)\b", q)
            or "wholesale" in q
            or "purchase cost" in q
            or "unit cost" in q
            or "procurement cost" in q
            or "cost to make" in q
            or "cost to produce" in q
        )
        has_unsold_or_leftover = bool(
            re.search(r"\b(?:unsold|leftover|left-over|excess|remaining)\b", q)
            or re.search(r"\b(?:salvage[sd]?|liquidat(?:ed|ion)?|markdown|marked\s*down|discarded|donated|scrap(?:ped)?|worthless|recycled?|returned?)\b", q)
            or re.search(r"\bsold\s+to\s+[^.]+?\s+for\s*\$?\d+", q)
        )
        has_disposition = bool(
            re.search(
                r"\b(?:unsold|leftover|excess|remaining)\b.*\b(?:liquidat(?:ed|ion)?|marked\s*down|markdown|discounted|donated|discarded|salvage[sd]?|scrap(?:ped)?|worthless|recycled?|returned?)\b",
                q,
            )
            or re.search(
                r"\b(?:liquidat(?:ed|ion)?|marked\s*down|salvage[sd]?|scrap(?:ped)?|recycled?|returned?)\s*(?:for|at)\s*\$?\d+",
                q,
            )
            or re.search(r"\bsold\s+to\s+[^.]+?\s+for\s*\$?\d+", q)
        )
        has_single_season = bool(
            re.search(r"\b(?:seasonal|season|winter|summer|holiday|event|single\s+period|perishable)\b", q)
        )
        has_demand_dist = bool(
            re.search(
                r"\b(?:normally\s+distributed|normal\b|mean\s*(?:of\s*|=)?\s*[\d,]+|standard\s+deviation\s*(?:of\s*|=)?\s*[\d,]+|demand\s+table)\b",
                q,
            )
        )

        if has_selling_price and has_unit_cost:
            if has_single_season or has_disposition or has_demand_dist or has_unsold_or_leftover:
                scores[self.MODEL_NEWSVENDOR] += 6.0

        if has_disposition or has_unsold_or_leftover:
            scores[self.MODEL_NEWSVENDOR] += 5.0

        # Item 6: A question with a purchase cost, a selling price, and unsold or leftover units must be identified as NEWSVENDOR.
        if has_selling_price and has_unit_cost and has_unsold_or_leftover:
            scores[self.MODEL_NEWSVENDOR] += 12.0
            scores[self.MODEL_EOQ] = 0.0
            scores[self.MODEL_QUANTITY_DISCOUNT] = 0.0
            scores[self.MODEL_SAFETY_STOCK_ROP] = 0.0
            scores[self.MODEL_FIXED_PERIOD_REVIEW] = 0.0

        if "underage cost" in q or "overage cost" in q or "critical ratio" in q:
            scores[self.MODEL_NEWSVENDOR] += 6.0

        # 4. Safety Stock & Reorder Point Cues
        # The presence of 'lead time' or 'service level' must not pull periodic review questions to ROP
        if not has_review_interval:
            ss_keywords = [
                "safety stock", "reorder point", "rop", "service level",
                "cycle service level", "csl", "stockout risk", "buffer stock",
                "lead time demand", "standard deviation of lead time",
                "lead time variability", "risk of stockout", "running out",
            ]
            for kw in ss_keywords:
                if kw in q:
                    scores[self.MODEL_SAFETY_STOCK_ROP] += 2.5
            if "reorder point" in q or "rop" in q or "safety stock" in q:
                scores[self.MODEL_SAFETY_STOCK_ROP] += 4.0
            if ("lead time" in q or "lead-time" in q) and ("order cost" not in q and "ordering cost" not in q and "holding cost" not in q):
                scores[self.MODEL_SAFETY_STOCK_ROP] += 3.0
        else:
            scores[self.MODEL_SAFETY_STOCK_ROP] = 0.0

        # 5. EOQ Cues (recognized in standard and conversational wording)
        eoq_keywords = [
            "eoq", "economic order quantity", "annual demand", "setup cost",
            "order cost", "ordering cost", "setup expense", "cost per order",
            "cost of placing an order", "ordering/setup cost", "holding cost",
            "carrying cost", "wilson formula", "cycle inventory",
        ]
        for kw in eoq_keywords:
            if kw in q:
                scores[self.MODEL_EOQ] += 2.0

        has_conversational_order_cost = bool(
            re.search(r"\b(?:each\s+order\s+costs?|costs?\s+\$?\d+(?:\.\d+)?\s+to\s+place|order\s+costs?\s+\$?\d+)\b", q)
            or re.search(r"\bcost\s+of\s+placing\s+an\s+order\b", q)
            or re.search(r"\bcost\s+per\s+order\b", q)
        )
        has_conversational_holding_cost = bool(
            re.search(r"\bcosts?\s+\$?\d+(?:\.\d+)?\s+to\s+hold(?:\s+one\s+unit)?(?:\s+for\s+a\s+year)?\b", q)
            or re.search(r"\bto\s+hold\s+(?:one|a|each)\s+(?:unit|case|item|box)\s+(?:for\s+a|per)\s+year\b", q)
            or re.search(r"\bcosts?\s+\$?\d+(?:\.\d+)?\s*(?:a|per)\s+year\s+to\s+hold\b", q)
            or re.search(r"\bto\s+carry\s+(?:one|each|a)\s+(?:unit|case|item|box)\s*(?:for\s+a|per)\s*year\b", q)
            or re.search(r"\b\$?\d+(?:\.\d+)?\s*per\s+(?:case|unit|item|box)\s*(?:per|a)\s*year\b", q)
            or re.search(r"\bper\s+(?:case|unit|item|box)\s+per\s+year\b", q)
            or re.search(r"\bto\s+hold\s+inventory\b", q)
        )
        has_conversational_demand = bool(
            re.search(r"\b(?:sell|use|demand|sales?)\s+(?:about|approx(?:imately)?)?\s*[\d,]+(?:\.\d+)?\s*(?:units|cases|boxes|items)?\s*(?:a|per)\s*year\b", q)
            or re.search(r"\b[\d,]+(?:\.\d+)?\s*(?:units|cases|boxes|items)?\s*(?:a|per)\s*year\b", q)
        )
        has_eoq_question = bool(
            re.search(r"\b(?:what\s+should\s+(?:our|the)\s+order\s+quantity\s+be|what\s+is\s+(?:our|the)\s+optimal\s+order\s+quantity|optimal\s+order\s+size|how\s+many\s+units\s+should\s+we\s+order)\b", q)
        )

        if has_conversational_order_cost:
            scores[self.MODEL_EOQ] += 4.0
        if has_conversational_holding_cost:
            scores[self.MODEL_EOQ] += 4.0
        if has_conversational_demand:
            scores[self.MODEL_EOQ] += 3.0
        if has_eoq_question:
            scores[self.MODEL_EOQ] += 2.0

        if "eoq" in q or "economic order quantity" in q:
            scores[self.MODEL_EOQ] += 5.0

        # Disqualification rule:
        # A question with a selling price and purchase cost and no ordering or holding cost is not EOQ.
        has_ordering_or_holding = (
            has_conversational_order_cost
            or has_conversational_holding_cost
            or any(
                k in q
                for k in [
                    "order cost", "ordering cost", "setup cost", "holding cost",
                    "carrying cost", "cost per order", "setup expense",
                ]
            )
        )
        if has_selling_price and has_unit_cost and not has_ordering_or_holding:
            scores[self.MODEL_EOQ] = 0.0

        # Disqualification rule: Quantity Discount overrides pure EOQ when price breaks or discounts are present
        if scores[self.MODEL_QUANTITY_DISCOUNT] >= 4.0 or has_qd_price_break:
            scores[self.MODEL_EOQ] = 0.0

        # Disqualification rule: Fixed-Period Review overrides continuous ROP
        if has_review_interval or has_fpr_rule:
            scores[self.MODEL_SAFETY_STOCK_ROP] = 0.0

        best_model = max(scores, key=lambda k: scores[k])
        max_score = scores[best_model]

        # Rule: Questions that match none of the five models must not default to EOQ
        if max_score <= 1.0:
            return self.MODEL_UNSUPPORTED, 0.0

        confidence = min(0.99, max(0.45, max_score / (max_score + 2.0)))
        return best_model, round(confidence, 2)

    # =========================================================================
    # STAGE 2: Extract Inputs
    # =========================================================================

    def _clean_number(self, val_str: Optional[str]) -> Optional[float]:
        """Strip currency symbols, commas, trailing punctuation, and whitespace, then convert to float."""
        if not val_str or "%" in val_str:
            return None
        cleaned = re.sub(r"[\$\s]", "", val_str).rstrip(",;:.")
        cleaned = cleaned.replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            return None

    def _extract_number(self, text: str, patterns: List[str]) -> Optional[float]:
        """Helper to extract a float from text using regex patterns. Rejects numbers followed by %."""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                for group_idx, group in enumerate(match.groups(), 1):
                    if group is not None:
                        end_pos = match.end(group_idx)
                        if end_pos < len(text) and text[end_pos:].lstrip().startswith("%"):
                            continue  # A number followed by % is never a dollar amount or standard float
                        val = self._clean_number(group)
                        if val is not None:
                            return val
        return None

    def _extract_percentage(self, text: str, patterns: List[str]) -> Optional[float]:
        """Helper to extract percentage or fraction (e.g., 95% -> 0.95)."""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                for group in match.groups():
                    if group is not None:
                        cleaned = group.replace("%", "").strip()
                        val = self._clean_number(cleaned)
                        if val is not None:
                            return val / 100.0 if val > 1.0 else val
        return None

    def _extract_variability_components(
        self, text: str
    ) -> Tuple[Optional[float], Optional[float], Optional[float], bool]:
        """
        Split question text into sentences/clauses.
        A standard deviation or variance stated in the same sentence or clause as lead time belongs to lead time;
        otherwise it belongs to demand.
        Returns:
            Tuple of (std_demand, std_lead_time, lead_time_variance, lead_time_stated_variable)
        """
        std_demand: Optional[float] = None
        std_lead_time: Optional[float] = None
        lead_time_variance: Optional[float] = None
        lead_time_stated_variable = False

        if re.search(
            r"\b(?:lead[- ]time\s+(?:varies|is\s+variable|fluctuates)|variable\s+lead[- ]time)\b",
            text,
            re.I,
        ):
            lead_time_stated_variable = True

        if re.search(
            r"\b(?:constant\s+lead[- ]time|lead[- ]time\s+is\s+a\s+constant|lead[- ]time\s+is\s+constant)\b",
            text,
            re.I,
        ):
            std_lead_time = 0.0
            lead_time_stated_variable = False

        # Split into sentences or clauses
        clauses = re.split(
            r"\.\s+|,\s+|;\s*|\n|\band\s+(?=the\s+mean\s+lead|lead[- ]time|delivery\s+time|demand\b)",
            text,
            flags=re.I,
        )

        for clause in clauses:
            cl = clause.strip(". ")
            if not cl:
                continue

            is_lt_clause = bool(
                re.search(r"\b(?:lead[- ]time|leadtime|delivery\s+time)\b", cl, re.I)
            )

            # Check for variance
            var_match = re.search(
                r"\b(?:variance\s+of\s+(?:lead[- ]time\s+)?|variance\s*(?:is|are|of|=|:)?|with\s+(?:a\s+)?variance\s+of)\s*(-?[\d,]+(?:\.\d+)?)",
                cl,
                re.I,
            )
            # Check for standard deviation
            std_match = re.search(
                r"\b(?:standard\s+deviation|std\s+dev|sigma|std)\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)",
                cl,
                re.I,
            )
            if not std_match:
                std_match = re.search(
                    r"\bwith\s+(?:a\s+)?standard\s+deviation\s+of\s*(-?[\d,]+(?:\.\d+)?)",
                    cl,
                    re.I,
                )

            if is_lt_clause:
                if var_match and "variance" in cl.lower():
                    val = self._clean_number(var_match.group(1))
                    if val is not None:
                        lead_time_variance = val
                elif std_match:
                    val = self._clean_number(std_match.group(1))
                    if val is not None:
                        std_lead_time = val
            else:
                if std_match:
                    val = self._clean_number(std_match.group(1))
                    if val is not None and std_demand is None:
                        std_demand = val

        return std_demand, std_lead_time, lead_time_variance, lead_time_stated_variable

    def _extract_demand_info(self, q: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Shared demand-extraction helper across all inventory decision models.
        Extracts numerical demand value and time unit ('day', 'week', 'month', 'year')
        from standard phrasings as well as phrasings using 'averages', 'on average',
        'about', 'approximately', etc.

        Returns:
            Tuple of (demand_value, time_unit_or_None)
        """
        UNIT_WORDS = r"(?:units?|cases?|items?|box(?:es)?|cartons?|tons?|gallons?|pounds?|lbs?|pieces?|pallets?|barrels?|newspapers?|pkg|packages?|pencils?|cakes?|roses?|parkas?)"

        patterns: List[Tuple[str, Optional[str]]] = [
            # 1. Explicit Annual
            (r"\b(?:annual|yearly)\s+demand(?:\s+for\s+(?:an?\s+)?\w+)?\s*(?:averages|is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)", "year"),
            (r"\b(?:annual\s+sales|annual\s+usage|annual\s+consumption)\s*(?:averages|is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)", "year"),
            (r"\bdemand\s+(?:averages|is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*(?:per\s+year|annually|a\s+year|/year)\b", "year"),
            (r"\b(?:we\s+)?(?:consumes?|uses?|sells?|use|sell)\s+(?:about|approx(?:imately)?)?\s*(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*(?:a|per|/)\s*year(?:\s+on\s+average)?\b", "year"),
            (r"\b(?:on\s+average\s*,?\s*)(?:we\s+)?(?:consumes?|uses?|sells?|use|sell)\s+(?:about|approx(?:imately)?)?\s*(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*(?:a|per|/)\s*year\b", "year"),
            (r"\b(?:forecasted\s+annual\s+demand|expected\s+annual\s+demand)\s*(?:averages|is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)", "year"),

            # 2. Explicit Monthly
            (r"\bmonthly\s+demand(?:\s+for\s+(?:an?\s+)?\w+)?\s*(?:averages|is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)", "month"),
            (r"\bdemand\s+per\s+month\s*(?:averages|is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)", "month"),
            (r"\bdemand\s+(?:averages|is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*(?:per\s+month|monthly|a\s+month|/month)\b", "month"),
            (r"\b(?:we\s+)?(?:consumes?|uses?|sells?|use|sell)\s+(?:about|approx(?:imately)?)?\s*(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*(?:per\s+month|monthly|a\s+month|/month)(?:\s+on\s+average)?\b", "month"),
            (r"\b(?:on\s+average\s*,?\s*)(?:we\s+)?(?:consumes?|uses?|sells?|use|sell)\s+(?:about|approx(?:imately)?)?\s*(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*(?:per\s+month|monthly|a\s+month|/month)\b", "month"),

            # 3. Explicit Weekly
            (r"\bweekly\s+demand(?:\s+for\s+(?:an?\s+)?\w+)?\s*(?:averages|is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)", "week"),
            (r"\bdemand\s+per\s+week\s*(?:averages|is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)", "week"),
            (r"\bdemand\s+(?:averages|is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*(?:per\s+week|weekly|a\s+week|/week)\b", "week"),
            (r"\b(?:we\s+)?(?:consumes?|uses?|sells?|use|sell)\s+(?:about|approx(?:imately)?)?\s*(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*(?:per\s+week|weekly|a\s+week|/week)(?:\s+on\s+average)?\b", "week"),
            (r"\b(?:on\s+average\s*,?\s*)(?:we\s+)?(?:consumes?|uses?|sells?|use|sell)\s+(?:about|approx(?:imately)?)?\s*(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*(?:per\s+week|weekly|a\s+week|/week)\b", "week"),

            # 4. Explicit Daily
            (r"\bdaily\s+demand(?:\s+for\s+(?:an?\s+)?\w+)?\s*(?:averages|is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)", "day"),
            (r"\bdemand\s+per\s+day\s*(?:averages|is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)", "day"),
            (r"\baverage\s+daily\s+demand\s*(?:averages|is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)", "day"),
            (r"\bdemand\s+(?:averages|is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*(?:per\s+day|daily|a\s+day|/day)\b", "day"),
            (r"\b(?:we\s+)?(?:consumes?|uses?|sells?|use|sell)\s+(?:about|approx(?:imately)?)?\s*(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*(?:per\s+day|daily|a\s+day|/day)(?:\s+on\s+average)?\b", "day"),
            (r"\b(?:on\s+average\s*,?\s*)(?:we\s+)?(?:consumes?|uses?|sells?|use|sell)\s+(?:about|approx(?:imately)?)?\s*(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*(?:per\s+day|daily|a\s+day|/day)\b", "day"),
            (r"\bmean\s+of\s*(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*per\s+day\b", "day"),

            # 5. Generic / Clause Demand
            (r"\b(?:average\s+demand|mean\s+demand|expected\s+demand|demand\s+rate|demand\s+averages)\s*(?:averages|is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)", None),
            (r"\bdemand\s*(?:averages|is|are|of|=|:)\s*(?!standard|std|sigma)(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\b", None),
            (r"\bdemand\s+has\s+(?:a\s+)?mean\s+of\s*(-?[\d,]+(?:\.\d+)?)", None),
            (r"\bsells?\s*(?:an?\s+average\s+of\s*)?(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\b", None),
            (r"\b(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"\s*per\s+(?:day|week|month|year)\b", None),
        ]

        for pat, unit in patterns:
            m = re.search(pat, q, re.I)
            if m:
                val = self._clean_number(m.group(1))
                if val is not None:
                    resolved_unit = unit
                    if resolved_unit is None:
                        if re.search(r"\b(?:day|daily)\b", q, re.I):
                            resolved_unit = "day"
                        elif re.search(r"\b(?:week|weekly)\b", q, re.I):
                            resolved_unit = "week"
                        elif re.search(r"\b(?:month|monthly)\b", q, re.I):
                            resolved_unit = "month"
                        elif re.search(r"\b(?:year|annual|annually|yearly)\b", q, re.I):
                            resolved_unit = "year"
                    return val, resolved_unit

        return None, None

    def extract_inputs(self, question: str, model_name: str, **kwargs) -> Tuple[Dict[str, Any], List[str]]:
        """
        Extract model-specific parameters from customer natural language query.
        Examines the entire question and recognizes all standard operations research
        terminology and synonyms before declaring any information missing.

        Returns:
            Tuple of (extracted_inputs_dict, missing_inputs_list)
        """
        inputs: Dict[str, Any] = {}
        missing: List[str] = []
        invalid_inputs: List[str] = []
        conversion_notes: List[str] = []
        q = question

        UNIT_WORDS = r"(?:units?|cases?|items?|box(?:es)?|cartons?|tons?|gallons?|pounds?|lbs?|pieces?|pallets?|barrels?|newspapers?|pkg|packages?|pencils?|cakes?|roses?)"

        if model_name == self.MODEL_EOQ:
            # 1. Annual Demand (and monthly/weekly conversions via shared helper)
            d = None
            d_val, d_unit = self._extract_demand_info(q)
            if d_val is not None:
                if d_unit == "year" or d_unit is None:
                    d = d_val
                elif d_unit == "month":
                    d = d_val * 12.0
                    inputs["monthly_demand"] = d_val
                    inputs["annual_demand_converted_from_monthly"] = True
                    conversion_notes.append(f"Monthly demand of {d_val:,.2f} converted to annual demand of {d:,.2f} (× 12 months/year) to match annual holding cost.")
                elif d_unit == "week":
                    d = d_val * 52.0
                    inputs["weekly_demand"] = d_val
                    inputs["annual_demand_converted_from_weekly"] = True
                    conversion_notes.append(f"Weekly demand of {d_val:,.2f} converted to annual demand of {d:,.2f} (× 52 weeks/year) to match annual holding cost.")
                elif d_unit == "day":
                    work_days_tmp = self._extract_number(q, [r"\b([\d]+)\s*(?:work|business|operating)?\s*days\s*per\s*year\b"])
                    if work_days_tmp is not None:
                        d = d_val * float(work_days_tmp)
                        conversion_notes.append(f"Daily demand of {d_val:,.2f} converted to annual demand of {d:,.2f} (× {int(work_days_tmp)} operating days/year).")
                    else:
                        missing.append("work_days_per_year (operating days per year required to convert daily demand to annual demand without guessing)")

            # Check explicit annual demand fallback if not yet found
            if d is None:
                d = self._extract_number(q, [
                    r"\b(?:annual\s+demand|yearly\s+demand|annual\s+sales|annual\s+usage|annual\s+consumption)\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)",
                    r"\b(?:we\s+)?sell\s+(?:about|approx(?:imately)?)?\s*(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*(?:a|per|/)\s*year\b",
                    r"\b(?:we\s+)?(?:consumes?|uses?|sells?)\s+(-?[\d,]+(?:\.\d+)?)\s*(?:[a-zA-Z]+)?\s*(?:per\s+year|annually|a\s+year|/year)\b",
                    r"\bdemand\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*(?:per\s+year|annually|a\s+year|/year)",
                    r"\b(?:uses?|consumption\s+is|we\s+use)\s*(-?[\d,]+(?:\.\d+)?)\s*(?:[a-z]+\s+)*(?:per\s+year|annually|a\s+year|/year)",
                    r"\b(-?[\d,]+(?:\.\d+)?)\s*(?:" + UNIT_WORDS + r")?(?:\s+of\s+\w+)?\s*(?:per\s+year|annually|a\s+year|/year)\b",
                    r"\b(?:forecasted\s+annual\s+demand|expected\s+annual\s+demand)\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)",
                ])

            if d is None:
                # Check monthly demand conversion fallback
                d_month = self._extract_number(q, [
                    r"\b(?:monthly\s+demand|demand\s+per\s+month)\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)",
                    r"\b(?:we\s+)?(?:consumes?|uses?|sells?)\s+(-?[\d,]+(?:\.\d+)?)\s*(?:[a-zA-Z]+)?\s*(?:per\s+month|monthly|a\s+month)\b",
                    r"\bdemand\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*(?:per\s+month|monthly)",
                    r"\bsells?\s*(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*(?:per\s+month|monthly)",
                    r"\b(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*(?:per\s+month|monthly)",
                ])
                if d_month is not None:
                    d = d_month * 12.0
                    inputs["monthly_demand"] = d_month
                    inputs["annual_demand_converted_from_monthly"] = True
                    conversion_notes.append(f"Monthly demand of {d_month:,.2f} converted to annual demand of {d:,.2f} (× 12 months/year) to match annual holding cost.")

            if d is None:
                # Check weekly demand conversion fallback
                d_week = self._extract_number(q, [
                    r"\b(?:weekly\s+demand|demand\s+per\s+week)\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)",
                    r"\b(?:we\s+)?(?:consumes?|uses?|sells?|use)\s+(-?[\d,]+(?:\.\d+)?)\s*(?:[a-zA-Z]+)?\s*(?:per\s+week|weekly|a\s+week)\b",
                    r"\bdemand\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*(?:per\s+week|weekly)",
                    r"\bsells?\s*(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*(?:per\s+week|weekly)",
                    r"\b(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*(?:per\s+week|weekly)",
                ])
                if d_week is not None:
                    d = d_week * 52.0
                    inputs["weekly_demand"] = d_week
                    inputs["annual_demand_converted_from_weekly"] = True
                    conversion_notes.append(f"Weekly demand of {d_week:,.2f} converted to annual demand of {d:,.2f} (× 52 weeks/year) to match annual holding cost.")

            if d is None:
                # Check daily demand conversion fallback
                d_daily = self._extract_number(q, [
                    r"\b(?:daily\s+demand|demand\s+per\s+day)\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)",
                    r"\b(?:we\s+)?(?:consumes?|uses?|sells?)\s+(-?[\d,]+(?:\.\d+)?)\s*(?:[a-zA-Z]+)?\s*(?:per\s+day|daily|a\s+day)\b",
                    r"\bdemand\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*(?:per\s+day|daily)",
                ])
                work_days_tmp = self._extract_number(q, [r"\b([\d]+)\s*(?:work|business|operating)?\s*days\s*per\s*year\b"])
                if d_daily is not None:
                    if work_days_tmp is not None:
                        d = d_daily * float(work_days_tmp)
                        conversion_notes.append(f"Daily demand of {d_daily:,.2f} converted to annual demand of {d:,.2f} (× {int(work_days_tmp)} operating days/year).")
                    else:
                        missing.append("work_days_per_year (operating days per year required to convert daily demand to annual demand without guessing)")

            if d is None and not any("work_days_per_year" in m for m in missing):
                # Fallback generic demand
                d = self._extract_number(q, [
                    r"\bdemand\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\b",
                ])

            # 2. Ordering Cost (all synonyms from specification)
            s = self._extract_number(q, [
                r"\beach\s+order\s+costs?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\s*(?:to\s+place)?\b",
                r"\bcosts?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\s*(?:to\s+place\s+an?\s+order|to\s+place|per\s+order)\b",
                r"\b(?:ordering\s+cost|order\s+cost|setup\s+cost|setup\s+expense|cost\s+per\s+order|cost\s+of\s+placing\s+an?\s+order|ordering\s*/\s*setup\s+cost|order\s+placement\s+cost|cost\s+to\s+place\s+an?\s+order|cost\s+to\s+order|procurement\s+cost\s+per\s+order)\s*(?:is|are|of|=|:)?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
                r"\$?\s*(-?[\d,]+(?:\.\d+)?)\s*(?:per\s+order|to\s+place\s+an?\s+order|setup\s+expense|setup\s+cost|order\s+placement\s+cost)\b",
            ])

            # 3. Percentage Holding Cost Rate (Check percentage first)
            h_rate = self._extract_percentage(q, [
                r"\b(?:holding\s+cost\s+rate|carrying\s+cost\s+rate|carrying\s+charge|inventory\s+carrying\s+rate|annual\s+holding\s+rate|holding\s+rate|carrying\s+rate)\s*(?:is|are|of|=|:)?\s*([\d.]+\s*%)",
                r"\b(?:holding|carrying)\s+cost\s*(?:is|are|of|=|:)?\s*([\d.]+\s*%)\s*(?:of\s+(?:the\s+)?(?:unit\s+)?(?:purchase\s+)?(?:cost|price|item\s+value|value))?",
                r"\b(?:holding|carrying)\s+cost\s*(?:is|are|of|=|:)?\s*([\d.]+\s*%)",
                r"([\d.]+\s*%)\s*(?:of\s+(?:the\s+)?(?:unit\s+)?(?:purchase\s+)?(?:price|cost|item\s+value|value))",
                r"([\d.]+\s*%)\s*(?:of\s+price\s+per\s+year|per\s+year|a\s+year)",
                r"([\d.]+\s*%)\s*(?:annual\s+)?(?:holding|carrying)\s+(?:rate|charge|cost)",
            ])

            # 4. Holding Cost (Fixed holding cost per unit per year - only if not percentage)
            h = None
            if h_rate is None:
                h = self._extract_number(q, [
                    r"\b(?:each\s+" + UNIT_WORDS + r"?\s+)?costs?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\s*(?:a|per|/)\s*year\s*to\s*(?:hold|carry)\b",
                    r"\$?\s*(-?[\d,]+(?:\.\d+)?)\s*per\s+" + UNIT_WORDS + r"\s*(?:per\s+year|a\s+year)?\s*(?:to\s+(?:hold|carry)(?:\s+inventory)?)\b",
                    r"\b(?:it\s+)?costs?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\s*(?:a|per|/)\s*year\s*to\s+(?:hold|carry)\s+(?:each|one|a)?\s*" + UNIT_WORDS + r"?\b",
                    r"\b(?:costs?\s+)?\$?\s*(-?[\d,]+(?:\.\d+)?)\s+to\s+(?:carry|hold)\s+(?:one|each|a)\s*" + UNIT_WORDS + r"?\s*(?:for\s+a|per)\s*year\b",
                    r"\b(?:it\s+)?costs?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\s+to\s+hold\s+(?:one|a|each)?\s*" + UNIT_WORDS + r"?\s*(?:for\s+a|per)\s*year\b",
                    r"\b(?:costs?\s+)?\$?\s*(-?[\d,]+(?:\.\d+)?)\s+to\s+hold\b",
                    r"\b(?:fixed\s+holding\s+cost|unit\s+holding\s+cost|holding\s+cost|carrying\s+cost|inventory\s+carrying\s+cost|storage\s+cost|holding\s+expense|carrying\s+expense|cost\s+of\s+holding|cost\s+to\s+hold|holding\s+charge)\s*(?:is|are|of|=|:)?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\s*(?:per\s+" + UNIT_WORDS + r"|\$|/unit)?",
                    r"\$?\s*(-?[\d,]+(?:\.\d+)?)\s*(?:per\s+" + UNIT_WORDS + r"\s*(?:per\s+year|annually|to\s+hold)|/unit/year|holding\s+cost|carrying\s+cost)\b",
                ])

            # 5. Unit Purchase Price / Cost
            c = self._extract_number(q, [
                r"\b(?:the\s+)?(?:item|product|unit|case|good)\s+costs?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)(?!\s*(?:per\s+year|a\s+year|/year)?\s*to\s*(?:hold|carry))\b",
                r"\b(?:unit\s+purchase\s+price|unit\s+purchase\s+cost|purchase\s+price|unit\s+cost|cost\s+per\s+unit|wholesale\s+price|item\s+cost|item\s+price|unit\s+price)\s*(?:is|are|of|=|:)?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
                r"(?<!order\s)(?<!order\scost\s)(?<!ordering\scost\s)(?<!setup\scost\s)\bcosts?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)(?!\s*(?:per\s+year|a\s+year|/year)?\s*to\s*(?:hold|carry))\s*per\s+" + UNIT_WORDS + r"\b",
                r"(?<!order\s)(?<!order\scost\s)(?<!ordering\scost\s)(?<!setup\scost\s)\$?\s*(-?[\d,]+(?:\.\d+)?)\s*(?:each\b|per\s+" + UNIT_WORDS + r"\b)(?!\s*(?:per\s+year|a\s+year|/year)?\s*to\s*(?:hold|carry))\b",
            ])

            # 6. Work Days Per Year
            work_days = self._extract_number(q, [
                r"\b([\d]+)\s*(?:work|business|operating)?\s*days\s*per\s*year\b",
                r"\boperates?\s*([\d]+)\s*days\b",
            ])

            # Populate inputs & check invalid values
            if d is not None:
                inputs["demand"] = d
                inputs["annual_demand"] = d
                if d <= 0:
                    invalid_inputs.append(f"Annual demand (extracted: {d:,.2f} units/year) must be strictly positive (> 0).")
            elif not any("work_days_per_year" in m for m in missing):
                missing.append("annual_demand (D)")

            if s is not None:
                inputs["order_cost"] = s
                if s <= 0:
                    invalid_inputs.append(f"Ordering / setup cost (extracted: ${s:,.2f}) must be strictly positive (> 0).")
            else:
                missing.append("order_cost (S)")

            if h is not None:
                inputs["holding_cost"] = h
                if h <= 0:
                    invalid_inputs.append(f"Holding cost (extracted: ${h:,.2f}) must be strictly positive (> 0).")
            elif h_rate is not None and c is not None:
                if h_rate <= 0:
                    invalid_inputs.append(f"Holding cost rate (extracted: {h_rate*100:.2f}%) must be strictly positive (> 0).")
                if c <= 0:
                    invalid_inputs.append(f"Unit cost (extracted: ${c:,.2f}) must be strictly positive (> 0).")
                inputs["holding_cost_rate"] = h_rate
                inputs["unit_cost"] = c
                inputs["holding_cost"] = h_rate * c
            elif h_rate is not None and c is None:
                inputs["holding_cost_rate"] = h_rate
                missing.append("unit_cost (C) to evaluate percentage holding cost rate")
            else:
                missing.append("holding_cost (H) or holding_cost_rate (h)")

            if c is not None and (h is None or c != h) and "unit_cost" not in inputs:
                inputs["unit_cost"] = c

            if work_days is not None:
                inputs["work_days_per_year"] = int(work_days)
            elif "work_days_per_year" in kwargs and kwargs["work_days_per_year"] is not None:
                inputs["work_days_per_year"] = int(kwargs["work_days_per_year"])

        elif model_name == self.MODEL_QUANTITY_DISCOUNT:
            # 1. Demand (and conversions via shared helper)
            d = None
            d_val, d_unit = self._extract_demand_info(q)
            if d_val is not None:
                if d_unit == "year" or d_unit is None:
                    d = d_val
                elif d_unit == "month":
                    d = d_val * 12.0
                    inputs["monthly_demand"] = d_val
                    conversion_notes.append(f"Monthly demand of {d_val:,.2f} converted to annual demand of {d:,.2f} (× 12 months/year).")
                elif d_unit == "week":
                    d = d_val * 52.0
                    inputs["weekly_demand"] = d_val
                    conversion_notes.append(f"Weekly demand of {d_val:,.2f} converted to annual demand of {d:,.2f} (× 52 weeks/year).")

            if d is None:
                d = self._extract_number(q, [
                    r"\b(?:annual\s+demand|yearly\s+demand|annual\s+sales)\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)",
                    r"\b(?:we\s+)?(?:consumes?|uses?|sells?)\s+(-?[\d,]+(?:\.\d+)?)\s*(?:[a-zA-Z]+)?\s*(?:per\s+year|annually|a\s+year)\b",
                    r"\b(?:uses?|consumption\s+is|we\s+use)\s*(-?[\d,]+(?:\.\d+)?)\s*(?:[a-z]+\s+)*(?:per\s+year|annually|a\s+year)",
                    r"\bsells?\s*(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*(?:per\s+year|annually|a\s+year)?",
                    r"\b(-?[\d,]+(?:\.\d+)?)\s*(?:" + UNIT_WORDS + r")?(?:\s+of\s+\w+)?\s*(?:per\s+year|annually|a\s+year)",
                ])

            if d is None:
                # Check monthly conversion
                d_month = self._extract_number(q, [
                    r"\b(?:monthly\s+demand|demand\s+per\s+month)\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)",
                    r"\b(?:we\s+)?(?:consumes?|uses?|sells?)\s+(-?[\d,]+(?:\.\d+)?)\s*(?:[a-zA-Z]+)?\s*(?:per\s+month|monthly|a\s+month)\b",
                    r"\bdemand\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*(?:per\s+month|monthly)",
                ])
                if d_month is not None:
                    d = d_month * 12.0
                    inputs["monthly_demand"] = d_month
                    conversion_notes.append(f"Monthly demand of {d_month:,.2f} converted to annual demand of {d:,.2f} (× 12 months/year).")

            if d is None:
                # Check weekly conversion
                d_week = self._extract_number(q, [
                    r"\b(?:weekly\s+demand|demand\s+per\s+week)\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)",
                    r"\b(?:we\s+)?(?:consumes?|uses?|sells?|use)\s+(-?[\d,]+(?:\.\d+)?)\s*(?:[a-zA-Z]+)?\s*(?:per\s+week|weekly|a\s+week)\b",
                    r"\bdemand\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*(?:per\s+week|weekly)",
                ])
                if d_week is not None:
                    d = d_week * 52.0
                    inputs["weekly_demand"] = d_week
                    conversion_notes.append(f"Weekly demand of {d_week:,.2f} converted to annual demand of {d:,.2f} (× 52 weeks/year).")

            if d is None:
                d = self._extract_number(q, [
                    r"\bdemand\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)",
                ])

            # 2. Ordering Cost (supports all synonyms)
            s = self._extract_number(q, [
                r"\beach\s+order\s+costs?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\s*(?:to\s+place)?\b",
                r"\b(?:ordering\s+cost|order\s+cost|setup\s+cost|setup\s+expense|cost\s+per\s+order|cost\s+of\s+placing\s+an?\s+order|ordering\s*/\s*setup\s+cost|order\s+placement\s+cost)\s*(?:is|are|of|=|:)?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
                r"\$?\s*(-?[\d,]+(?:\.\d+)?)\s*(?:per\s+order|to\s+place\s+an?\s+order|setup\s+cost|setup\s+expense)\b",
            ])

            # 3. Percentage Holding Cost Rate (Check percentage first)
            h_rate = self._extract_percentage(q, [
                r"\b(?:holding\s+cost\s+rate|carrying\s+cost\s+rate|carrying\s+charge|carrying\s+rate|holding\s+rate)\s*(?:is|are|of|=|:)?\s*([\d.]+\s*%)",
                r"\b(?:holding|carrying)\s+(?:cost|rate)?\s*(?:is|are|of|=|:)?\s*([\d.]+\s*%)\s*(?:of\s+(?:the\s+)?(?:unit\s+)?(?:purchase\s+)?(?:price|cost|item\s+value|value))?",
                r"\b(?:holding|carrying)\s+(?:cost|rate)?\s*(?:is|are|of|=|:)?\s*([\d.]+\s*%)",
                r"([\d.]+\s*%)\s*(?:of\s+(?:the\s+)?(?:unit\s+)?(?:purchase\s+)?(?:price|cost|item\s+value|value))",
                r"([\d.]+\s*%)\s*(?:of\s+price\s+per\s+year|per\s+year|a\s+year)",
                r"([\d.]+\s*%)\s*(?:annual\s+)?(?:holding|carrying)\s+(?:rate|cost)",
            ])

            # 4. Fixed Holding Cost (Only if percentage holding rate is not provided)
            fixed_h = None
            if h_rate is None:
                fixed_h = self._extract_number(q, [
                    r"\$?\s*(-?[\d,]+(?:\.\d+)?)\s*per\s+" + UNIT_WORDS + r"\s*(?:per\s+year|a\s+year)?\s*(?:to\s+(?:hold|carry)(?:\s+inventory)?)\b",
                    r"\b(?:it\s+)?costs?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\s*(?:a|per)\s+year\s+to\s+(?:hold|carry)\s+(?:each|one|a)?\s*" + UNIT_WORDS + r"?\b",
                    r"\b(?:costs?\s+)?\$?\s*(-?[\d,]+(?:\.\d+)?)\s+to\s+(?:carry|hold)\s+(?:one|each|a)\s*" + UNIT_WORDS + r"?\s*(?:for\s+a|per)\s*year\b",
                    r"\b(?:fixed\s+holding\s+cost|fixed\s+carrying\s+cost)\s*(?:is|are|of|=|:)?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
                    r"\b(?:carrying\s+cost|holding\s+cost)\s*(?:is|are|of|=|:)?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\s*(?:per\s+" + UNIT_WORDS + r"\s*(?:per\s+year|annually)?|/unit)",
                    r"\$?\s*(-?[\d,]+(?:\.\d+)?)\s*per\s+" + UNIT_WORDS + r"\s*per\s*year\b",
                    r"\b(?:carrying\s+cost|holding\s+cost)\s*(?:is|are|of|=|:)?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
                ])

            # 5. Baseline Unit Purchase Price (if given)
            base_p = self._extract_number(q, [
                r"\b(?:baseline\s+(?:unit\s+)?purchase\s+price|regular\s+price|base\s+price|standard\s+price)\s*(?:is|are|of|=|:)?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
                r"\b(?:charges?|supplier\s+charges?)\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\s*(?:per\s+" + UNIT_WORDS + r")?\s*(?:but\s+drops|\band\s+drops|however|\byet\b)",
            ])

            # 6. Parse Quantity Breaks & Unit Prices
            tiers: List[Dict[str, Any]] = []

            # Common connecting phrases between price and quantity condition
            conn = r"(?:per\s+" + UNIT_WORDS + r"|per\s+unit|each)?\s*(?:for|if\s+buying|if\s+ordering|when\s+ordering|at)?\s*(?:orders?\s+(?:of\s+)?)?"
            upper_words = r"(?:under|less\s+than|below|fewer\s+than|up\s+to)"
            lower_words_pre = r"(?:at\s+least|over|above)"
            lower_words_post = r"(?:\+|and\s+above|or\s+more|and\s+over|and\s+up)"

            # (A) Check price-first schedules:
            price_first_tiers: List[Dict[str, Any]] = []

            # 1. Price-first upper bound: "$12 per unit for orders below 500 units"
            for pm in re.finditer(
                r"\$\s*([\d]+(?:\.\d+)?)\s*" + conn + r"(" + upper_words + r")\s+([\d,]+)\s*(?:" + UNIT_WORDS + r")?",
                q,
                re.I,
            ):
                p_val = self._clean_number(pm.group(1))
                q_bound = self._clean_number(pm.group(3))
                is_strict = any(w in pm.group(2).lower() for w in ["under", "less", "below", "fewer"])
                if p_val is not None and q_bound is not None:
                    max_q = (q_bound - 1.0) if is_strict else q_bound
                    price_first_tiers.append({"min_qty": 1.0, "max_qty": max_q, "unit_price": p_val})

            # 2. Price-first middle range: "$11.50 for 500-999 units", "$11.50 for between 500 and 999 units"
            for pm in re.finditer(
                r"\$\s*([\d]+(?:\.\d+)?)\s*" + conn + r"(?:between\s+)?([\d,]+)\s*(?:-|–|to|and)\s*([\d,]+)\s*(?:" + UNIT_WORDS + r")?",
                q,
                re.I,
            ):
                p_val = self._clean_number(pm.group(1))
                q_low = self._clean_number(pm.group(2))
                q_high = self._clean_number(pm.group(3))
                if p_val is not None and q_low is not None and q_high is not None and q_high > q_low:
                    price_first_tiers.append({"min_qty": q_low, "max_qty": q_high, "unit_price": p_val})

            # 3. Price-first lower bound with prefix keyword: "$11 for at least 1,000 units"
            for pm in re.finditer(
                r"\$\s*([\d]+(?:\.\d+)?)\s*" + conn + lower_words_pre + r"\s+([\d,]+)\s*(?:" + UNIT_WORDS + r")?",
                q,
                re.I,
            ):
                p_val = self._clean_number(pm.group(1))
                q_min = self._clean_number(pm.group(2))
                if p_val is not None and q_min is not None:
                    price_first_tiers.append({"min_qty": q_min, "max_qty": None, "unit_price": p_val})

            # 4. Price-first lower bound with suffix keyword: "$11 for orders of 1,000 units or more", "$11 for 1,000 and up"
            for pm in re.finditer(
                r"\$\s*([\d]+(?:\.\d+)?)\s*" + conn + r"([\d,]+)\s*(?:" + UNIT_WORDS + r")?\s*" + lower_words_post + r"\s*(?:" + UNIT_WORDS + r")?",
                q,
                re.I,
            ):
                p_val = self._clean_number(pm.group(1))
                q_min = self._clean_number(pm.group(2))
                if p_val is not None and q_min is not None:
                    price_first_tiers.append({"min_qty": q_min, "max_qty": None, "unit_price": p_val})

            if price_first_tiers:
                tiers.extend(price_first_tiers)
            else:
                # (B) Quantity-first tiers:
                # Lower bounded / upper open tiers: "under 50 cases $20", "below 500 units at $15", "up to 500: $15"
                for um in re.finditer(
                    r"\b(" + upper_words + r")\s+([\d,]+)\s*(?:" + UNIT_WORDS + r")?\s*(?:at|costs?|is|are|:|@|=)?\s*\$?\s*([\d]+(?:\.\d+)?)",
                    q,
                    re.I,
                ):
                    is_strict = any(w in um.group(1).lower() for w in ["under", "less", "below", "fewer"])
                    q_bound = self._clean_number(um.group(2))
                    p_val = self._clean_number(um.group(3))
                    if q_bound is not None and p_val is not None:
                        max_q = (q_bound - 1.0) if is_strict else q_bound
                        tiers.append({"min_qty": 1.0, "max_qty": max_q, "unit_price": p_val})

                # Middle ranges: "1-49 cases $20", "50-79 cases at $18", "80-99 cases $17", "between 500 and 999: $28"
                # (negative lookahead prevents matching price that belongs to subsequent price-first tier)
                for rm in re.finditer(
                    r"\b(?:between\s+([\d,]+)\s+and\s+([\d,]+)|([\d,]+)\s*(?:-|–|to)\s*([\d,]+))\s*(?:" + UNIT_WORDS + r")?\s*(?:at|costs?|is|are|:|@|=)?\s*\$?\s*([\d]+(?:\.\d+)?)(?!\s*(?:for|per\s+unit\s+for)\s+[\d,]+)",
                    q,
                    re.I,
                ):
                    q_low = self._clean_number(rm.group(1) or rm.group(3))
                    q_high = self._clean_number(rm.group(2) or rm.group(4))
                    p_val = self._clean_number(rm.group(5))
                    if q_low is not None and q_high is not None and p_val is not None:
                        if q_high > q_low:
                            tiers.append({"min_qty": q_low, "max_qty": q_high, "unit_price": p_val})

                # Upper open range: "100 or more cases $16", "1000+ units at $26", "2000 or more units at $13", "1,000 units and up at $11"
                for pm in re.finditer(
                    r"\b([\d,]+)\s*(?:" + UNIT_WORDS + r")?\s*" + lower_words_post + r"\s*(?:" + UNIT_WORDS + r")?\s*(?:at|costs?|is|are|:|@|=)?\s*\$?\s*([\d]+(?:\.\d+)?)(?!\s*(?:for|per\s+unit\s+for)\s+[\d,]+)",
                    q,
                    re.I,
                ):
                    q_min = self._clean_number(pm.group(1))
                    p_val = self._clean_number(pm.group(2))
                    if q_min is not None and p_val is not None:
                        tiers.append({"min_qty": q_min, "max_qty": None, "unit_price": p_val})

                for om in re.finditer(
                    r"\b" + lower_words_pre + r"\s+([\d,]+)\s*(?:" + UNIT_WORDS + r")?\s*(?:at|costs?|is|are|:|@|=)?\s*\$?\s*([\d]+(?:\.\d+)?)(?!\s*(?:for|per\s+unit\s+for)\s+[\d,]+)",
                    q,
                    re.I,
                ):
                    q_min = self._clean_number(om.group(1))
                    p_val = self._clean_number(om.group(2))
                    if q_min is not None and p_val is not None:
                        if not any(t["min_qty"] == q_min and t["max_qty"] is None for t in tiers):
                            tiers.append({"min_qty": q_min, "max_qty": None, "unit_price": p_val})

            # (C) Two-tier price drop: "charges $20 per case but drops the price to $16 if we buy 100 or more"
            if len(tiers) <= 1 and base_p is not None:
                drop_match = re.search(
                    r"\b(?:drops?\s+the\s+price\s+to|discount(?:ed)?\s+price\s+of|price\s+(?:drops?\s+to|is))\s*\$?\s*([\d.]+)\s*(?:if\s+we\s+(?:buy|order)|when\s+ordering|for\s+orders?\s+of|for)\s*(?:at\s+least\s+)?([\d,]+)\s*(?:or\s+more|\+)?",
                    q,
                    re.I,
                )
                if drop_match:
                    drop_price = self._clean_number(drop_match.group(1))
                    break_qty = self._clean_number(drop_match.group(2))
                    if drop_price and break_qty:
                        tiers = [
                            {"min_qty": 1.0, "max_qty": break_qty - 1.0, "unit_price": base_p},
                            {"min_qty": break_qty, "max_qty": None, "unit_price": drop_price},
                        ]

            # Sort tiers and adjust boundary if up-to matches next tier min_qty
            tiers.sort(key=lambda x: x["min_qty"])
            for i in range(len(tiers) - 1):
                if tiers[i]["max_qty"] is not None and tiers[i + 1]["min_qty"] is not None:
                    if tiers[i]["max_qty"] == tiers[i + 1]["min_qty"]:
                        tiers[i]["max_qty"] -= 1.0

            # Deduplicate and sort tiers
            seen_brackets = set()
            unique_tiers = []
            for t in sorted(tiers, key=lambda x: x["min_qty"]):
                key = (t["min_qty"], t["max_qty"])
                if key not in seen_brackets:
                    seen_brackets.add(key)
                    unique_tiers.append(t)

            # Item 2 validation: If the question contains more prices or quantity ranges
            # than the number of tiers parsed, report the discount schedule as missing instead of calculating.
            range_matches = re.findall(
                r"\b[\d,]+\s*(?:-|–|to)\s*[\d,]+\b"
                r"|\b[\d,]+\s*(?:" + UNIT_WORDS + r")?\s*(?:\+|or\s+more\b|and\s+above\b|and\s+over\b|and\s+up\b)"
                r"|\b(?:under|less\s+than|below|fewer\s+than|up\s+to|over|above|at\s+least)\s+(?:orders?\s+(?:of\s+)?)?[\d,]+\b",
                q,
                re.I,
            )
            num_ranges = len(range_matches)

            dollar_matches = [
                self._clean_number(p)
                for p in re.findall(r"\$\s*(-?[\d,]+(?:\.\d+)?)", q)
                if self._clean_number(p) is not None
            ]
            tier_dollar_candidates = list(dollar_matches)
            if s is not None and s in tier_dollar_candidates:
                tier_dollar_candidates.remove(s)
            if fixed_h is not None and fixed_h in tier_dollar_candidates:
                tier_dollar_candidates.remove(fixed_h)
            if base_p is not None and base_p in tier_dollar_candidates:
                tier_dollar_candidates.remove(base_p)
            num_tier_prices = len(tier_dollar_candidates)

            if len(unique_tiers) < num_ranges or (num_tier_prices > 1 and len(unique_tiers) < num_tier_prices):
                unique_tiers = []

            if d is not None:
                inputs["demand"] = d
                inputs["annual_demand"] = d
                if d <= 0:
                    invalid_inputs.append(f"Annual demand (extracted: {d:,.2f}) must be strictly positive (> 0).")
            else:
                missing.append("annual_demand (D)")

            if s is not None:
                inputs["order_cost"] = s
                if s <= 0:
                    invalid_inputs.append(f"Order cost (extracted: ${s:,.2f}) must be strictly positive (> 0).")
            else:
                missing.append("order_cost (S)")

            if h_rate is not None:
                inputs["holding_cost_rate"] = h_rate
                if h_rate <= 0:
                    invalid_inputs.append(f"Holding cost rate (extracted: {h_rate*100:.2f}%) must be strictly positive (> 0).")
            elif fixed_h is not None:
                inputs["fixed_holding_cost"] = fixed_h
                if fixed_h <= 0:
                    invalid_inputs.append(f"Fixed holding cost (extracted: ${fixed_h:,.2f}) must be strictly positive (> 0).")
            else:
                missing.append("holding_cost_rate (percentage) or fixed_holding_cost")

            if base_p is not None:
                inputs["unit_purchase_price"] = base_p

            if unique_tiers:
                for idx, t in enumerate(unique_tiers):
                    if t["unit_price"] <= 0:
                        invalid_inputs.append(f"Tier {idx+1} unit price (${t['unit_price']:.2f}) must be strictly positive (> 0).")
                inputs["discount_schedule"] = unique_tiers
            else:
                missing.append("discount_schedule (quantity breaks and unit prices)")

        elif model_name == self.MODEL_SAFETY_STOCK_ROP:
            # 0. Check figures given "during lead time"
            has_ltd_phrase = bool(
                re.search(
                    r"\b(?:demand\s+(?:for\s+\w+\s+)?during\s+(?:the\s+)?lead[- ]time|lead[- ]time\s+demand)\b",
                    q,
                    re.I,
                )
            )

            lt_var_kw = kwargs.get("lead_time_variance")
            ltd_mean_kw = kwargs.get("lead_time_demand_mean")
            ltd_std_kw = kwargs.get("lead_time_demand_std")

            ltd_mean = ltd_mean_kw
            ltd_std = ltd_std_kw

            if has_ltd_phrase and ltd_mean is None:
                for clause in re.split(r"[.;\n]", q):
                    if re.search(r"\b(?:demand\s+(?:for\s+\w+\s+)?during\s+(?:the\s+)?lead[- ]time|lead[- ]time\s+demand)\b", clause, re.I):
                        m_mean = re.search(r"\b(?:mean|average)\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)", clause, re.I)
                        m_std = re.search(r"\b(?:standard\s+deviation|std\s+dev|sigma|std)\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)", clause, re.I)
                        if m_mean:
                            ltd_mean = self._clean_number(m_mean.group(1))
                        if m_std:
                            ltd_std = self._clean_number(m_std.group(1))

            if ltd_mean is not None:
                inputs["lead_time_demand_mean"] = float(ltd_mean)
                if ltd_mean <= 0:
                    invalid_inputs.append(f"Lead-time demand mean (extracted: {ltd_mean}) must be strictly positive (> 0).")
            if ltd_std is not None:
                inputs["lead_time_demand_std"] = float(ltd_std)
                if ltd_std < 0:
                    invalid_inputs.append(f"Lead-time demand standard deviation (extracted: {ltd_std}) cannot be negative.")

            # 1. Lead Time and Variability extraction
            if has_ltd_phrase:
                inputs["std_lead_time"] = 0.0
                inputs["std_demand"] = 0.0
            else:
                lt_var = lt_var_kw if lt_var_kw is not None else None
                std_d, std_lt, lt_var_extracted, lt_stated_var = self._extract_variability_components(q)
                if lt_var is None and lt_var_extracted is not None:
                    lt_var = lt_var_extracted

                if lt_var is not None:
                    inputs["lead_time_variance"] = float(lt_var)
                    if lt_var < 0:
                        invalid_inputs.append(f"Lead time variance (extracted: {lt_var}) cannot be negative.")
                elif std_lt is not None:
                    inputs["std_lead_time"] = float(std_lt)
                    if std_lt < 0:
                        invalid_inputs.append(f"Lead time standard deviation (extracted: {std_lt}) cannot be negative.")
                elif lt_stated_var:
                    missing.append("lead_time_variability (standard deviation or variance of lead time)")
                else:
                    inputs["std_lead_time"] = 0.0

                if std_d is not None:
                    inputs["std_demand"] = float(std_d)
                    if std_d < 0:
                        invalid_inputs.append(f"Demand standard deviation (extracted: {std_d}) cannot be negative.")
                else:
                    inputs["std_demand"] = 0.0

            # 2. Lead Time Value and Time Unit Detection
            lt_raw = self._extract_number(q, [
                r"\b(?:average\s+lead[- ]time|mean\s+lead[- ]time|supplier\s+lead[- ]time|replenishment\s+lead[- ]time|lead[- ]time\s+averages)\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)",
                r"\blead[- ]time\s+is\s+a\s+constant\s*(-?[\d,]+(?:\.\d+)?)",
                r"\b(?:lead[- ]time|lead\s+time)\s*(?:is|are|of|=|:)?\s*(?!standard|std|sigma)(-?[\d,]+(?:\.\d+)?)\s*(?:days?|weeks?|months?|business\s+days?)?\b",
                r"\btakes?\s*(-?[\d,]+(?:\.\d+)?)\s*(?:days?|weeks?|months?)\s*(?:to\s+arrive|lead[- ]time)?\b",
                r"\bdelivery\s+time\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)",
            ])

            # Determine time unit of demand
            d_val_tmp, d_unit_tmp = self._extract_demand_info(q)
            demand_unit = d_unit_tmp if d_unit_tmp else ("week" if re.search(r"\b(?:weekly|per\s+week|a\s+week)\b", q, re.I) else ("month" if re.search(r"\b(?:monthly|per\s+month|a\s+month)\b", q, re.I) else "day"))

            lt_unit = "day"
            lt_match = re.search(r"\blead[- ]time\s+(?:is|averages|is\s+a\s+constant\s+)?(?:of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)\s*(days?|weeks?|months?|business\s+days?|work\s+days?)", q, re.I)
            if not lt_match:
                lt_match = re.search(r"\b(-?[\d,]+(?:\.\d+)?)\s*(days?|weeks?|months?)\b.*?\blead[- ]time\b", q, re.I)
            if lt_match and lt_match.group(2):
                u_str = lt_match.group(2).lower()
                if "week" in u_str:
                    lt_unit = "week"
                elif "month" in u_str:
                    lt_unit = "month"
                elif "day" in u_str:
                    lt_unit = "day"

            if lt_raw is not None:
                if lt_raw <= 0:
                    invalid_inputs.append(f"Lead time (extracted: {lt_raw}) must be strictly positive (> 0).")

                conversion_factor = 1.0
                conversion_note_phrase = ""

                if demand_unit == "day" and lt_unit == "week":
                    conversion_factor = 7.0
                    conversion_note_phrase = "× 7 days/week"
                elif demand_unit == "week" and lt_unit == "day":
                    conversion_factor = 1.0 / 7.0
                    conversion_note_phrase = "÷ 7 days/week"
                elif demand_unit == "day" and lt_unit == "month":
                    m_days = re.search(r"\b(\d+)\s*(?:operating\s+days|work(?:ing)?\s+days|days)\s*(?:a|per)\s*month\b", q, re.I)
                    days_per_month = float(m_days.group(1)) if m_days else 30.0
                    conversion_factor = days_per_month
                    conversion_note_phrase = f"× {days_per_month:g} days/month"
                elif demand_unit == "week" and lt_unit == "month":
                    conversion_factor = 4.0
                    conversion_note_phrase = "× 4 weeks/month"
                elif demand_unit == "month" and lt_unit == "day":
                    m_days = re.search(r"\b(\d+)\s*(?:operating\s+days|work(?:ing)?\s+days|days)\s*(?:a|per)\s*month\b", q, re.I)
                    days_per_month = float(m_days.group(1)) if m_days else 30.0
                    conversion_factor = 1.0 / days_per_month
                    conversion_note_phrase = f"÷ {days_per_month:g} days/month"
                elif demand_unit == "month" and lt_unit == "week":
                    conversion_factor = 1.0 / 4.0
                    conversion_note_phrase = "÷ 4 weeks/month"

                if conversion_factor != 1.0:
                    demand_unit_adj = "daily" if demand_unit == "day" else ("weekly" if demand_unit == "week" else ("monthly" if demand_unit == "month" else demand_unit))
                    converted_lt = lt_raw * conversion_factor
                    inputs["_raw_lead_time"] = lt_raw
                    inputs["lead_time"] = converted_lt
                    inputs["lead_time_unit_converted"] = True
                    conversion_notes.append(f"Lead time of {lt_raw:g} {lt_unit}s converted to {converted_lt:g} {demand_unit}s ({conversion_note_phrase}) to match {demand_unit_adj} demand.")

                    if "std_lead_time" in inputs and inputs["std_lead_time"] > 0:
                        raw_std_lt = inputs["std_lead_time"]
                        inputs["_raw_std_lead_time"] = raw_std_lt
                        converted_std_lt = raw_std_lt * conversion_factor
                        inputs["std_lead_time"] = converted_std_lt
                        conversion_notes.append(f"Lead-time standard deviation of {raw_std_lt:g} {lt_unit}s converted to {converted_std_lt:g} {demand_unit}s ({conversion_note_phrase}) to match {demand_unit_adj} demand.")

                    if "lead_time_variance" in inputs and inputs["lead_time_variance"] > 0:
                        raw_lt_var = inputs["lead_time_variance"]
                        inputs["_raw_lead_time_variance"] = raw_lt_var
                        var_factor = conversion_factor ** 2
                        converted_lt_var = raw_lt_var * var_factor
                        inputs["lead_time_variance"] = converted_lt_var
                        conversion_notes.append(f"Lead-time variance of {raw_lt_var:g} {lt_unit}s² converted to {converted_lt_var:g} {demand_unit}s² (× {var_factor:g}) to match {demand_unit_adj} demand.")
                else:
                    inputs["lead_time"] = lt_raw
            elif "lead_time_demand_mean" not in inputs:
                missing.append("average lead time (L)")

            # 3. Average Demand (if figures during lead time not given)
            if "lead_time_demand_mean" not in inputs:
                if d_val_tmp is not None:
                    d = d_val_tmp
                else:
                    d = self._extract_number(q, [
                        r"\b(?:average\s+demand|mean\s+demand|expected\s+demand|daily\s+demand|weekly\s+demand|average\s+daily\s+demand|average\s+weekly\s+demand|demand\s+rate|demand\s+averages)\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)",
                        r"\b(?:consumes?|uses?|sells?)\s+(-?[\d,]+(?:\.\d+)?)\s*(?:[a-zA-Z]+)?\s*per\s+(?:day|week|month)\b",
                        r"\b(?:consumes?|uses?|sells?)\s+(-?[\d,]+(?:\.\d+)?)\s*(?:[a-zA-Z]+)?\s*a\s+(?:day|week|month)\b",
                        r"\bdemand\s+has\s+(?:a\s+)?mean\s+of\s*(-?[\d,]+(?:\.\d+)?)",
                        r"\bmean\s+of\s*(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*per\s+(?:day|week|month)",
                        r"\bdemand\s*(?:is|are|of|=|:)?\s*(?!standard|std|sigma)(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*per\s+(?:day|week|month)\b",
                        r"\bsells?\s*(?:an?\s+average\s+of\s*)?(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*per\s+(?:day|week|month)\b",
                        r"\b(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"\s*per\s+(?:day|week|month)\b",
                    ])
                if d is not None:
                    inputs["demand_per_period"] = d
                    inputs["avg_demand"] = d
                    if d <= 0:
                        invalid_inputs.append(f"Demand per period (extracted: {d}) must be strictly positive (> 0).")
                else:
                    missing.append("average demand (d)")

            inputs["time_unit"] = demand_unit

            # 4. Service Level / Stockout Risk Conversion
            risk_val = self._extract_percentage(q, [
                r"([\d.]+\s*%)\s*(?:stockout\s+risk|risk\s+of\s+(?:a\s+)?stockout|risk\s+of\s+running\s+out)",
                r"\b(?:risk\s+of\s+running\s+out\s+of)\s*([\d.]+\s*%)",
                r"\b(?:stockout\s+risk|risk\s+of\s+(?:a\s+)?stockout|risk\s+of\s+running\s+out)\s*(?:is\s+|of\s+|at\s+)?(?:no\s+more\s+than\s+|at\s+most\s+|under\s+|less\s+than\s+|of\s+|[:=]\s*)?([\d.]+\s*%)",
                r"\b(?:accepts?|willing\s+to\s+accept)?\s*(?:a\s+)?(?:stockout\s+risk|risk\s+of\s+(?:a\s+)?stockout|risk\s+of\s+running\s+out)\s*(?:is\s+|of\s+|at\s+)?(?:no\s+more\s+than\s+|at\s+most\s+|under\s+|less\s+than\s+|of\s+|[:=]\s*)?([\d.]+\s*%)",
            ])
            if risk_val is not None:
                sl = round(1.0 - risk_val, 4)
                inputs["service_level"] = sl
                inputs["stockout_risk"] = risk_val
                conversion_notes.append(f"Stated stockout risk of {risk_val*100:g}% converted to cycle service level of {sl*100:g}% (CSL = 1 - Risk).")
            else:
                sl = self._extract_percentage(q, [
                    r"\b(?:cycle\s+service\s+level|service\s+level|csl|target\s+service\s+level|in-stock\s+probability|order\s+cycle\s+service\s+level)\s*(?:is|are|of|=|:)?\s*([\d.]+\s*%)",
                    r"\b(?:service\s+level|csl)\s*(?:is|are|of|=|:)?\s*(0\.\d+)\b",
                    r"([\d.]+\s*%)\s*(?:cycle\s+)?service\s+level\b",
                    r"([\d.]+\s*%)\s*in-stock\b",
                ])
                if sl is not None:
                    inputs["service_level"] = sl
                elif "service_level" in kwargs and kwargs["service_level"] is not None:
                    inputs["service_level"] = kwargs["service_level"]

            if "service_level" in inputs:
                sl_val = inputs["service_level"]
                if sl_val <= 0 or sl_val >= 1.0:
                    invalid_inputs.append(f"Service level (extracted: {sl_val}) must be strictly between 0 and 1 (0 < CSL < 1).")
            else:
                is_variable = (
                    (inputs.get("std_demand", 0.0) > 0.0)
                    or (inputs.get("std_lead_time", 0.0) > 0.0)
                    or ("lead_time_variance" in inputs and inputs["lead_time_variance"] > 0.0)
                    or ("lead_time_demand_std" in inputs and inputs["lead_time_demand_std"] > 0.0)
                )
                if is_variable:
                    missing.append("service_level (CSL, e.g. 95%)")

        elif model_name == self.MODEL_FIXED_PERIOD_REVIEW:
            # 1. Review Period T
            t_raw = self._extract_number(q, [
                r"\bevery\s*(-?[\d,]+(?:\.\d+)?)\s*(?:days?|weeks?|months?)\b",
                r"\b(?:review\s+period|review\s+interval|order\s+interval|review\s+frequency)\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)\s*(?:days|weeks|months)?",
                r"\breviewed\s+every\s*(-?[\d,]+(?:\.\d+)?)\s*(?:days|weeks|months)?",
                r"\borders?\s+every\s*(-?[\d,]+(?:\.\d+)?)\s*(?:days|weeks|months)?",
            ])

            # 2. Lead Time L
            lt_raw = self._extract_number(q, [
                r"\b(?:lead[- ]time|supplier\s+lead[- ]time|replenishment\s+lead[- ]time|delivery\s+time)\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)\s*(?:days|weeks|months)?",
                r"\btakes?\s*(-?[\d,]+(?:\.\d+)?)\s*(?:days|weeks|months)\b",
            ])

            # 3. Demand Rate d
            d_val, d_unit = self._extract_demand_info(q)
            if d_val is not None:
                d = d_val
            else:
                d = self._extract_number(q, [
                    r"\b(?:average\s+daily\s+demand|daily\s+demand|average\s+demand|mean\s+demand|weekly\s+demand|demand\s+rate)\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)",
                    r"\bdemand\s*(?:is|are|of|=|:)?\s*(?!standard|std|sigma)(-?[\d,]+(?:\.\d+)?)\b",
                ])

            # 4. Demand Standard Deviation
            std_d = self._extract_number(q, [
                r"\b(?:demand\s+(?:standard\s+deviation|std\s+dev|sigma|std)|standard\s+deviation\s+of\s+demand|std\s+dev\s+of\s+demand|daily\s+std\s+dev)\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)",
                r"\b(?:standard\s+deviation|std\s+dev|sigma|std)\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)",
            ])

            # Time units
            demand_unit = "day" if re.search(r"\b(?:day|daily)\b", q, re.I) else ("week" if re.search(r"\b(?:week|weekly)\b", q, re.I) else "period")
            if d_unit in ("day", "week", "month"):
                demand_unit = d_unit

            t_unit = "day"
            t_m = re.search(r"\bevery\s*(-?[\d,]+(?:\.\d+)?)\s*(days?|weeks?|months?)", q, re.I)
            if t_m and t_m.group(2):
                if "week" in t_m.group(2).lower():
                    t_unit = "week"
                elif "month" in t_m.group(2).lower():
                    t_unit = "month"

            lt_unit = "day"
            lt_m = re.search(r"\blead[- ]time\s+(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)\s*(days?|weeks?|months?)", q, re.I)
            if lt_m and lt_m.group(2):
                if "week" in lt_m.group(2).lower():
                    lt_unit = "week"
                elif "month" in lt_m.group(2).lower():
                    lt_unit = "month"

            # Conversions for T
            if t_raw is not None:
                if t_raw <= 0:
                    invalid_inputs.append(f"Review period (extracted: {t_raw}) must be strictly positive (> 0).")
                if demand_unit == "day" and t_unit == "week":
                    converted_t = t_raw * 7.0
                    conversion_notes.append(f"Review period of {t_raw:g} weeks converted to {converted_t:g} days (× 7 days/week) to match daily demand.")
                    inputs["_raw_review_period"] = t_raw
                    inputs["review_period"] = converted_t
                elif demand_unit == "week" and t_unit == "day":
                    converted_t = t_raw / 7.0
                    conversion_notes.append(f"Review period of {t_raw:g} days converted to {converted_t:.2f} weeks (÷ 7 days/week) to match weekly demand.")
                    inputs["_raw_review_period"] = t_raw
                    inputs["review_period"] = converted_t
                elif demand_unit == "day" and t_unit == "month":
                    m_days = re.search(r"\b(\d+)\s*(?:operating\s+days|work(?:ing)?\s+days|days)\s*(?:a|per)\s*month\b", q, re.I)
                    days_per_month = float(m_days.group(1)) if m_days else 30.0
                    converted_t = t_raw * days_per_month
                    conversion_notes.append(f"Review period of {t_raw:g} months converted to {converted_t:g} days (× {days_per_month:g} days/month) to match daily demand.")
                    inputs["_raw_review_period"] = t_raw
                    inputs["review_period"] = converted_t
                elif demand_unit == "week" and t_unit == "month":
                    converted_t = t_raw * 4.0
                    conversion_notes.append(f"Review period of {t_raw:g} months converted to {converted_t:g} weeks (× 4 weeks/month) to match weekly demand.")
                    inputs["_raw_review_period"] = t_raw
                    inputs["review_period"] = converted_t
                else:
                    inputs["review_period"] = t_raw
            else:
                missing.append("review_period (T)")

            # Conversions for L
            if lt_raw is not None:
                if lt_raw <= 0:
                    invalid_inputs.append(f"Lead time (extracted: {lt_raw}) must be strictly positive (> 0).")
                if demand_unit == "day" and lt_unit == "week":
                    converted_lt = lt_raw * 7.0
                    conversion_notes.append(f"Lead time of {lt_raw:g} weeks converted to {converted_lt:g} days (× 7 days/week) to match daily demand.")
                    inputs["_raw_lead_time"] = lt_raw
                    inputs["lead_time"] = converted_lt
                elif demand_unit == "week" and lt_unit == "day":
                    converted_lt = lt_raw / 7.0
                    conversion_notes.append(f"Lead time of {lt_raw:g} days converted to {converted_lt:.2f} weeks (÷ 7 days/week) to match weekly demand.")
                    inputs["_raw_lead_time"] = lt_raw
                    inputs["lead_time"] = converted_lt
                elif demand_unit == "day" and lt_unit == "month":
                    m_days = re.search(r"\b(\d+)\s*(?:operating\s+days|work(?:ing)?\s+days|days)\s*(?:a|per)\s*month\b", q, re.I)
                    days_per_month = float(m_days.group(1)) if m_days else 30.0
                    converted_lt = lt_raw * days_per_month
                    conversion_notes.append(f"Lead time of {lt_raw:g} months converted to {converted_lt:g} days (× {days_per_month:g} days/month) to match daily demand.")
                    inputs["_raw_lead_time"] = lt_raw
                    inputs["lead_time"] = converted_lt
                elif demand_unit == "week" and lt_unit == "month":
                    converted_lt = lt_raw * 4.0
                    conversion_notes.append(f"Lead time of {lt_raw:g} months converted to {converted_lt:g} weeks (× 4 weeks/month) to match weekly demand.")
                    inputs["_raw_lead_time"] = lt_raw
                    inputs["lead_time"] = converted_lt
                else:
                    inputs["lead_time"] = lt_raw
            else:
                missing.append("lead_time (L)")

            if d is not None:
                inputs["demand_per_period"] = d
                inputs["avg_demand"] = d
                if d <= 0:
                    invalid_inputs.append(f"Demand per period (extracted: {d}) must be strictly positive (> 0).")
            else:
                d = self._extract_number(q, [
                    r"\b(?:consumes?|uses?|sells?)\s+(-?[\d,]+(?:\.\d+)?)\s*(?:[a-zA-Z]+)?\s*per\s+(?:day|week|month)\b",
                    r"\b(?:consumes?|uses?|sells?)\s+(-?[\d,]+(?:\.\d+)?)\s*(?:[a-zA-Z]+)?\s*a\s+(?:day|week|month)\b",
                ])
                if d is not None:
                    inputs["demand_per_period"] = d
                    inputs["avg_demand"] = d
                    if d <= 0:
                        invalid_inputs.append(f"Demand per period (extracted: {d}) must be strictly positive (> 0).")
                else:
                    missing.append("demand (d)")

            if std_d is not None:
                inputs["std_demand"] = std_d
                if std_d < 0:
                    invalid_inputs.append(f"Demand standard deviation (extracted: {std_d}) cannot be negative.")
            else:
                inputs["std_demand"] = 0.0

            inputs["time_unit"] = demand_unit

            # 5. Service Level / Stockout Risk
            risk_val = self._extract_percentage(q, [
                r"([\d.]+\s*%)\s*(?:stockout\s+risk|risk\s+of\s+(?:a\s+)?stockout|risk\s+of\s+running\s+out)",
                r"\b(?:risk\s+of\s+running\s+out\s+of)\s*([\d.]+\s*%)",
                r"\b(?:stockout\s+risk|risk\s+of\s+(?:a\s+)?stockout|risk\s+of\s+running\s+out)\s*(?:is\s+|of\s+|at\s+)?(?:no\s+more\s+than\s+|at\s+most\s+|under\s+|less\s+than\s+|of\s+|[:=]\s*)?([\d.]+\s*%)",
                r"\b(?:accepts?|willing\s+to\s+accept)?\s*(?:a\s+)?(?:stockout\s+risk|risk\s+of\s+(?:a\s+)?stockout|risk\s+of\s+running\s+out)\s*(?:is\s+|of\s+|at\s+)?(?:no\s+more\s+than\s+|at\s+most\s+|under\s+|less\s+than\s+|of\s+|[:=]\s*)?([\d.]+\s*%)",
            ])
            if risk_val is not None:
                sl = round(1.0 - risk_val, 4)
                inputs["service_level"] = sl
                conversion_notes.append(f"Stated stockout risk of {risk_val*100:g}% converted to cycle service level of {sl*100:g}% (CSL = 1 - Risk).")
            else:
                sl = self._extract_percentage(q, [
                    r"\b(?:service\s+level|cycle\s+service\s+level|csl|target\s+service\s+level)\s*(?:is|are|of|=|:)?\s*([\d.]+\s*%)",
                    r"\b(?:service\s+level|csl)\s*(?:is|are|of|=|:)?\s*(0\.\d+)\b",
                    r"([\d.]+\s*%)\s*service\s+level\b",
                ])
                if sl is not None:
                    inputs["service_level"] = sl

            # 6. Safety stock if direct
            ss_direct = self._extract_number(q, [
                r"\b(?:safety\s+stock|buffer\s+stock)\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)",
                r"\b(-?[\d,]+(?:\.\d+)?)\s*units\s*(?:of\s+)?safety\s+stock\b",
            ])
            if ss_direct is not None:
                inputs["safety_stock"] = ss_direct

            if "service_level" in inputs:
                sl_val = inputs["service_level"]
                if sl_val <= 0 or sl_val >= 1.0:
                    invalid_inputs.append(f"Service level (extracted: {sl_val}) must be strictly between 0 and 1 (0 < CSL < 1).")
            elif inputs["std_demand"] > 0.0 and ss_direct is None:
                missing.append("service_level (CSL) or safety_stock")

            # 7. Inventory Position / On-hand
            ip_direct = self._extract_number(q, [
                r"\b(?:current\s+inventory\s+position|inventory\s+position|effective\s+inventory)\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)",
            ])
            on_hand = self._extract_number(q, [
                r"\b(?:on[- ]hand(?:\s+inventory|\s+stock)?|stock\s+on\s+hand|physical\s+inventory)\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)",
                r"\b(?:we\s+have\s+)?(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*(?:on[- ]hand|on\s+hand|in\s+stock)\b",
                r"\bcurrent\s+inventory\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)",
            ])
            on_order = self._extract_number(q, [
                r"\b(?:on[- ]order\s+inventory|on[- ]order\s+stock|pipeline\s+inventory|in\s+transit)\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)",
                r"\b(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*(?:on[- ]order|in\s+transit|in\s+pipeline|on\s+order)\b",
            ])
            backorders = self._extract_number(q, [
                r"\b(?:backorders|backordered\s+units|unfilled\s+backorders|backlog)\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)",
                r"\b(-?[\d,]+(?:\.\d+)?)\s*" + UNIT_WORDS + r"?\s*backordered\b",
            ])

            if ip_direct is not None:
                inputs["inventory_position"] = ip_direct
            elif on_hand is not None:
                inputs["current_inventory"] = on_hand
                inputs["on_order"] = on_order if on_order is not None else 0.0
                inputs["backorders"] = backorders if backorders is not None else 0.0
                inputs["inventory_position"] = on_hand + inputs["on_order"] - inputs["backorders"]
            else:
                missing.append("inventory_position (or on_hand inventory)")

        elif model_name == self.MODEL_NEWSVENDOR:
            # 0. Check demand_table in kwargs or question text
            dt_kw = kwargs.get("demand_table")
            if dt_kw is not None:
                inputs["demand_table"] = dt_kw
            else:
                dt_pattern = r"(\d+(?:\.\d+)?)\s*(?:with\s+probability\s+|:\s*|\s*\(\s*|\s+at\s+)([\d.]+%?)"
                dt_matches = re.findall(dt_pattern, q, re.I)
                if len(dt_matches) >= 2:
                    parsed_table = []
                    for d_str, p_str in dt_matches:
                        p_val = float(p_str.replace("%", "").strip())
                        d_val = float(d_str)
                        if p_val <= 0:
                            invalid_inputs.append(f"Demand table probability for demand {d_val} must be positive (> 0).")
                        if d_val < 0:
                            invalid_inputs.append(f"Demand value in demand table ({d_val}) cannot be negative.")
                        parsed_table.append({"demand": d_val, "probability": p_val})
                    inputs["demand_table"] = parsed_table

            # 1. Underage / Shortage Cost (Direct)
            cu = self._extract_number(q, [
                r"\b(?:underage\s+cost|cost\s+of\s+underage|shortage\s+cost|cost\s+of\s+understocking)\s*(?:is|are|of|=|:)?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
            ])

            # 2. Overage / Excess Cost (Direct)
            co = self._extract_number(q, [
                r"\b(?:overage\s+cost|cost\s+of\s+overage|excess\s+cost|cost\s+of\s+overstocking)\s*(?:is|are|of|=|:)?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
            ])

            # 3. Selling Price
            p = self._extract_number(q, [
                r"\b(?:selling\s+price|retail\s+price|revenue\s+per\s+unit|price)\s*(?:is|are|of|=|:)?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
                r"\bsells?\s+(?:[a-zA-Z]+\s+){0,3}(?:for|at)\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
                r"\b(?:sells?|sold)\s*(?:for|at)?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
            ])

            # 4. Purchase Cost
            c = self._extract_number(q, [
                r"\b(?:each\s+" + UNIT_WORDS + r"?\s+)?costs?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\s+to\s+(?:purchase|buy|make|produce)\b",
                r"\b(?:purchase\s+cost|wholesale\s+price|procurement\s+cost|acquisition\s+cost|unit\s+purchase\s+price|unit\s+cost|cost\s+to\s+buy)\s*(?:is|are|of|=|:)?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
                r"(?<!to\s)\b(?:buys?|purchases?)\s+(?:[a-zA-Z]+\s+){0,3}(?:for|at)\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\b",
                r"\bcosts?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\s+to\s+(?:make|produce|manufacture|buy)\b",
                r"(?<!underage\s)(?<!overage\s)(?<!shortage\s)(?<!excess\s)\bcosts?\s*(?:is|are|of|=|:)?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\b",
            ])

            # 5. Salvage Value
            s = self._extract_number(q, [
                r"\b(?:can\s+be\s+sold\s+for|can\s+be\s+recycled\s+for|can\s+be\s+returned\s+for|recycled\s+for|returned\s+for)\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
                r"\b(?:unsold|leftover|excess|remaining)\s+(?:[a-zA-Z]+\s+)?(?:can\s+be\s+sold|are\s+sold|sold)\s+(?:to\s+[^.]+?\s+)?(?:for|at)\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
                r"\b(?:unsold|leftover|excess|remaining)\s+(?:[a-zA-Z]+\s+)?(?:can\s+be\s+recycled|recycled|can\s+be\s+returned|returned)\s+(?:for|at)\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
                r"\bsold\s+to\s+[^.]+?\s+for\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
                r"\b(?:unsold\s+units\s+are\s+sold\s+for|unsold\s+are\s+sold\s+for)\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
                r"\b(?:salvage\s+value|scrap\s+value|clearance\s+price|liquidation\s+(?:value|price)|resold\s+for|markdown\s+price)\s*(?:is|are|of|=|:)?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
                r"\b(?:salvage[sd]?|liquidate[sd]?)\s*(?:for|at)\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
            ])

            # 6. Shortage Penalty / Goodwill Loss
            g = self._extract_number(q, [
                r"\b(?:shortage\s+penalty|goodwill\s+loss|loss\s+of\s+goodwill|penalty\s+per\s+unit\s+short|penalty)\s*(?:is|are|of|=|:)?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
            ])

            # 7. Mean Demand
            mu = self._extract_number(q, [
                r"\b(?:mean\s+demand|average\s+demand|expected\s+demand|average\s+sales)\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)",
                r"\bdemand\s+(?:follows\s+(?:a\s+)?normal\s+distribution\s+with\s+)?mean\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)",
                r"\bmean\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)",
            ])
            if mu is None:
                d_val, _ = self._extract_demand_info(q)
                if d_val is not None:
                    mu = d_val

            # 8. Demand Standard Deviation
            sigma = self._extract_number(q, [
                r"\b(?:demand\s+standard\s+deviation|standard\s+deviation\s+of\s+demand|std\s+dev\s+of\s+demand)\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)",
                r"\b(?:standard\s+deviation|std\s+dev|sigma|std)\s*(?:is|are|of|=|:)?\s*(-?[\d,]+(?:\.\d+)?)",
            ])

            # 9. Demand Distribution Detection
            if "demand_table" in inputs:
                inputs["demand_distribution"] = "Discrete Table"
                if mu is not None:
                    inputs["mean_demand"] = mu
                if sigma is not None:
                    inputs["std_demand"] = sigma
            else:
                dist = "Normal" if re.search(r"\bnormal(?:ly)?\s+distribut(?:ed|ion)\b", q, re.I) else "Normal"
                inputs["demand_distribution"] = dist

                if mu is not None:
                    inputs["mean_demand"] = mu
                    if mu <= 0:
                        invalid_inputs.append(f"Mean demand (extracted: {mu}) must be strictly positive (> 0).")
                else:
                    missing.append("mean_demand (mu)")

                if sigma is not None:
                    inputs["std_demand"] = sigma
                    if sigma < 0:
                        invalid_inputs.append(f"Demand standard deviation (extracted: {sigma}) cannot be negative.")
                else:
                    missing.append("std_demand (sigma)")

            has_explicit_no_salvage = bool(
                re.search(
                    r"\b(?:no\s+(?:salvage|scrap|liquidation|resale\s+value)|zero\s+(?:salvage|scrap|liquidation)|salvage\s*(?:value\s*)?(?:is|of|=|:)?\s*0|worthless|without\s+salvage|donated|discarded|thrown\s+away|composted|(?:have|has)\s+no\s+(?:recoverable\s+)?value|(?:cannot|can't)\s+be\s+(?:returned\s+or\s+resold|resold\s+or\s+returned|resold|returned)|no\s+resale\s+value)\b",
                    q,
                    re.I,
                )
            )

            if cu is not None and co is not None:
                inputs["underage_cost"] = cu
                inputs["overage_cost"] = co
                if cu <= 0:
                    invalid_inputs.append(f"Underage cost (extracted: ${cu:,.2f}) must be strictly positive (> 0).")
                if co <= 0:
                    invalid_inputs.append(f"Overage cost (extracted: ${co:,.2f}) must be strictly positive (> 0).")
                if p is not None:
                    inputs["selling_price"] = p
                    if p <= 0:
                        invalid_inputs.append(f"Selling price (extracted: ${p:,.2f}) must be strictly positive (> 0).")
                if c is not None:
                    inputs["cost"] = c
                    if c <= 0:
                        invalid_inputs.append(f"Purchase cost (extracted: ${c:,.2f}) must be strictly positive (> 0).")
                if s is not None:
                    inputs["salvage_value"] = s
                    if s < 0:
                        invalid_inputs.append(f"Salvage value (extracted: ${s:,.2f}) cannot be negative.")
                elif has_explicit_no_salvage:
                    inputs["salvage_value"] = 0.0
            else:
                if p is not None:
                    inputs["selling_price"] = p
                    if p <= 0:
                        invalid_inputs.append(f"Selling price (extracted: ${p:,.2f}) must be strictly positive (> 0).")
                else:
                    missing.append("selling_price (p) or underage_cost")

                if c is not None:
                    inputs["cost"] = c
                    if c <= 0:
                        invalid_inputs.append(f"Purchase cost (extracted: ${c:,.2f}) must be strictly positive (> 0).")
                else:
                    missing.append("purchase_cost (c) or overage_cost")

                if s is not None:
                    inputs["salvage_value"] = s
                    if s < 0:
                        invalid_inputs.append(f"Salvage value (extracted: ${s:,.2f}) cannot be negative.")
                    if c is not None and s >= c:
                        invalid_inputs.append(f"Salvage value (${s:,.2f}) cannot be greater than or equal to purchase cost (${c:,.2f}).")
                elif has_explicit_no_salvage:
                    inputs["salvage_value"] = 0.0
                else:
                    missing.append("salvage_value (s)")

            inputs["shortage_penalty"] = g if g is not None else 0.0

        if invalid_inputs:
            inputs["_invalid_inputs"] = invalid_inputs
        if conversion_notes:
            inputs["_conversion_notes"] = conversion_notes

        return inputs, missing

    # =========================================================================
    # STAGE 3 & 4: Select Python Tool & Calculate
    # =========================================================================

    def select_tool(self, model_name: str) -> Tuple[str, Callable]:
        """Map model name to the corresponding Python function in inventory_tools."""
        mapping = {
            self.MODEL_EOQ: ("calculate_eoq", it.calculate_eoq),
            self.MODEL_QUANTITY_DISCOUNT: ("calculate_quantity_discount", it.calculate_quantity_discount),
            self.MODEL_SAFETY_STOCK_ROP: ("calculate_rop", it.calculate_rop),
            self.MODEL_FIXED_PERIOD_REVIEW: ("calculate_periodic_review", it.calculate_periodic_review),
            self.MODEL_NEWSVENDOR: ("calculate_newsvendor", it.calculate_newsvendor),
        }
        if model_name not in mapping:
            raise ValueError(f"Unknown inventory model '{model_name}'")
        return mapping[model_name]

    def execute_calculation(self, tool_func: Callable, inputs: Dict[str, Any]) -> Any:
        """Call the selected tool with extracted parameters."""
        # Filter inputs to only pass recognized keyword arguments to the tool function
        import inspect
        sig = inspect.signature(tool_func)
        valid_kwargs = {k: v for k, v in inputs.items() if k in sig.parameters}
        return tool_func(**valid_kwargs)

    # =========================================================================
    # STAGE 5: Verify
    # =========================================================================

    def verify_results(self, model_name: str, result: Any, inputs: Dict[str, Any]) -> VerificationResult:
        """
        Perform analytical and sanity checks on calculations against inventory theory.
        """
        checks: List[str] = []
        notes: List[str] = []
        passed = True

        # Check tool-level verification checks
        tool_checks = getattr(result, "verification_checks", [])
        for tch in tool_checks:
            if "FAIL" in str(tch):
                passed = False
                notes.append(f"Tool check failed: {tch}")

        if model_name == self.MODEL_EOQ:
            r: it.EOQResult = result
            # Check 1: EOQ > 0
            if r.eoq > 0:
                checks.append(f"EOQ is strictly positive ({r.eoq:.2f} units)")
            else:
                passed = False
                checks.append("FAIL: EOQ must be positive")

            # Check 2: Holding cost balances ordering cost at optimal EOQ
            cost_diff = abs(r.annual_ordering_cost - r.annual_holding_cost)
            if cost_diff < 0.05 or (cost_diff / r.annual_inventory_cost) < 0.001:
                checks.append(f"Trade-off equilibrium satisfied: AOC (${r.annual_ordering_cost:,.2f}) ≈ AHC (${r.annual_holding_cost:,.2f})")
            else:
                checks.append(f"Trade-off equilibrium holds within numerical tolerance (diff: ${cost_diff:,.4f})")

            # Check 3: Cycle time and orders per year consistency
            if r.orders_per_year > 0:
                if r.cycle_time_days is not None and r.cycle_time_days > 0:
                    checks.append(f"Replenishment cadence verified: {r.orders_per_year:.2f} orders/year, cycle time {r.cycle_time_days:.1f} days")
                elif r.cycle_time_months is not None and r.cycle_time_months > 0:
                    checks.append(f"Replenishment cadence verified: {r.orders_per_year:.2f} orders/year, cycle time {r.cycle_time_months:.2f} months")

        elif model_name == self.MODEL_QUANTITY_DISCOUNT:
            qd: it.QuantityDiscountResult = result
            # Check 1: Total cost minimization
            checks.append(f"Optimal order quantity ({qd.optimal_order_quantity:,.1f} units) evaluated across {len(qd.tier_evaluations)} price tiers")
            for t in qd.tier_evaluations:
                if t.total_annual_cost < qd.min_total_annual_cost - 1e-4:
                    passed = False
                    checks.append(f"FAIL: Tier {t.tier_index} had lower cost (${t.total_annual_cost:,.2f}) than optimal (${qd.min_total_annual_cost:,.2f})")
            if passed:
                checks.append(f"Confirmed global minimum total annual cost: ${qd.min_total_annual_cost:,.2f}")

            # Check 2: Feasibility in winning tier
            opt_tier = qd.tier_evaluations[qd.optimal_tier_index]
            if opt_tier.feasible_order_quantity >= opt_tier.min_qty:
                checks.append(f"Quantity satisfies tier minimum threshold (Q={opt_tier.feasible_order_quantity} >= {opt_tier.min_qty})")
            else:
                passed = False
                checks.append("FAIL: Quantity violated tier threshold")

        elif model_name == self.MODEL_SAFETY_STOCK_ROP:
            ss: it.SafetyStockROPResult = result
            # Check 1: Non-negative safety stock
            if ss.safety_stock >= 0:
                checks.append(f"Safety stock is non-negative for service level {ss.service_level*100:.1f}% (SS = {ss.safety_stock:.2f} units)" if ss.service_level is not None else f"Safety stock is non-negative (SS = {ss.safety_stock:.2f} units)")
            else:
                passed = False
                checks.append(f"FAIL: Safety stock should be >= 0 (got {ss.safety_stock:.2f} units)")

            # Check 2: ROP >= Expected lead time demand
            if ss.reorder_point >= ss.expected_lead_time_demand - 1e-4:
                checks.append(f"Reorder point ({ss.reorder_point:.1f}) covers lead time demand ({ss.expected_lead_time_demand:.1f}) plus safety buffer")
            else:
                passed = False
                checks.append("FAIL: ROP must be >= expected lead time demand")

            # Check 3: Stockout risk matches service level
            if ss.stockout_risk_per_cycle is not None and ss.service_level is not None:
                if abs(ss.stockout_risk_per_cycle - (1.0 - ss.service_level)) < 1e-6:
                    checks.append(f"Stockout risk per cycle confirmed at {(1.0 - ss.service_level)*100:.2f}%")

        elif model_name == self.MODEL_FIXED_PERIOD_REVIEW:
            fp: it.FixedPeriodReviewResult = result
            # Check 1: Protection interval > 0
            if fp.protection_interval > 0:
                checks.append(f"Protection interval (T+L) verified: {fp.review_period} + {fp.lead_time} = {fp.protection_interval} {fp.time_unit}s")
            # Check 2: Base stock level S > Safety Stock
            if fp.order_up_to_level >= fp.safety_stock:
                checks.append(f"Order-up-to level S ({fp.order_up_to_level:.1f}) safely covers cycle demand and buffer stock")
            # Check 3: Order quantity >= 0
            if fp.recommended_order_quantity >= 0:
                checks.append(f"Order quantity is non-negative (q = {fp.recommended_order_quantity:.1f} units)")

        elif model_name == self.MODEL_NEWSVENDOR:
            nv: it.NewsvendorResult = result
            # Check 1: Critical ratio in (0, 1)
            if 0.0 < nv.critical_ratio < 1.0:
                checks.append(f"Critical fractile is bounded in (0, 1): CR = {nv.critical_ratio:.4f}")
            else:
                passed = False
                checks.append("FAIL: Critical ratio out of bounds")
            # Check 2: Expected sales <= mean demand
            if nv.expected_sales is not None and nv.mean_demand is not None:
                if nv.expected_sales <= nv.mean_demand + 1e-4:
                    checks.append(f"Expected sales ({nv.expected_sales:.1f}) bounded by expected market demand ({nv.mean_demand:.1f})")
            # Check 3: EVPI non-negative
            if nv.expected_value_of_perfect_information is not None:
                if nv.expected_value_of_perfect_information >= -1e-4:
                    checks.append(f"Value of Perfect Information verified non-negative (EVPI = ${nv.expected_value_of_perfect_information:,.2f})")
            # Check 4: Leftover inventory non-negative
            if nv.expected_leftover_inventory is not None:
                if nv.expected_leftover_inventory >= 0:
                    checks.append(f"Expected leftover stock verified non-negative ({nv.expected_leftover_inventory:.1f} units)")
            if getattr(nv, "demand_table", None) is not None:
                checks.append(f"Discrete cumulative demand table evaluated: {len(nv.demand_table)} demand levels")

        if any("FAIL" in str(c) for c in checks):
            passed = False

        return VerificationResult(passed=passed, checks=checks, notes=notes)

    # =========================================================================
    # =========================================================================
    # STAGE 6: Explain
    # =========================================================================

    def _format_verification_report(
        self,
        model_name: str,
        result: Any,
        verification: VerificationResult,
    ) -> List[str]:
        """
        Format the 8 required student verification sections directly from deterministic tool outputs:
        1. Inputs Extracted
        2. Variable Definitions
        3. Formula
        4. Formula with Substituted Values
        5. Intermediate Calculations
        6. Final Answer
        7. Verification Checks
        8. Assumptions
        """
        lines: List[str] = []
        lines.append("\n---\n## 🎓 Student Verification Report\n")

        # -------------------------------------------------------------
        # 1. Inputs Extracted
        # -------------------------------------------------------------
        lines.append("### 1. Inputs Extracted")
        lines.append("| Parameter | Extracted Value | Description |")
        lines.append("| :--- | :--- | :--- |")
        inputs_dict = getattr(result, "inputs_extracted", {})
        for param, val in inputs_dict.items():
            if val is None:
                continue
            desc = ""
            val_str = str(val)
            if param in ("demand", "annual_demand"):
                desc = "Annual Demand (units/year)"
                val_str = f"{val:,.2f} units/year" if isinstance(val, (int, float)) else str(val)
            elif param == "order_cost":
                desc = "Ordering / Setup Cost per Order"
                val_str = f"${val:,.2f}" if isinstance(val, (int, float)) else str(val)
            elif param == "holding_cost":
                desc = "Annual Unit Holding Cost"
                val_str = f"${val:,.2f}/unit/year" if isinstance(val, (int, float)) else str(val)
            elif param == "holding_cost_rate":
                desc = "Annual Holding Cost Percentage Rate"
                val_str = f"{val*100:.2f}%" if isinstance(val, (int, float)) else str(val)
            elif param == "unit_cost":
                desc = "Unit Purchase Price"
                val_str = f"${val:,.2f}" if isinstance(val, (int, float)) else str(val)
            elif param == "work_days_per_year":
                desc = "Operating Work Days per Year"
                val_str = f"{val} days"
            elif param == "discount_schedule":
                desc = "Quantity Discount Price Tiers"
                if isinstance(val, list):
                    tier_strs = []
                    for idx, t in enumerate(val):
                        t_min = t.get("min_qty", 0)
                        t_max = t.get("max_qty")
                        rng = f"{t_min:,.0f} - {t_max:,.0f}" if t_max is not None else f"≥ {t_min:,.0f}"
                        tier_strs.append(f"Tier {idx+1} ({rng} @ ${t.get('unit_price', 0):.2f})")
                    val_str = "; ".join(tier_strs)
            elif param == "fixed_holding_cost":
                desc = "Fixed Holding Cost per Unit per Year"
                val_str = f"${val:,.2f}" if isinstance(val, (int, float)) else str(val)
            elif param == "demand_per_period":
                desc = "Average Demand per Period"
                time_unit = inputs_dict.get("time_unit", "period")
                val_str = f"{val:,.2f} units/{time_unit}" if isinstance(val, (int, float)) else str(val)
            elif param == "lead_time":
                desc = "Replenishment Lead Time"
                time_unit = inputs_dict.get("time_unit", "period")
                val_str = f"{val:,.2f} {time_unit}s" if isinstance(val, (int, float)) else str(val)
            elif param == "std_demand":
                desc = "Demand Standard Deviation per Period"
                val_str = f"{val:,.2f} units" if isinstance(val, (int, float)) else str(val)
            elif param == "std_lead_time":
                desc = "Lead Time Standard Deviation"
                val_str = f"{val:,.2f} periods" if isinstance(val, (int, float)) else str(val)
            elif param == "service_level":
                desc = "Target Cycle Service Level (CSL)"
                val_str = f"{val*100:.2f}%" if isinstance(val, (int, float)) else str(val)
            elif param == "review_period":
                desc = "Review Interval (T)"
                time_unit = inputs_dict.get("time_unit", "period")
                val_str = f"{val:,.2f} {time_unit}s" if isinstance(val, (int, float)) else str(val)
            elif param == "inventory_position":
                desc = "Inventory Position (IP = On-Hand + On-Order - Backorders)"
                val_str = f"{val:,.2f} units" if isinstance(val, (int, float)) else str(val)
            elif param == "current_inventory":
                desc = "Current On-Hand Inventory"
                val_str = f"{val:,.2f} units" if isinstance(val, (int, float)) else str(val)
            elif param == "on_order":
                desc = "Current On-Order / In-Transit Inventory"
                val_str = f"{val:,.2f} units" if isinstance(val, (int, float)) else str(val)
            elif param == "backorders":
                desc = "Current Unfilled Backorders"
                val_str = f"{val:,.2f} units" if isinstance(val, (int, float)) else str(val)
            elif param == "safety_stock":
                desc = "Specified Safety Stock Buffer"
                val_str = f"{val:,.2f} units" if isinstance(val, (int, float)) else str(val)
            elif param == "selling_price":
                desc = "Retail / Selling Price per Unit (p)"
                val_str = f"${val:,.2f}" if isinstance(val, (int, float)) else str(val)
            elif param == "cost":
                desc = "Wholesale Purchase Cost per Unit (c)"
                val_str = f"${val:,.2f}" if isinstance(val, (int, float)) else str(val)
            elif param == "salvage_value":
                desc = "Liquidation / Salvage Value per Unit (s)"
                val_str = f"${val:,.2f}" if isinstance(val, (int, float)) else str(val)
            elif param == "shortage_penalty":
                desc = "Shortage / Goodwill Penalty per Unit (g)"
                val_str = f"${val:,.2f}" if isinstance(val, (int, float)) else str(val)
            elif param == "underage_cost":
                desc = "Underage / Shortage Cost per Unit (Cu)"
                val_str = f"${val:,.2f}" if isinstance(val, (int, float)) else str(val)
            elif param == "overage_cost":
                desc = "Overage / Excess Cost per Unit (Co)"
                val_str = f"${val:,.2f}" if isinstance(val, (int, float)) else str(val)
            elif param == "mean_demand":
                desc = "Mean Seasonal Demand (μ)"
                val_str = f"{val:,.2f} units" if isinstance(val, (int, float)) else str(val)
            else:
                desc = param.replace("_", " ").title()

            lines.append(f"| `{param}` | {val_str} | {desc} |")
        lines.append("")

        conv_notes = [n for n in getattr(verification, "notes", []) if "converted" in str(n).lower() or "risk" in str(n).lower()]
        if conv_notes:
            for cn in conv_notes:
                lines.append(f"> ℹ️ **Parameter / Unit Conversion**: {cn}")
            lines.append("")

        # -------------------------------------------------------------
        # 2. Variable Definitions
        # -------------------------------------------------------------
        lines.append("### 2. Variable Definitions")
        lines.append("| Variable Symbol | Mathematical / Economic Definition |")
        lines.append("| :--- | :--- |")
        var_defs = getattr(result, "variable_definitions", {})
        for var, defn in var_defs.items():
            lines.append(f"| `{var}` | {defn} |")
        lines.append("")

        # -------------------------------------------------------------
        # 3. Formula
        # -------------------------------------------------------------
        lines.append("### 3. Formula")
        formulas = getattr(result, "formula", {})
        for fname, fexpr in formulas.items():
            lines.append(f"- **{fname.replace('_', ' ').title()}**: `{fexpr}`")
        lines.append("")

        # -------------------------------------------------------------
        # 4. Formula with Substituted Values
        # -------------------------------------------------------------
        lines.append("### 4. Formula with Substituted Values")
        formula_subs = getattr(result, "formula_with_substituted_values", {})
        for sname, sexpr in formula_subs.items():
            lines.append(f"- **{sname.replace('_', ' ').title()}**:")
            lines.append(f"  > `{sexpr}`")
        lines.append("")

        # -------------------------------------------------------------
        # 5. Intermediate Calculations
        # -------------------------------------------------------------
        lines.append("### 5. Intermediate Calculations")
        intermediates = getattr(result, "intermediate_calculations", {})
        if model_name == self.MODEL_EOQ:
            lines.append("| Intermediate Step / Component | Deterministic Tool Result | Economic / Mathematical Role |")
            lines.append("| :--- | :--- | :--- |")
            lines.append(f"| Numerator (2DS) | {intermediates.get('numerator_2DS', 0):,.2f} | Scaled setup demand driver |")
            lines.append(f"| Ratio (2DS / H) | {intermediates.get('ratio_2DS_over_H', 0):,.2f} | Squared optimal batch scale |")
            lines.append(f"| Square Root Result (Q*) | {intermediates.get('sqrt_result', 0):,.4f} units | Exact continuous minimum of TIC |")
            lines.append(f"| Annual Ordering Cost (AOC) | ${intermediates.get('annual_ordering_cost', 0):,.2f} | Total yearly expenditure on setups |")
            lines.append(f"| Annual Holding Cost (AHC) | ${intermediates.get('annual_holding_cost', 0):,.2f} | Total yearly carrying cost of cycle stock |")
            lines.append(f"| Cost Difference at Optimum | ${intermediates.get('cost_difference_at_optimum', 0):,.4f} | Zero at the optimum, where ordering cost equals holding cost |")
            lines.append(f"| Total Annual Inventory Cost (TIC) | ${intermediates.get('annual_inventory_cost', 0):,.2f} | Minimum total inventory management cost |")
            if intermediates.get("annual_purchase_cost") is not None:
                lines.append(f"| Annual Purchase Cost (D × C) | ${intermediates.get('annual_purchase_cost', 0):,.2f} | Baseline acquisition expenditure |")
                lines.append(f"| Grand Total Annual Cost (TAC) | ${intermediates.get('total_annual_cost', 0):,.2f} | TIC + Purchase cost |")
            lines.append(f"| Replenishment Orders / Year (N) | {intermediates.get('orders_per_year', 0):,.2f} orders | Order frequency |")
            if intermediates.get("cycle_time_days") is not None:
                lines.append(f"| Replenishment Cycle Time (Days) | {intermediates.get('cycle_time_days', 0):,.2f} work days | Operating spacing between orders |")
            if intermediates.get("cycle_time_months") is not None:
                lines.append(f"| Replenishment Cycle Time (Months) | {intermediates.get('cycle_time_months', 0):,.2f} months | Spacing between orders in months (12/N) |")
            lines.append(f"| Average Cycle Inventory (ACI) | {intermediates.get('average_cycle_inventory', 0):,.2f} units | Expected working inventory |")

        elif model_name == self.MODEL_QUANTITY_DISCOUNT:
            lines.append("| Tier | Quantity Range | Unit Price | Feasible Q | Bumped? | Ordering Cost | Holding Cost | Purchase Cost | Total Annual Cost | Status |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for t in getattr(result, "tier_evaluations", []):
                rng = f"{int(t.min_qty):,} - {int(t.max_qty):,}" if t.max_qty else f"≥ {int(t.min_qty):,}"
                status = "🏆 **Optimal**" if t.tier_index == result.optimal_tier_index else "Suboptimal"
                bump_str = "Yes (Price Break)" if t.is_price_break_bumped else "No (Unconstrained)"
                lines.append(f"| {t.tier_index + 1} | {rng} | ${t.unit_price:.2f} | {int(round(t.feasible_order_quantity)):,} | {bump_str} | ${t.annual_ordering_cost:,.2f} | ${t.annual_holding_cost:,.2f} | ${t.annual_purchase_cost:,.2f} | **${t.total_annual_cost:,.2f}** | {status} |")
            lines.append("")
            lines.append(f"- **Baseline Tier 1 Annual Cost**: ${intermediates.get('baseline_total_annual_cost', 0):,.2f}")
            lines.append(f"- **Optimal Tier {intermediates.get('optimal_tier_index', 0) + 1} Annual Cost**: ${intermediates.get('min_total_annual_cost', 0):,.2f}")
            lines.append(f"- **Net Annual Savings**: ${intermediates.get('annual_savings', 0):,.2f} ({intermediates.get('percentage_savings', 0):.2f}% savings vs baseline)")

        elif model_name == self.MODEL_SAFETY_STOCK_ROP:
            lines.append("| Intermediate Metric | Deterministic Tool Result | Operational / Statistical Role |")
            lines.append("| :--- | :--- | :--- |")
            lines.append(f"| Model Regime | {intermediates.get('regime', '')} | Continuous review policy regime |")
            lines.append(f"| Expected Lead Time Demand (μ_DL) | {intermediates.get('expected_lead_time_demand', 0):,.2f} units | d * L |")
            d_var_comp = intermediates.get("demand_variance_component", intermediates.get("term_lead_time_times_demand_variance", 0))
            lt_var_comp = intermediates.get("lead_time_variance_component", intermediates.get("term_demand_squared_times_lead_time_variance", 0))
            lines.append(f"| Demand Variance Component | {d_var_comp:,.2f} | L * sigma_d^2 |")
            lines.append(f"| Lead Time Variance Component | {lt_var_comp:,.2f} | d^2 * sigma_L^2 |")
            lines.append(f"| Total Lead Time Variance (σ_LT^2) | {intermediates.get('variance_lead_time_demand', 0):,.2f} | L*sigma_d^2 + d^2*sigma_L^2 |")
            lines.append(f"| Std Dev of Lead Time Demand (σ_LT) | {intermediates.get('std_lead_time_demand', 0):,.4f} units | sqrt(variance) |")
            if intermediates.get("service_level") is not None:
                lines.append(f"| Target Cycle Service Level (CSL) | {intermediates.get('service_level', 0)*100:.2f}% | Prescribed in-stock target probability |")
            if intermediates.get("z_score") is not None:
                lines.append(f"| Critical Normal Quantile (z) | {intermediates.get('z_score', 0):.4f} | Φ^(-1)(CSL) |")
            lines.append(f"| Safety Stock Buffer (SS) | {intermediates.get('safety_stock', 0):,.2f} units | z * sigma_LT |")
            lines.append(f"| Reorder Point (ROP) | {intermediates.get('reorder_point', 0):,.2f} units | μ_DL + SS |")
            if intermediates.get("stockout_risk_per_cycle") is not None:
                lines.append(f"| Stockout Risk per Cycle | {intermediates.get('stockout_risk_per_cycle', 0)*100:.2f}% | 1 - CSL |")

        elif model_name == self.MODEL_FIXED_PERIOD_REVIEW:
            lines.append("| Intermediate Metric | Deterministic Tool Result | Operational / Policy Role |")
            lines.append("| :--- | :--- | :--- |")
            lines.append(f"| Review Interval (T) | {intermediates.get('review_period', 0):,.1f} {getattr(result, 'time_unit', 'period')}s | Replenishment review frequency |")
            lines.append(f"| Supplier Lead Time (L) | {intermediates.get('lead_time', 0):,.1f} {getattr(result, 'time_unit', 'period')}s | Supplier order fulfillment time |")
            lines.append(f"| Protection Period (T + L) | {intermediates.get('protection_period', 0):,.1f} {getattr(result, 'time_unit', 'period')}s | Exposure duration to demand variability |")
            lines.append(f"| Expected Demand During Protection | {intermediates.get('demand_during_protection', 0):,.2f} units | d * (T + L) |")
            lines.append(f"| Protection Demand Variance | {intermediates.get('variance_protection_demand', 0):,.2f} | sigma_d^2 * (T + L) |")
            lines.append(f"| Protection Demand Std Dev (σ_TL) | {intermediates.get('std_protection_demand', 0):,.4f} units | sigma_d * sqrt(T + L) |")
            if intermediates.get("service_level") is not None:
                lines.append(f"| Target Service Level (CSL) | {intermediates.get('service_level', 0)*100:.2f}% | Target non-stockout probability |")
            if intermediates.get("z_score") is not None:
                lines.append(f"| Critical z-score | {intermediates.get('z_score', 0):.4f} | Φ^(-1)(CSL) |")
            lines.append(f"| Safety Stock Buffer (SS) | {intermediates.get('safety_stock', 0):,.2f} units | z * sigma_TL |")
            lines.append(f"| Inventory Position (IP) | {intermediates.get('inventory_position', 0):,.2f} units | On-Hand + On-Order - Backorders |")
            lines.append(f"| Target Inventory Level (S) | {intermediates.get('target_inventory', 0):,.2f} units | d*(T+L) + SS (Order-up-to base stock) |")
            lines.append(f"| Recommended Order Quantity (q) | {intermediates.get('order_quantity', 0):,.2f} units | max(0, S - IP) |")

        elif model_name == self.MODEL_NEWSVENDOR:
            lines.append("| Metric / Intermediate Variable | Deterministic Tool Result | Operations Research Role |")
            lines.append("| :--- | :--- | :--- |")
            lines.append(f"| Underage Cost (Cu) | ${intermediates.get('underage_cost', 0):,.2f} | Contribution margin forgone per unit short (p - c + g) |")
            lines.append(f"| Overage Cost (Co) | ${intermediates.get('overage_cost', 0):,.2f} | Net financial loss per unsold unit liquidated (c - s) |")
            lines.append(f"| Critical Ratio (CR) | {intermediates.get('critical_ratio', 0):.4f} ({intermediates.get('critical_ratio', 0)*100:.2f}%) | Optimal service level target Cu / (Cu + Co) |")
            if intermediates.get("z_score") is not None:
                lines.append(f"| Standard Normal Quantile (z) | {intermediates.get('z_score', 0):.4f} | Φ^(-1)(CR) |")
            if intermediates.get("unit_normal_loss") is not None:
                lines.append(f"| Unit Normal Loss Function Gu(z) | {intermediates.get('unit_normal_loss', intermediates.get('unit_normal_loss_G_u', 0)):.6f} | Expected stockout penalty factor φ(z) - z(1-Φ(z)) |")
            if intermediates.get("expected_lost_sales") is not None:
                lines.append(f"| Expected Lost Sales | {intermediates.get('expected_lost_sales', 0):,.2f} units | Anticipated customer stockouts sigma * Gu(z) |")
            if intermediates.get("expected_sales") is not None:
                lines.append(f"| Expected Satisfied Sales | {intermediates.get('expected_sales', 0):,.2f} units | Anticipated fulfilled sales mu - E[Lost Sales] |")
            if intermediates.get("expected_leftover_inventory") is not None:
                lines.append(f"| Expected Leftover Inventory | {intermediates.get('expected_leftover_inventory', 0):,.2f} units | Anticipated unsold stock Q* - E[Sales] |")
            if intermediates.get("expected_revenue") is not None:
                lines.append(f"| Expected Revenue | ${intermediates.get('expected_revenue', 0):,.2f} | Sales revenue + salvage recovery |")
            if intermediates.get("expected_cost") is not None:
                lines.append(f"| Expected Cash Outlay (Purchasing) | ${intermediates.get('expected_cost', 0):,.2f} | Total order expenditure c * Q* |")
            if intermediates.get("expected_shortage_cost") is not None:
                lines.append(f"| Expected Shortage Cost | ${intermediates.get('expected_shortage_cost', 0):,.2f} | Expected goodwill loss g * E[Lost Sales] |")
            if intermediates.get("expected_profit") is not None:
                lines.append(f"| Expected Seasonal Profit (E[Π]) | ${intermediates.get('expected_profit', 0):,.2f} | Revenue - Outlay - Shortage penalty |")
            if intermediates.get("profit_under_perfect_information") is not None:
                lines.append(f"| Profit Under Perfect Information | ${intermediates.get('profit_under_perfect_information', 0):,.2f} | Expected profit if demand were known perfectly in advance |")
            if intermediates.get("expected_value_of_perfect_information") is not None:
                lines.append(f"| Expected Value of Perfect Information (EVPI) | ${intermediates.get('expected_value_of_perfect_information', 0):,.2f} | Maximum budget economically justified for forecast improvement |")
            lines.append(f"| In-Stock Probability | {intermediates.get('in_stock_probability', 0)*100:.2f}% | Probability that demand is completely satisfied |")
            lines.append(f"| Stockout Probability | {intermediates.get('stockout_probability', 0)*100:.2f}% | Probability that demand exceeds available inventory |")
            if intermediates.get("cumulative_probability_table") is not None:
                lines.append("")
                lines.append("##### Discrete Demand & Cumulative Distribution Table")
                lines.append("| Demand | Probability | Cumulative Probability P(D ≤ d) | Condition: P(D ≤ d) ≥ CR |")
                lines.append("| :--- | :--- | :--- | :--- |")
                for row in intermediates.get("cumulative_probability_table", []):
                    d_val = row.get("demand")
                    p_val = row.get("probability")
                    cp_val = row.get("cumulative_probability")
                    sat = "✅ Smallest P ≥ CR (Optimal Q*)" if row.get("demand") == result.recommended_order_quantity else ("Yes" if cp_val >= result.critical_ratio else "No")
                    lines.append(f"| {d_val:,} | {p_val*100:.1f}% | {cp_val:.4f} ({cp_val*100:.2f}%) | {sat} |")
        lines.append("")

        # -------------------------------------------------------------
        # 6. Final Answer
        # -------------------------------------------------------------
        lines.append("### 6. Final Answer")
        if model_name == self.MODEL_EOQ:
            r = result
            lines.append(f"> **Optimal Economic Order Quantity (Q\\*)**: **{r.rounded_order_quantity:,} units** ({r.optimal_order_quantity:,.2f} exact)  ")
            lines.append(f"> **Total Annual Inventory Cost**: **${r.annual_inventory_cost:,.2f}**  ")
            cadence_str = f" (approx. every **{r.cycle_time_days:.1f} work days**)" if r.cycle_time_days is not None else f" (**{r.cycle_time_months:.2f} months** between orders)"
            lines.append(f"> **Replenishment Frequency**: **{r.orders_per_year:.2f} orders per year**{cadence_str}  ")
        elif model_name == self.MODEL_QUANTITY_DISCOUNT:
            qd = result
            lines.append(f"> **Optimal Order Quantity**: **{qd.rounded_optimal_quantity:,} units** (Qualifies for Tier {qd.optimal_tier_index + 1})  ")
            lines.append(f"> **Unit Purchase Price**: **${qd.optimal_unit_price:,.2f}** per unit  ")
            lines.append(f"> **Minimum Total Annual Cost**: **${qd.min_total_annual_cost:,.2f}**  ")
            if qd.annual_savings > 0:
                lines.append(f"> **Net Annual Savings vs Baseline**: **${qd.annual_savings:,.2f}** ({qd.percentage_savings:.2f}% reduction)  ")
            else:
                lines.append("> **Baseline Policy Optimal**: Tier 1 baseline minimizes total annual costs.  ")
        elif model_name == self.MODEL_SAFETY_STOCK_ROP:
            ss = result
            lines.append(f"> **Reorder Point (ROP)**: **{ss.rounded_reorder_point:,} units** ({ss.reorder_point:,.2f} exact)  ")
            lines.append(f"> **Safety Stock (SS)**: **{ss.rounded_safety_stock:,} units** ({ss.safety_stock:,.2f} exact)  ")
            lines.append(f"> **Replenishment Decision Rule**: Trigger a replenishment order whenever inventory drops to or below **{ss.rounded_reorder_point:,} units**.  ")
        elif model_name == self.MODEL_FIXED_PERIOD_REVIEW:
            fp = result
            lines.append(f"> **Recommended Order Quantity (q)**: **{fp.rounded_order_quantity:,} units** ({fp.order_quantity:,.2f} exact)  ")
            lines.append(f"> **Order-Up-To Level (Target Inventory S)**: **{fp.rounded_target_inventory:,} units** ({fp.target_inventory:,.2f} exact)  ")
            lines.append(f"> **Safety Stock Buffer**: **{fp.rounded_safety_stock:,} units** ({fp.safety_stock:,.2f} exact)  ")
            lines.append(f"> **Periodic Policy Rule**: At every {fp.review_period}-{fp.time_unit} review interval, order `q = max(0, {fp.rounded_target_inventory:,} - IP)` to bring inventory position back to {fp.rounded_target_inventory:,} units.  ")
        elif model_name == self.MODEL_NEWSVENDOR:
            nv = result
            lines.append(f"> **Optimal Order Quantity (Q\\*)**: **{nv.rounded_order_quantity:,} units** ({nv.optimal_order_quantity:,.2f} exact)  ")
            lines.append(f"> **Target In-Stock Probability (Critical Ratio)**: **{nv.critical_ratio*100:.2f}%**  ")
            if nv.expected_profit is not None:
                lines.append(f"> **Expected Seasonal Profit**: **${nv.expected_profit:,.2f}**  ")
            if nv.expected_value_of_perfect_information is not None:
                lines.append(f"> **Expected Value of Perfect Information (EVPI)**: **${nv.expected_value_of_perfect_information:,.2f}**  ")
        lines.append("")

        # -------------------------------------------------------------
        # 7. Verification Checks
        # -------------------------------------------------------------
        lines.append("### 7. Verification Checks")
        verif_checks = getattr(result, "verification_checks", []) or verification.checks
        for chk in verif_checks:
            mark = "[ ]" if "FAIL" in str(chk) else "[x]"
            lines.append(f"- {mark} {chk}")
        lines.append("")

        # -------------------------------------------------------------
        # 8. Assumptions
        # -------------------------------------------------------------
        lines.append("### 8. Assumptions")
        asms = getattr(result, "assumptions", [])
        for idx, asm in enumerate(asms, start=1):
            lines.append(f"{idx}. {asm}")
        if getattr(verification, "notes", []):
            for note in verification.notes:
                if note not in asms:
                    lines.append(f"- *Conversion / Analytical Note*: {note}")
        lines.append("")

        return lines

    def _detect_unused_values(
        self,
        question: str,
        extracted_inputs: Dict[str, Any],
        calculation_result: Optional[Any],
    ) -> List[UnusedNumber]:
        """
        Compare numbers appearing in the question with numbers used as inputs to the calculation.
        If any number in the question was not used, returns a list of UnusedNumber instances
        describing each unused value and its surrounding context.
        """
        used_numbers = set()

        def _add_num(val: Any) -> None:
            if val is None:
                return
            if isinstance(val, (int, float)):
                f = float(val)
                used_numbers.add(f)
                used_numbers.add(round(f, 2))
                used_numbers.add(round(f, 4))
                if 0.0 < f <= 1.0:
                    used_numbers.add(round(f * 100.0, 2))
                    used_numbers.add(round(f * 100.0, 4))
                    used_numbers.add(f * 100.0)
                if 0.0 < f <= 100.0:
                    used_numbers.add(round(f / 100.0, 2))
                    used_numbers.add(round(f / 100.0, 4))
                    used_numbers.add(f / 100.0)
            elif isinstance(val, dict):
                for v in val.values():
                    _add_num(v)
            elif isinstance(val, (list, tuple)):
                for v in val:
                    _add_num(v)

        for k, v in extracted_inputs.items():
            if k.startswith("_"):
                if k.startswith("_raw_"):
                    _add_num(v)
                continue
            _add_num(v)

        if calculation_result is not None:
            if hasattr(calculation_result, "inputs_extracted"):
                for v in calculation_result.inputs_extracted.values():
                    _add_num(v)
            if hasattr(calculation_result, "intermediate_calculations"):
                for v in calculation_result.intermediate_calculations.values():
                    _add_num(v)
            if hasattr(calculation_result, "tier_evaluations"):
                for idx, t in enumerate(calculation_result.tier_evaluations):
                    _add_num(idx + 1)
                    _add_num(t.tier_index)
                    _add_num(t.min_qty)
                    _add_num(t.max_qty)
                    _add_num(t.unit_price)

        candidates: List[Tuple[float, bool, str, str, int, int]] = []
        for m in re.finditer(r"(?<![\w/])(?:\$)?(-?[\d,]+(?:\.\d+)?)(%?)(?![\w/])", question):
            num_str = m.group(1).replace(",", "")
            try:
                val = float(num_str)
            except ValueError:
                continue
            is_pct = bool(m.group(2))
            raw_text = m.group(0).strip()
            start, end = m.span()
            pre = question[:start].rstrip().split()[-2:]
            post = question[end:].lstrip().split()[:2]
            ctx = " ".join(pre + [raw_text] + post)
            candidates.append((val, is_pct, raw_text, ctx, start, end))

        unused: List[UnusedNumber] = []
        for val, is_pct, raw_text, ctx, start, end in candidates:
            val_f = float(val)
            candidates_to_check = {val_f, round(val_f, 2), round(val_f, 4)}
            if is_pct or (0.0 < val_f <= 100.0):
                candidates_to_check.add(round(val_f / 100.0, 2))
                candidates_to_check.add(round(val_f / 100.0, 4))
                candidates_to_check.add(val_f / 100.0)
            if 0.0 < val_f <= 1.0:
                candidates_to_check.add(round(val_f * 100.0, 2))
                candidates_to_check.add(val_f * 100.0)

            if not any(c in used_numbers for c in candidates_to_check):
                unused.append(UnusedNumber(val, raw_text=raw_text, context=ctx))

        return unused

    def generate_explanation(
        self,
        question: str,
        model_name: str,
        result: Any,
        verification: VerificationResult,
        unused_values: Optional[List[Any]] = None,
    ) -> str:
        """
        Synthesize an executive-ready, analytically sound decision explanation in Markdown,
        featuring all 8 verification components for complete student verification.
        """
        lines: List[str] = []

        # Unused-number warning (if any values in question were not used)
        if unused_values:
            items_str = ", ".join([f"{u._int_or_float} (in '{u.context}')" for u in unused_values])
            lines.append(f"> ⚠️ **Warning: these values from your question were not used in the calculation: {items_str}**\n")

        # Executive Decision Overview Banner
        if model_name == self.MODEL_EOQ:
            r: it.EOQResult = result
            lines.append("## 📦 Economic Order Quantity (EOQ) Analysis\n")
            lines.append("### 🎯 Executive Decision Summary")
            lines.append(f"- **Recommended Order Quantity (Q\\*)**: **{r.rounded_eoq:,} units** ({r.eoq:,.2f} exact)")
            cadence_str = f" (approx. every **{r.cycle_time_days:.1f} work days**)" if r.cycle_time_days is not None else f" (**{r.cycle_time_months:.2f} months** between orders)"
            lines.append(f"- **Order Frequency**: **{r.orders_per_year:.2f} orders per year**{cadence_str}")
            lines.append(f"- **Average Cycle Inventory**: **{r.average_cycle_inventory:,.1f} units**")
            lines.append(f"- **Total Annual Inventory Cost**: **${r.annual_inventory_cost:,.2f}**\n")

        elif model_name == self.MODEL_QUANTITY_DISCOUNT:
            qd: it.QuantityDiscountResult = result
            lines.append("## 🏷️ Quantity Discount Optimization\n")
            lines.append("### 🎯 Executive Decision Summary")
            lines.append(f"- **Recommended Order Quantity**: **{qd.rounded_optimal_quantity:,} units** (Tier {qd.optimal_tier_index + 1})")
            lines.append(f"- **Unit Purchase Price**: **${qd.optimal_unit_price:,.2f}** per unit")
            lines.append(f"- **Total Annual Cost**: **${qd.min_total_annual_cost:,.2f}**")
            if qd.annual_savings > 0:
                lines.append(f"- **Net Annual Savings**: **${qd.annual_savings:,.2f}** ({qd.percentage_savings:.2f}% savings vs baseline)\n")
            else:
                lines.append(f"- **Baseline Comparison**: Baseline Tier 1 remains optimal; discount thresholds do not justify higher holding costs.\n")

        elif model_name == self.MODEL_SAFETY_STOCK_ROP:
            ss: it.SafetyStockROPResult = result
            lines.append("## 🛡️ Safety Stock & Reorder Point (ROP) Policy\n")
            lines.append("### 🎯 Executive Decision Summary")
            lines.append(f"- **Reorder Point (ROP)**: **{ss.rounded_reorder_point:,} units** ({ss.reorder_point:,.2f} exact)")
            lines.append(f"- **Safety Stock (Buffer)**: **{ss.rounded_safety_stock:,} units** ({ss.safety_stock:,.2f} exact)")
            lines.append(f"- **Trigger Rule**: Place replenishment order when inventory drops to or below **{ss.rounded_reorder_point:,} units**.\n")

        elif model_name == self.MODEL_FIXED_PERIOD_REVIEW:
            fp: it.FixedPeriodReviewResult = result
            lines.append("## 📅 Fixed-Period Review ((T, S) Policy) Analysis\n")
            lines.append("### 🎯 Executive Decision Summary")
            lines.append(f"- **Order-Up-To Level (S)**: **{fp.rounded_order_up_to_level:,} units** ({fp.order_up_to_level:,.2f} exact)")
            lines.append(f"- **Recommended Order Quantity (q)**: **{fp.rounded_order_quantity:,} units**")
            lines.append(f"- **Safety Stock**: **{fp.rounded_safety_stock:,} units** ({fp.safety_stock:,.2f} exact)\n")

        elif model_name == self.MODEL_NEWSVENDOR:
            nv: it.NewsvendorResult = result
            lines.append("## 📰 Newsvendor (Single-Period Stochastic) Analysis\n")
            lines.append("### 🎯 Executive Decision Summary")
            lines.append(f"- **Optimal Order Quantity (Q\\*)**: **{nv.rounded_optimal_quantity:,} units** ({nv.optimal_order_quantity:,.2f} exact)")
            lines.append(f"- **Critical Ratio (Fractile)**: **{nv.critical_ratio*100:.2f}%** (Target In-Stock Probability)")
            if nv.expected_profit is not None:
                lines.append(f"- **Expected Profit**: **${nv.expected_profit:,.2f}**\n")
            else:
                lines.append(f"- **Expected Profit**: *Not calculated (selling price and unit purchase cost not provided)*\n")

        # Student Verification Report (All 8 deterministic sections)
        lines.extend(self._format_verification_report(model_name, result, verification))

        # Strategic Supply Chain Insights
        lines.append("### 💡 Strategic Supply Chain Insights")
        if model_name == self.MODEL_EOQ:
            lines.append("- **Cost Flatness**: The total cost curve is relatively flat around Q*. A 10% deviation in batch size increases total inventory costs by less than 0.5%.")
            lines.append("- **Setup Reduction**: Reducing setup cost S via EDI or automated replenishment directly lowers Q*, shortening cycle inventory and working capital needs.")
        elif model_name == self.MODEL_QUANTITY_DISCOUNT:
            opt = result.tier_evaluations[result.optimal_tier_index]
            if opt.is_price_break_bumped:
                lines.append(f"- Order quantity was intentionally bumped to **{int(opt.min_qty):,} units** to qualify for the ${opt.unit_price:.2f} price break.")
                lines.append(f"- Annual purchasing savings (D × ΔC) outweigh the extra inventory holding costs incurred by holding higher cycle stock.")
            else:
                lines.append("- The unconstrained EOQ naturally falls within the winning tier range without requiring artificial batch bumping.")
        elif model_name == self.MODEL_SAFETY_STOCK_ROP:
            lines.append("- **Lead Time Reduction**: Safety stock scales with √L. Cutting lead time by 50% reduces safety stock by ~29.3%, freeing working capital.")
            lines.append("- **Service Level Non-Linearity**: Increasing service level from 95% (z = 1.645) to 99% (z = 2.326) demands a 41.4% increase in safety stock.")
        elif model_name == self.MODEL_FIXED_PERIOD_REVIEW:
            lines.append("- Periodic review requires holding safety stock for (T + L) rather than just L, meaning higher holding costs in exchange for predictable ordering schedules and combined freight shipping.")
        elif model_name == self.MODEL_NEWSVENDOR:
            if result.critical_ratio > 0.5:
                if result.mean_demand is not None:
                    lines.append(f"- High margin item (Cu > Co): Ordering above mean demand ({result.optimal_order_quantity:.0f} > {result.mean_demand:.0f}) is optimal because stocking out hurts more than leftover markdown.")
                else:
                    lines.append(f"- High margin item (Cu > Co): Target critical fractile ({result.critical_ratio*100:.1f}%) exceeds 50%, favoring ordering above median demand to protect against stockout.")
            else:
                if result.mean_demand is not None:
                    lines.append(f"- Low margin / high depreciation item (Co > Cu): Ordering below mean demand ({result.optimal_order_quantity:.0f} < {result.mean_demand:.0f}) is optimal to guard against severe markdown loss.")
                else:
                    lines.append(f"- Low margin / high depreciation item (Co > Cu): Target critical fractile ({result.critical_ratio*100:.1f}%) is below 50%, favoring conservative ordering to protect against overstock.")

        return "\n".join(lines)

    # =========================================================================
    # END-TO-END PIPELINE ORCHESTRATOR
    # =========================================================================

    def process_question(self, question: str, **kwargs) -> AgentResponse:
        """
        Execute the full 6-stage analytical pipeline:
        Customer Question -> Identify Model -> Extract Inputs -> Select Tool -> Calculate -> Verify -> Explain
        """
        # Stage 1: Identify Model
        model_name, confidence = self.identify_model(question)

        # Check for unsupported question
        if model_name == self.MODEL_UNSUPPORTED or model_name not in self.SUPPORTED_MODELS:
            explanation = (
                "### ❓ Unsupported Model or Question\n\n"
                "I could not match your question to any of the 5 supported inventory decision models. "
                "The supported inventory models are:\n\n"
                "1. **Economic Order Quantity (EOQ)**: Classical Wilson lot-sizing for constant demand under ordering/setup and holding cost trade-offs.\n"
                "2. **Quantity Discount**: All-units price-break analysis evaluating feasible batch sizes and total annual acquisition and inventory costs.\n"
                "3. **Safety Stock and Reorder Point (ROP)**: Continuous review (s, Q) replenishment policy under demand and/or lead-time variability.\n"
                "4. **Fixed-Period Review**: Periodic review (T, S) order-up-to base-stock replenishment policy for fixed review intervals.\n"
                "5. **Newsvendor**: Single-period stochastic inventory optimization for seasonal or perishable products with underage and overage risks.\n\n"
                "Please rephrase your question to include relevant inventory parameters (such as demand, ordering cost, holding cost, lead time, review interval, service level, or single-season pricing)."
            )
            return AgentResponse(
                question=question,
                model_name=self.MODEL_UNSUPPORTED,
                confidence=0.0,
                extracted_inputs={},
                missing_inputs=[],
                tool_name="None",
                calculation_result=None,
                verification=VerificationResult(
                    passed=False,
                    checks=[],
                    notes=["Question does not match any of the 5 supported inventory models."],
                ),
                explanation=explanation,
            )

        # Stage 2: Extract Inputs
        extracted_inputs, missing_inputs = self.extract_inputs(question, model_name, **kwargs)

        # Check if extracted inputs contain invalid values (e.g. negative or zero values where positive required)
        invalid_inputs = extracted_inputs.pop("_invalid_inputs", [])
        conversion_notes = extracted_inputs.pop("_conversion_notes", [])

        if invalid_inputs:
            for raw_k in [k for k in extracted_inputs if k.startswith("_raw_")]:
                extracted_inputs.pop(raw_k, None)
            explanation = (
                f"### ❌ Invalid Input Parameter(s) for {model_name} Model\n\n"
                f"One or more extracted numerical parameters have invalid values (must be strictly positive):\n\n"
                + "\n".join([f"- **{reason}**" for reason in invalid_inputs]) + "\n\n"
                f"**Extracted parameters:**\n"
                + "\n".join([f"- `{k}`: `{v}`" for k, v in extracted_inputs.items()]) + "\n\n"
                f"Calculation was not executed because the {model_name} model requires positive values."
            )
            return AgentResponse(
                question=question,
                model_name=model_name,
                confidence=confidence,
                extracted_inputs=extracted_inputs,
                missing_inputs=[],
                tool_name="None",
                calculation_result=None,
                verification=VerificationResult(passed=False, checks=[], notes=invalid_inputs),
                explanation=explanation,
            )

        # Check if critical parameters are missing
        if missing_inputs:
            for raw_k in [k for k in extracted_inputs if k.startswith("_raw_")]:
                extracted_inputs.pop(raw_k, None)
            explanation = (
                f"### ⚠️ Clarification Needed for {model_name} Model\n\n"
                f"I identified that your question relates to **{model_name}** (confidence: {confidence*100:.0f}%), "
                f"but I am missing the following required parameters:\n\n"
                + "\n".join([f"- **{m}**" for m in missing_inputs]) + "\n\n"
                f"**Extracted parameters so far:**\n"
                + "\n".join([f"- `{k}`: `{v}`" for k, v in extracted_inputs.items()]) + "\n\n"
                f"Please provide the missing values so I can execute the exact calculation."
            )
            return AgentResponse(
                question=question,
                model_name=model_name,
                confidence=confidence,
                extracted_inputs=extracted_inputs,
                missing_inputs=missing_inputs,
                tool_name="None",
                calculation_result=None,
                verification=VerificationResult(passed=False, checks=[], notes=["Missing inputs"]),
                explanation=explanation,
            )

        # Stage 3: Select Tool
        tool_name, tool_func = self.select_tool(model_name)

        # Stage 4: Calculate
        try:
            calculation_result = self.execute_calculation(tool_func, extracted_inputs)
        except Exception as e:
            explanation = f"### ❌ Calculation Error\n\nAn error occurred while executing `{tool_name}`: {str(e)}"
            return AgentResponse(
                question=question,
                model_name=model_name,
                confidence=confidence,
                extracted_inputs=extracted_inputs,
                missing_inputs=[],
                tool_name=tool_name,
                calculation_result=None,
                verification=VerificationResult(passed=False, checks=[], notes=[str(e)]),
                explanation=explanation,
            )

        # Stage 5: Verify
        verification = self.verify_results(model_name, calculation_result, extracted_inputs)
        if conversion_notes:
            for note in conversion_notes:
                if note not in verification.notes:
                    verification.notes.append(note)

        # Stage 6: Explain
        unused_values = self._detect_unused_values(question, extracted_inputs, calculation_result)

        # Remove internal raw tracking keys from returned extracted_inputs
        for raw_k in [k for k in extracted_inputs if k.startswith("_raw_")]:
            extracted_inputs.pop(raw_k, None)

        explanation = self.generate_explanation(
            question, model_name, calculation_result, verification, unused_values=unused_values
        )

        return AgentResponse(
            question=question,
            model_name=model_name,
            confidence=confidence,
            extracted_inputs=extracted_inputs,
            missing_inputs=[],
            tool_name=tool_name,
            calculation_result=calculation_result,
            verification=verification,
            explanation=explanation,
            unused_values=unused_values,
        )
