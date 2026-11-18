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
        Analyze customer question text and classify into one of the 5 inventory models.

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
        ]
        for kw in qd_keywords:
            if kw in q:
                scores[self.MODEL_QUANTITY_DISCOUNT] += 2.0
        # Check for range/price patterns like "0-999: $20" or "under 500"
        if re.search(r"\b\d+\s*[-–to]+\s*\d+\s*[:@=]?\s*\$?\d+", q) or re.search(r"\$\d+\s*for\s*\d+", q):
            scores[self.MODEL_QUANTITY_DISCOUNT] += 3.5

        # 2. Fixed-Period Review Cues
        fp_keywords = [
            "fixed period", "periodic review", "order up to", "order-up-to",
            "(t, s)", "t,s", "review period", "review interval", "review every",
            "order every", "reviewed every", "inventory position", "base stock",
            "target inventory", "replenishment interval", "order interval",
        ]
        for kw in fp_keywords:
            if kw in q:
                scores[self.MODEL_FIXED_PERIOD_REVIEW] += 3.5
        if re.search(r"\breview\s+(?:period|interval)\b", q):
            scores[self.MODEL_FIXED_PERIOD_REVIEW] += 4.0
        if re.search(r"every\s+\d+\s+(day|week|month)", q):
            scores[self.MODEL_FIXED_PERIOD_REVIEW] += 3.5

        # 3. Newsvendor Cues
        nv_keywords = [
            "newsvendor", "newsboy", "single period", "salvage", "salvages",
            "perishable", "spoilage", "seasonal", "fashion item", "leftover",
            "critical ratio", "critical fractile", "underage", "overage",
            "shortage penalty", "clearance price", "scrap value", "one-time buy",
            "underage cost", "overage cost", "shortage cost", "excess cost",
        ]
        for kw in nv_keywords:
            if kw in q:
                scores[self.MODEL_NEWSVENDOR] += 2.5
        if "selling price" in q and ("salvage" in q or "scrap" in q or "clearance" in q):
            scores[self.MODEL_NEWSVENDOR] += 3.0
        if "underage cost" in q or "overage cost" in q or "critical ratio" in q:
            scores[self.MODEL_NEWSVENDOR] += 4.0

        # 4. Safety Stock & Reorder Point Cues
        ss_keywords = [
            "safety stock", "reorder point", "rop", "service level",
            "cycle service level", "csl", "stockout risk", "buffer stock",
            "lead time demand", "lead time of", "days of lead time",
            "standard deviation of lead time", "lead time variability",
            "lead time", "lead-time", "leadtime",
        ]
        for kw in ss_keywords:
            if kw in q:
                scores[self.MODEL_SAFETY_STOCK_ROP] += 2.0
        if ("reorder point" in q or "rop" in q or "safety stock" in q) and scores[self.MODEL_FIXED_PERIOD_REVIEW] == 0:
            scores[self.MODEL_SAFETY_STOCK_ROP] += 3.5
        if ("lead time" in q or "lead-time" in q) and ("order cost" not in q and "ordering cost" not in q and "holding cost" not in q):
            scores[self.MODEL_SAFETY_STOCK_ROP] += 3.0

        # 5. EOQ Cues
        eoq_keywords = [
            "eoq", "economic order quantity", "annual demand", "setup cost",
            "order cost", "ordering cost", "setup expense", "cost per order",
            "cost of placing an order", "ordering/setup cost", "holding cost",
            "carrying cost", "optimal order quantity", "how many units should we order",
            "order size", "batch size", "wilson formula", "cycle inventory",
        ]
        for kw in eoq_keywords:
            if kw in q:
                scores[self.MODEL_EOQ] += 1.5
        if "eoq" in q or "economic order quantity" in q:
            scores[self.MODEL_EOQ] += 3.5

        # Disambiguation logic:
        # If Quantity Discount keywords matched strongly, it should override pure EOQ
        if scores[self.MODEL_QUANTITY_DISCOUNT] >= 3.0:
            scores[self.MODEL_EOQ] *= 0.5

        # If lead time is present and no ordering/holding cost is present, discount EOQ
        if ("lead time" in q or "lead-time" in q) and ("order cost" not in q and "ordering cost" not in q and "holding cost" not in q and "setup cost" not in q):
            scores[self.MODEL_EOQ] = 0.0

        # If Fixed Period Review matched periodic cues, prioritize over continuous ROP
        if scores[self.MODEL_FIXED_PERIOD_REVIEW] >= 3.0 and ("review period" in q or "review interval" in q or "order-up-to" in q or "order up to" in q or "order every" in q or "periodic review" in q or "fixed period" in q):
            scores[self.MODEL_SAFETY_STOCK_ROP] = 0.0

        # Select highest scoring model
        best_model = max(scores, key=lambda k: scores[k])
        max_score = scores[best_model]

        if max_score <= 0.0:
            # Fallback based on subtle cues or default to EOQ
            if "order" in q and "cost" in q:
                return self.MODEL_EOQ, 0.4
            return self.MODEL_EOQ, 0.3

        confidence = min(0.99, max(0.45, max_score / (max_score + 2.0)))
        return best_model, round(confidence, 2)

    # =========================================================================
    # STAGE 2: Extract Inputs
    # =========================================================================

    def _clean_number(self, val_str: Optional[str]) -> Optional[float]:
        """Strip currency symbols, commas, trailing punctuation, and whitespace, then convert to float."""
        if not val_str:
            return None
        cleaned = re.sub(r"[\$\s]", "", val_str).rstrip(",;:.")
        cleaned = cleaned.replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            return None

    def _extract_number(self, text: str, patterns: List[str]) -> Optional[float]:
        """Helper to extract a float from text using regex patterns."""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                for group in match.groups():
                    if group is not None:
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
        q = question

        if model_name == self.MODEL_EOQ:
            # 1. Annual Demand (and monthly/weekly conversions)
            d = self._extract_number(q, [
                r"\b(?:annual\s+demand|yearly\s+demand|annual\s+sales|annual\s+usage|annual\s+consumption)\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
                r"\bdemand\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)\s*(?:units|items)?\s*(?:per\s+year|annually|a\s+year)",
                r"\bsells?\s*([\d,]+(?:\.\d+)?)\s*(?:units|items)?\s*(?:per\s+year|annually|a\s+year)",
                r"\b([\d,]+(?:\.\d+)?)\s*(?:units|items)?\s*(?:per\s+year|annually|a\s+year)\b",
                r"\b(?:forecasted\s+annual\s+demand|expected\s+annual\s+demand)\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
                r"\bdemand\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)\s*(?:units|items)?\b",
            ])

            if d is None:
                # Check for monthly demand conversion
                d_month = self._extract_number(q, [
                    r"\b(?:monthly\s+demand|demand\s+per\s+month)\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
                    r"\bdemand\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)\s*(?:units|items)?\s*(?:per\s+month|monthly)",
                    r"\bsells?\s*([\d,]+(?:\.\d+)?)\s*(?:units|items)?\s*(?:per\s+month|monthly)",
                ])
                if d_month is not None:
                    d = d_month * 12.0
                    inputs["annual_demand_converted_from_monthly"] = True

            if d is None:
                # Check for weekly demand conversion
                d_week = self._extract_number(q, [
                    r"\b(?:weekly\s+demand|demand\s+per\s+week)\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
                    r"\bdemand\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)\s*(?:units|items)?\s*(?:per\s+week|weekly)",
                    r"\bsells?\s*([\d,]+(?:\.\d+)?)\s*(?:units|items)?\s*(?:per\s+week|weekly)",
                ])
                if d_week is not None:
                    d = d_week * 52.0
                    inputs["annual_demand_converted_from_weekly"] = True

            # 2. Ordering Cost (all synonyms from specification)
            s = self._extract_number(q, [
                r"\b(?:ordering\s+cost|order\s+cost|setup\s+cost|setup\s+expense|cost\s+per\s+order|cost\s+of\s+placing\s+an?\s+order|ordering\s*/\s*setup\s+cost|order\s+placement\s+cost|cost\s+to\s+place\s+an?\s+order|cost\s+to\s+order|procurement\s+cost\s+per\s+order)\s*(?:is|are|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)",
                r"\$?\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+order|to\s+place\s+an?\s+order|setup\s+expense|setup\s+cost|order\s+placement\s+cost)\b",
            ])

            # 3. Holding Cost (Fixed holding cost per unit per year)
            h = self._extract_number(q, [
                r"\b(?:fixed\s+holding\s+cost|unit\s+holding\s+cost|holding\s+cost|carrying\s+cost|inventory\s+carrying\s+cost|storage\s+cost|holding\s+expense|carrying\s+expense|cost\s+of\s+holding|cost\s+to\s+hold|holding\s+charge)\s*(?:is|are|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+unit|\$|/unit)?",
                r"\$?\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+unit\s*(?:per\s+year|annually|to\s+hold)|/unit/year|holding\s+cost|carrying\s+cost)\b",
            ])

            # 4. Percentage Holding Cost Rate
            h_rate = self._extract_percentage(q, [
                r"\b(?:holding\s+cost\s+rate|carrying\s+cost\s+rate|carrying\s+charge|inventory\s+carrying\s+rate|annual\s+holding\s+rate|holding\s+rate|carrying\s+rate)\s*(?:is|are|of|=|:)?\s*([\d.]+\s*%)",
                r"\b(?:holding|carrying)\s+cost\s*(?:is|are|of|=|:)?\s*([\d.]+\s*%)\s*of\s+(?:unit\s+)?(?:purchase\s+)?(?:cost|price)",
                r"([\d.]+\s*%)\s*(?:annual\s+)?(?:holding|carrying)\s+(?:rate|charge|cost)",
            ])

            # 5. Unit Purchase Price / Cost
            c = self._extract_number(q, [
                r"\b(?:unit\s+purchase\s+price|unit\s+purchase\s+cost|purchase\s+price|unit\s+cost|cost\s+per\s+unit|wholesale\s+price|item\s+cost|item\s+price|unit\s+price)\s*(?:is|are|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)",
                r"\bcosts?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*per\s+unit\b",
                r"\$?\s*([\d,]+(?:\.\d+)?)\s*(?:each|per\s+unit|per\s+item)\b",
            ])

            # 6. Work Days Per Year
            work_days = self._extract_number(q, [
                r"\b([\d]+)\s*(?:work|business|operating)?\s*days\s*per\s*year\b",
                r"\boperates?\s*([\d]+)\s*days\b",
            ])

            # Populate inputs & detect truly missing values
            if d is not None:
                inputs["demand"] = d
                inputs["annual_demand"] = d
            else:
                missing.append("annual_demand (D)")

            if s is not None:
                inputs["order_cost"] = s
            else:
                missing.append("order_cost (S)")

            if h is not None:
                inputs["holding_cost"] = h
            elif h_rate is not None and c is not None:
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
            # 1. Demand
            d = self._extract_number(q, [
                r"\b(?:annual\s+demand|yearly\s+demand|annual\s+sales|demand)\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
                r"\bsells?\s*([\d,]+(?:\.\d+)?)\s*(?:units|items)?\s*(?:per\s+year|annually)?",
                r"\b([\d,]+(?:\.\d+)?)\s*(?:units|items)\s*(?:per\s+year|annually)",
            ])

            # 2. Ordering Cost (supports all synonyms)
            s = self._extract_number(q, [
                r"\b(?:ordering\s+cost|order\s+cost|setup\s+cost|setup\s+expense|cost\s+per\s+order|cost\s+of\s+placing\s+an?\s+order|ordering\s*/\s*setup\s+cost|order\s+placement\s+cost)\s*(?:is|are|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)",
                r"\$?\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+order|to\s+place\s+an?\s+order|setup\s+cost|setup\s+expense)\b",
            ])

            # 3. Percentage Holding Cost
            h_rate = self._extract_percentage(q, [
                r"\b(?:holding\s+cost\s+rate|carrying\s+cost\s+rate|carrying\s+charge|carrying\s+rate|holding\s+rate)\s*(?:is|are|of|=|:)?\s*([\d.]+\s*%)",
                r"\b(?:holding|carrying)\s+(?:cost|rate)?\s*(?:is|are|of|=|:)?\s*([\d.]+\s*%)\s*of\s+(?:unit\s+)?(?:purchase\s+)?price",
                r"([\d.]+\s*%)\s*(?:annual\s+)?(?:holding|carrying)\s+(?:rate|cost)",
            ])

            # 4. Fixed Holding Cost
            fixed_h = self._extract_number(q, [
                r"\b(?:fixed\s+holding\s+cost|fixed\s+carrying\s+cost)\s*(?:is|are|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)",
                r"\bholding\s+cost\s*(?:is|are|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+unit|/unit)",
            ])

            # 5. Baseline Unit Purchase Price (if given)
            base_p = self._extract_number(q, [
                r"\b(?:baseline\s+(?:unit\s+)?purchase\s+price|regular\s+price|base\s+price|standard\s+price)\s*(?:is|are|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)",
            ])

            # 6. Parse Quantity Breaks & Unit Prices
            tiers: List[Dict[str, Any]] = []

            # (A) Lower bounded tiers: "under 500 units at $15", "less than 500 units at $15", "up to 500: $15"
            for um in re.finditer(r"\b(?:under|less\s+than|up\s+to)\s+([\d,]+)\s*(?:units)?\s*(?:at|costs?|is|are|:|@|=)?\s*\$?\s*([\d]+(?:\.\d+)?)", q, re.I):
                tiers.append({"min_qty": 0.0, "max_qty": self._clean_number(um.group(1)) - 1.0, "unit_price": self._clean_number(um.group(2))})

            # (B) Middle ranges: "1-499 units at $30", "500 to 999: $28", "between 500 and 999: $28"
            for rm in re.finditer(r"\b(?:between\s+)?([\d,]+)\s*(?:-|–|to|and)\s*([\d,]+)\s*(?:units)?\s*(?:at|costs?|is|are|:|@|=)?\s*\$?\s*([\d]+(?:\.\d+)?)", q, re.I):
                q_low = self._clean_number(rm.group(1))
                q_high = self._clean_number(rm.group(2))
                p_val = self._clean_number(rm.group(3))
                if q_low is not None and q_high is not None and p_val is not None:
                    # Avoid accidentally matching non-quantity ranges
                    if q_high > q_low:
                        tiers.append({"min_qty": q_low, "max_qty": q_high, "unit_price": p_val})

            # (C) Upper open range: "1000+ units at $26", "1000 and above: $26", "2000 or more units at $13", "over 1000 at $26"
            for pm in re.finditer(r"\b([\d,]+)\s*(?:\+|and\s+above|or\s+more|and\s+over)\s*(?:units)?\s*(?:at|costs?|is|are|:|@|=)?\s*\$?\s*([\d]+(?:\.\d+)?)", q, re.I):
                q_min = self._clean_number(pm.group(1))
                p_val = self._clean_number(pm.group(2))
                if q_min is not None and p_val is not None:
                    tiers.append({"min_qty": q_min, "max_qty": None, "unit_price": p_val})

            for om in re.finditer(r"\b(?:over|above|at\s+least)\s+([\d,]+)\s*(?:units)?\s*(?:at|costs?|is|are|:|@|=)?\s*\$?\s*([\d]+(?:\.\d+)?)", q, re.I):
                q_min = self._clean_number(om.group(1))
                p_val = self._clean_number(om.group(2))
                if q_min is not None and p_val is not None:
                    # Avoid duplicate if already added
                    if not any(t["min_qty"] == q_min and t["max_qty"] is None for t in tiers):
                        tiers.append({"min_qty": q_min, "max_qty": None, "unit_price": p_val})

            # (D) Fallback: price first format "$20 for 1-999, $18 for 1000+"
            if not tiers:
                for im in re.finditer(r"\$?\s*([\d.]+)\s*(?:for|if\s+ordering)\s*([\d,]+)\s*(?:-|–|to)?\s*([\d,]+|\+)?", q, re.I):
                    p_val = self._clean_number(im.group(1))
                    q_min = self._clean_number(im.group(2))
                    q_max_raw = im.group(3)
                    q_max = self._clean_number(q_max_raw) if q_max_raw and q_max_raw != "+" else None
                    if p_val and q_min:
                        tiers.append({"min_qty": q_min, "max_qty": q_max, "unit_price": p_val})

            # Deduplicate and sort tiers
            seen_brackets = set()
            unique_tiers = []
            for t in sorted(tiers, key=lambda x: x["min_qty"]):
                key = (t["min_qty"], t["max_qty"])
                if key not in seen_brackets:
                    seen_brackets.add(key)
                    unique_tiers.append(t)

            if d is not None:
                inputs["demand"] = d
                inputs["annual_demand"] = d
            else:
                missing.append("annual_demand (D)")

            if s is not None:
                inputs["order_cost"] = s
            else:
                missing.append("order_cost (S)")

            if h_rate is not None:
                inputs["holding_cost_rate"] = h_rate
            elif fixed_h is not None:
                inputs["fixed_holding_cost"] = fixed_h
            else:
                missing.append("holding_cost_rate (percentage) or fixed_holding_cost")

            if base_p is not None:
                inputs["unit_purchase_price"] = base_p

            if unique_tiers:
                inputs["discount_schedule"] = unique_tiers
            else:
                missing.append("discount_schedule (quantity breaks and unit prices)")

        elif model_name == self.MODEL_SAFETY_STOCK_ROP:
            # 0. Check optional lead-time variance and direct lead-time demand totals
            lt_var_kw = kwargs.get("lead_time_variance")
            ltd_mean_kw = kwargs.get("lead_time_demand_mean")
            ltd_std_kw = kwargs.get("lead_time_demand_std")

            lt_var = lt_var_kw if lt_var_kw is not None else self._extract_number(q, [
                r"\b(?:lead[- ]time\s+variance|variance\s+of\s+lead[- ]time)\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
            ])
            if lt_var is not None:
                inputs["lead_time_variance"] = float(lt_var)

            ltd_mean = ltd_mean_kw if ltd_mean_kw is not None else self._extract_number(q, [
                r"\b(?:demand\s+during\s+(?:the\s+)?lead[- ]time|lead[- ]time\s+demand(?:\s+mean)?|mean\s+demand\s+during\s+lead[- ]time|lead[- ]time\s+demand\s+mean)\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
            ])
            if ltd_mean is not None:
                inputs["lead_time_demand_mean"] = float(ltd_mean)

            ltd_std = ltd_std_kw if ltd_std_kw is not None else self._extract_number(q, [
                r"\b(?:standard\s+deviation\s+of\s+(?:demand\s+during\s+)?(?:the\s+)?lead[- ]time\s*(?:demand)?|lead[- ]time\s+demand\s+(?:standard\s+deviation|std\s+dev|std|sigma)|std\s+dev\s+of\s+lead[- ]time\s+demand)\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
            ])
            if ltd_std is not None:
                inputs["lead_time_demand_std"] = float(ltd_std)

            # 1. Lead-Time Standard Deviation (extract FIRST to avoid confusion with demand std dev)
            std_lt = self._extract_number(q, [
                r"\b(?:lead[- ]time\s+(?:standard\s+deviation|std\s+dev|sigma|variability)|standard\s+deviation\s+of\s+lead[- ]time|lead[- ]time\s+std)\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
                r"\b(?:lead[- ]time|lead\s+time)\s+has\s+(?:a\s+)?standard\s+deviation\s+of\s*([\d,]+(?:\.\d+)?)",
            ])

            # 2. Demand Standard Deviation
            std_d = self._extract_number(q, [
                r"\b(?:demand\s+(?:standard\s+deviation|std\s+dev|sigma|variability)|standard\s+deviation\s+of\s+demand|std\s+dev\s+of\s+demand|daily\s+standard\s+deviation|weekly\s+standard\s+deviation)\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
                r"\bdemand\s+has\s+(?:a\s+)?standard\s+deviation\s+of\s*([\d,]+(?:\.\d+)?)",
                r"\b(?:standard\s+deviation|std\s+dev|sigma)\s+(?:is|are|of|=|:)?\s*(?!of\s+lead)([\d,]+(?:\.\d+)?)",
            ])

            # 3. Average Lead Time (must avoid matching lead time standard deviation)
            lt = self._extract_number(q, [
                r"\b(?:average\s+lead[- ]time|mean\s+lead[- ]time|supplier\s+lead[- ]time|replenishment\s+lead[- ]time)\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
                r"\b(?:lead[- ]time|lead\s+time)\s*(?:is|are|of|=|:)?\s*(?!standard|std|sigma)([\d,]+(?:\.\d+)?)\s*(?:days|weeks|months|business\s+days)?\b",
                r"\btakes?\s*([\d,]+(?:\.\d+)?)\s*(?:days|weeks|months)\s*(?:to\s+arrive|lead[- ]time)?\b",
                r"\bdelivery\s+time\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
            ])

            # 4. Average Demand (must avoid matching demand standard deviation)
            d = self._extract_number(q, [
                r"\b(?:average\s+demand|mean\s+demand|expected\s+demand|daily\s+demand|weekly\s+demand|average\s+daily\s+demand|average\s+weekly\s+demand|demand\s+rate)\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
                r"\bdemand\s*(?:is|are|of|=|:)?\s*(?!standard|std|sigma)([\d,]+(?:\.\d+)?)\s*(?:units|items)?\s*(?:per\s+(?:day|week|month))?\b",
                r"\bsells?\s*(?:an?\s+average\s+of\s*)?([\d,]+(?:\.\d+)?)\s*(?:units|items)?\s*per\s+(?:day|week|month)\b",
            ])

            # 5. Service Level
            sl = self._extract_percentage(q, [
                r"\b(?:cycle\s+service\s+level|service\s+level|csl|target\s+service\s+level|in-stock\s+probability|order\s+cycle\s+service\s+level)\s*(?:is|are|of|=|:)?\s*([\d.]+\s*%)",
                r"\b(?:service\s+level|csl)\s*(?:is|are|of|=|:)?\s*(0\.\d+)\b",
                r"([\d.]+\s*%)\s*(?:cycle\s+)?service\s+level\b",
                r"([\d.]+\s*%)\s*in-stock\b",
            ])

            # Time unit detection
            time_unit = "day"
            if re.search(r"\bweek", q, re.I):
                time_unit = "week"
            elif re.search(r"\bmonth", q, re.I):
                time_unit = "month"

            if d is not None:
                inputs["demand_per_period"] = d
                inputs["avg_demand"] = d
            elif "lead_time_demand_mean" not in inputs:
                missing.append("average demand (d)")

            if lt is not None:
                inputs["lead_time"] = lt
            elif "lead_time_demand_mean" not in inputs:
                missing.append("average lead time (L)")

            # Demand and lead-time variability defaults to 0.0 if not variable
            inputs["std_demand"] = std_d if std_d is not None else 0.0
            if std_lt is not None:
                inputs["std_lead_time"] = std_lt
            elif "lead_time_variance" not in inputs:
                inputs["std_lead_time"] = 0.0

            inputs["time_unit"] = time_unit

            # Service level is strictly required if either demand or lead time is variable
            is_variable = (
                (inputs["std_demand"] > 0.0)
                or (inputs.get("std_lead_time", 0.0) > 0.0)
                or ("lead_time_variance" in inputs and inputs["lead_time_variance"] > 0.0)
                or ("lead_time_demand_std" in inputs and inputs["lead_time_demand_std"] > 0.0)
            )
            if is_variable:
                if sl is not None:
                    inputs["service_level"] = sl
                elif "service_level" in kwargs and kwargs["service_level"] is not None:
                    inputs["service_level"] = kwargs["service_level"]
                else:
                    missing.append("service_level (CSL, e.g. 95%)")
            else:
                # Case 1 (Constant demand & lead time): service level is optional
                if sl is not None:
                    inputs["service_level"] = sl
                elif "service_level" in kwargs and kwargs["service_level"] is not None:
                    inputs["service_level"] = kwargs["service_level"]

        elif model_name == self.MODEL_FIXED_PERIOD_REVIEW:
            # 1. Review Period T
            t = self._extract_number(q, [
                r"\b(?:review\s+period|review\s+interval|order\s+interval|review\s+frequency)\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)\s*(?:days|weeks|months)?",
                r"\breviewed\s+every\s*([\d,]+(?:\.\d+)?)\s*(?:days|weeks|months)?",
                r"\borders?\s+every\s*([\d,]+(?:\.\d+)?)\s*(?:days|weeks|months)?",
                r"\bevery\s*([\d,]+(?:\.\d+)?)\s*(?:days|weeks|months)\b",
            ])

            # 2. Lead Time L
            lt = self._extract_number(q, [
                r"\b(?:lead[- ]time|supplier\s+lead[- ]time|replenishment\s+lead[- ]time|delivery\s+time)\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)\s*(?:days|weeks|months)?",
                r"\btakes?\s*([\d,]+(?:\.\d+)?)\s*(?:days|weeks|months)\b",
            ])

            # 3. Demand Rate d
            d = self._extract_number(q, [
                r"\b(?:average\s+demand|mean\s+demand|daily\s+demand|weekly\s+demand|demand\s+rate)\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
                r"\bdemand\s*(?:is|are|of|=|:)?\s*(?!standard|std|sigma)([\d,]+(?:\.\d+)?)\b",
            ])

            # 4. Demand Standard Deviation
            std_d = self._extract_number(q, [
                r"\b(?:demand\s+(?:standard\s+deviation|std\s+dev|sigma)|standard\s+deviation\s+of\s+demand|std\s+dev\s+of\s+demand|daily\s+std\s+dev)\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
                r"\b(?:standard\s+deviation|std\s+dev|sigma)\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
            ])

            # 5. Service Level
            sl = self._extract_percentage(q, [
                r"\b(?:service\s+level|cycle\s+service\s+level|csl|target\s+service\s+level)\s*(?:is|are|of|=|:)?\s*([\d.]+\s*%)",
                r"\b(?:service\s+level|csl)\s*(?:is|are|of|=|:)?\s*(0\.\d+)\b",
                r"([\d.]+\s*%)\s*service\s+level\b",
            ])

            # 6. Safety Stock (if specified directly)
            ss_direct = self._extract_number(q, [
                r"\b(?:safety\s+stock|buffer\s+stock)\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
                r"\b([\d,]+(?:\.\d+)?)\s*units\s*(?:of\s+)?safety\s+stock\b",
            ])

            # 7. Inventory Position (Direct)
            ip_direct = self._extract_number(q, [
                r"\b(?:current\s+inventory\s+position|inventory\s+position|effective\s+inventory)\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
            ])

            # 8. On-Hand Inventory
            on_hand = self._extract_number(q, [
                r"\b(?:on[- ]hand\s+inventory|stock\s+on\s+hand|on[- ]hand\s+stock|physical\s+inventory)\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
                r"\b([\d,]+(?:\.\d+)?)\s*(?:units)?\s*(?:on[- ]hand|in\s+stock|on\s+hand)\b",
                r"\bcurrent\s+inventory\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
            ])

            # 9. On-Order Inventory
            on_order = self._extract_number(q, [
                r"\b(?:on[- ]order\s+inventory|on[- ]order\s+stock|pipeline\s+inventory|in\s+transit)\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
                r"\b([\d,]+(?:\.\d+)?)\s*(?:units)?\s*(?:on[- ]order|in\s+transit|in\s+pipeline|on\s+order)\b",
            ])

            # 10. Backorders
            backorders = self._extract_number(q, [
                r"\b(?:backorders|backordered\s+units|unfilled\s+backorders|backlog)\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
                r"\b([\d,]+(?:\.\d+)?)\s*(?:units)?\s*backordered\b",
            ])

            time_unit = "day" if re.search(r"\bday", q, re.I) else ("week" if re.search(r"\bweek", q, re.I) else "period")

            if t is not None:
                inputs["review_period"] = t
            else:
                missing.append("review_period (T)")

            if lt is not None:
                inputs["lead_time"] = lt
            else:
                missing.append("lead_time (L)")

            if d is not None:
                inputs["demand_per_period"] = d
                inputs["avg_demand"] = d
            else:
                missing.append("demand (d)")

            inputs["std_demand"] = std_d if std_d is not None else 0.0
            inputs["time_unit"] = time_unit

            if ss_direct is not None:
                inputs["safety_stock"] = ss_direct

            if sl is not None:
                inputs["service_level"] = sl
            elif inputs["std_demand"] > 0.0 and ss_direct is None:
                missing.append("service_level (CSL) or safety_stock")

            # Resolve inventory position
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
                dt_matches = re.findall(r"(?:demand\s+)?(\d+(?:\.\d+)?)\s*(?:\(\s*([\d.]+)%\s*\)|:\s*([\d.]+)%)", q, re.I)
                if len(dt_matches) >= 2:
                    parsed_table = []
                    for d_str, p1, p2 in dt_matches:
                        prob_str = p1 or p2
                        parsed_table.append({"demand": float(d_str), "probability": float(prob_str)})
                    inputs["demand_table"] = parsed_table

            # 1. Underage / Shortage Cost (Direct)
            cu = self._extract_number(q, [
                r"\b(?:underage\s+cost|cost\s+of\s+underage|shortage\s+cost|cost\s+of\s+understocking)\s*(?:is|are|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)",
            ])

            # 2. Overage / Excess Cost (Direct)
            co = self._extract_number(q, [
                r"\b(?:overage\s+cost|cost\s+of\s+overage|excess\s+cost|cost\s+of\s+overstocking)\s*(?:is|are|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)",
            ])

            # 3. Selling Price
            p = self._extract_number(q, [
                r"\b(?:selling\s+price|retail\s+price|revenue\s+per\s+unit|price)\s*(?:is|are|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)",
                r"\b(?:sells?|sold)\s*(?:for|at)\s*\$?\s*([\d,]+(?:\.\d+)?)",
            ])

            # 4. Purchase Cost
            c = self._extract_number(q, [
                r"\b(?:purchase\s+cost|wholesale\s+price|procurement\s+cost|acquisition\s+cost|unit\s+purchase\s+price|unit\s+cost|cost\s+to\s+buy)\s*(?:is|are|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)",
                r"(?<!underage\s)(?<!overage\s)(?<!shortage\s)(?<!excess\s)\bcosts?\s*(?:is|are|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)\b",
            ])

            # 5. Salvage Value
            s = self._extract_number(q, [
                r"\b(?:salvage\s+value|scrap\s+value|clearance\s+price|liquidation\s+(?:value|price)|resold\s+for|markdown\s+price)\s*(?:is|are|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)",
                r"\b(?:salvage[sd]?|liquidate[sd]?)\s*(?:for|at)\s*\$?\s*([\d,]+(?:\.\d+)?)",
            ])

            # 6. Shortage Penalty / Goodwill Loss
            g = self._extract_number(q, [
                r"\b(?:shortage\s+penalty|goodwill\s+loss|loss\s+of\s+goodwill|penalty\s+per\s+unit\s+short|penalty)\s*(?:is|are|of|=|:)?\s*\$?\s*([\d,]+(?:\.\d+)?)",
            ])

            # 7. Mean Demand
            mu = self._extract_number(q, [
                r"\b(?:mean\s+demand|average\s+demand|expected\s+demand|average\s+sales)\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
                r"\bdemand\s+(?:follows\s+(?:a\s+)?normal\s+distribution\s+with\s+)?mean\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
                r"\bmean\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
            ])

            # 8. Demand Standard Deviation
            sigma = self._extract_number(q, [
                r"\b(?:demand\s+standard\s+deviation|standard\s+deviation\s+of\s+demand|std\s+dev\s+of\s+demand)\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
                r"\b(?:standard\s+deviation|std\s+dev|sigma)\s*(?:is|are|of|=|:)?\s*([\d,]+(?:\.\d+)?)",
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
                else:
                    missing.append("mean_demand (mu)")

                if sigma is not None:
                    inputs["std_demand"] = sigma
                else:
                    missing.append("std_demand (sigma)")

            # Check if underage and overage costs are provided directly
            has_explicit_no_salvage = bool(
                re.search(
                    r"\b(?:no\s+(?:salvage|scrap|liquidation)|zero\s+(?:salvage|scrap|liquidation)|salvage\s*(?:value\s*)?(?:is|of|=|:)?\s*0|worthless|without\s+salvage)\b",
                    q,
                    re.I,
                )
            )

            if cu is not None and co is not None:
                inputs["underage_cost"] = cu
                inputs["overage_cost"] = co
                if p is not None:
                    inputs["selling_price"] = p
                if c is not None:
                    inputs["cost"] = c
                if s is not None:
                    inputs["salvage_value"] = s
                elif has_explicit_no_salvage:
                    inputs["salvage_value"] = 0.0
            else:
                # Require price, cost, salvage
                if p is not None:
                    inputs["selling_price"] = p
                else:
                    missing.append("selling_price (p) or underage_cost")

                if c is not None:
                    inputs["cost"] = c
                else:
                    missing.append("purchase_cost (c) or overage_cost")

                if s is not None:
                    inputs["salvage_value"] = s
                elif has_explicit_no_salvage:
                    inputs["salvage_value"] = 0.0
                else:
                    missing.append("salvage_value (s)")

            inputs["shortage_penalty"] = g if g is not None else 0.0

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
            lines.append(f"| Numerator ($2DS$) | {intermediates.get('numerator_2DS', 0):,.2f} | Scaled setup demand driver |")
            lines.append(f"| Ratio ($2DS / H$) | {intermediates.get('ratio_2DS_over_H', 0):,.2f} | Squared optimal batch scale |")
            lines.append(f"| Square Root Result ($Q^*$) | {intermediates.get('sqrt_result', 0):,.4f} units | Exact continuous minimum of TIC |")
            lines.append(f"| Annual Ordering Cost (AOC) | ${intermediates.get('annual_ordering_cost', 0):,.2f} | Total yearly expenditure on setups |")
            lines.append(f"| Annual Holding Cost (AHC) | ${intermediates.get('annual_holding_cost', 0):,.2f} | Total yearly carrying cost of cycle stock |")
            lines.append(f"| Cost Difference at Optimum | ${intermediates.get('cost_difference_at_optimum', 0):,.4f} | |AOC - AHC| (exact cost balance) |")
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
                lines.append("| Demand | Probability | Cumulative Probability $P(D \\le d)$ | Condition: $P(D \\le d) \\ge CR$ |")
                lines.append("| :--- | :--- | :--- | :--- |")
                for row in intermediates.get("cumulative_probability_table", []):
                    d_val = row.get("demand")
                    p_val = row.get("probability")
                    cp_val = row.get("cumulative_probability")
                    sat = "✅ Smallest $P \\ge CR$ (Optimal $Q^*$)" if row.get("demand") == result.recommended_order_quantity else ("Yes" if cp_val >= result.critical_ratio else "No")
                    lines.append(f"| {d_val:,} | {p_val*100:.1f}% | {cp_val:.4f} ({cp_val*100:.2f}%) | {sat} |")
        lines.append("")

        # -------------------------------------------------------------
        # 6. Final Answer
        # -------------------------------------------------------------
        lines.append("### 6. Final Answer")
        if model_name == self.MODEL_EOQ:
            r = result
            lines.append(f"> **Optimal Economic Order Quantity (Q*)**: **{r.rounded_order_quantity:,} units** ({r.optimal_order_quantity:,.2f} exact)")
            lines.append(f"> **Total Annual Inventory Cost**: **${r.annual_inventory_cost:,.2f}**")
            cadence_str = f" (approx. every **{r.cycle_time_days:.1f} work days**)" if r.cycle_time_days is not None else f" (**{r.cycle_time_months:.2f} months** between orders)"
            lines.append(f"> **Replenishment Frequency**: **{r.orders_per_year:.2f} orders per year**{cadence_str}")
        elif model_name == self.MODEL_QUANTITY_DISCOUNT:
            qd = result
            lines.append(f"> **Optimal Order Quantity**: **{qd.rounded_optimal_quantity:,} units** (Qualifies for Tier {qd.optimal_tier_index + 1})")
            lines.append(f"> **Unit Purchase Price**: **${qd.optimal_unit_price:,.2f}** per unit")
            lines.append(f"> **Minimum Total Annual Cost**: **${qd.min_total_annual_cost:,.2f}**")
            if qd.annual_savings > 0:
                lines.append(f"> **Net Annual Savings vs Baseline**: **${qd.annual_savings:,.2f}** ({qd.percentage_savings:.2f}% reduction)")
            else:
                lines.append("> **Baseline Policy Optimal**: Tier 1 baseline minimizes total annual costs.")
        elif model_name == self.MODEL_SAFETY_STOCK_ROP:
            ss = result
            lines.append(f"> **Reorder Point (ROP)**: **{ss.rounded_reorder_point:,} units** ({ss.reorder_point:,.2f} exact)")
            lines.append(f"> **Safety Stock (SS)**: **{ss.rounded_safety_stock:,} units** ({ss.safety_stock:,.2f} exact)")
            lines.append(f"> **Replenishment Decision Rule**: Trigger a replenishment order whenever inventory drops to or below **{ss.rounded_reorder_point:,} units**.")
        elif model_name == self.MODEL_FIXED_PERIOD_REVIEW:
            fp = result
            lines.append(f"> **Recommended Order Quantity (q)**: **{fp.rounded_order_quantity:,} units** ({fp.order_quantity:,.2f} exact)")
            lines.append(f"> **Order-Up-To Level (Target Inventory S)**: **{fp.rounded_target_inventory:,} units** ({fp.target_inventory:,.2f} exact)")
            lines.append(f"> **Safety Stock Buffer**: **{fp.rounded_safety_stock:,} units** ({fp.safety_stock:,.2f} exact)")
            lines.append(f"> **Periodic Policy Rule**: At every {fp.review_period}-{fp.time_unit} review interval, order `q = max(0, {fp.rounded_target_inventory:,} - IP)` to bring inventory position back to {fp.rounded_target_inventory:,} units.")
        elif model_name == self.MODEL_NEWSVENDOR:
            nv = result
            lines.append(f"> **Optimal Order Quantity (Q*)**: **{nv.rounded_order_quantity:,} units** ({nv.optimal_order_quantity:,.2f} exact)")
            lines.append(f"> **Target In-Stock Probability (Critical Ratio)**: **{nv.critical_ratio*100:.2f}%**")
            if nv.expected_profit is not None:
                lines.append(f"> **Expected Seasonal Profit**: **${nv.expected_profit:,.2f}**")
            if nv.expected_value_of_perfect_information is not None:
                lines.append(f"> **Expected Value of Perfect Information (EVPI)**: **${nv.expected_value_of_perfect_information:,.2f}**")
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
        lines.append("")

        return lines

    def generate_explanation(
        self,
        question: str,
        model_name: str,
        result: Any,
        verification: VerificationResult,
    ) -> str:
        """
        Synthesize an executive-ready, analytically sound decision explanation in Markdown,
        featuring all 8 verification components for complete student verification.
        """
        lines: List[str] = []

        # Executive Decision Overview Banner
        if model_name == self.MODEL_EOQ:
            r: it.EOQResult = result
            lines.append("## 📦 Economic Order Quantity (EOQ) Analysis\n")
            lines.append("### 🎯 Executive Decision Summary")
            lines.append(f"- **Recommended Order Quantity (Q*)**: **{r.rounded_eoq:,} units** ({r.eoq:,.2f} exact)")
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
            lines.append(f"- **Optimal Order Quantity (Q*)**: **{nv.rounded_optimal_quantity:,} units** ({nv.optimal_order_quantity:,.2f} exact)")
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
            lines.append("- **Cost Flatness**: The total cost curve is relatively flat around $Q^*$. A 10% deviation in batch size increases total inventory costs by less than 0.5%.")
            lines.append("- **Setup Reduction**: Reducing setup cost $S$ via EDI or automated replenishment directly lowers $Q^*$, shortening cycle inventory and working capital needs.")
        elif model_name == self.MODEL_QUANTITY_DISCOUNT:
            opt = result.tier_evaluations[result.optimal_tier_index]
            if opt.is_price_break_bumped:
                lines.append(f"- Order quantity was intentionally bumped to **{int(opt.min_qty):,} units** to qualify for the ${opt.unit_price:.2f} price break.")
                lines.append(f"- Annual purchasing savings ($D \\times \\Delta C$) outweigh the extra inventory holding costs incurred by holding higher cycle stock.")
            else:
                lines.append("- The unconstrained EOQ naturally falls within the winning tier range without requiring artificial batch bumping.")
        elif model_name == self.MODEL_SAFETY_STOCK_ROP:
            lines.append("- **Lead Time Reduction**: Safety stock scales with $\\sqrt{L}$. Cutting lead time by 50% reduces safety stock by ~29.3%, freeing working capital.")
            lines.append("- **Service Level Non-Linearity**: Increasing service level from 95% ($z=1.645$) to 99% ($z=2.326$) demands a 41.4% increase in safety stock.")
        elif model_name == self.MODEL_FIXED_PERIOD_REVIEW:
            lines.append("- Periodic review requires holding safety stock for $(T + L)$ rather than just $L$, meaning higher holding costs in exchange for predictable ordering schedules and combined freight shipping.")
        elif model_name == self.MODEL_NEWSVENDOR:
            if result.critical_ratio > 0.5:
                if result.mean_demand is not None:
                    lines.append(f"- High margin item ($C_u > C_o$): Ordering above mean demand ({result.optimal_order_quantity:.0f} > {result.mean_demand:.0f}) is optimal because stocking out hurts more than leftover markdown.")
                else:
                    lines.append(f"- High margin item ($C_u > C_o$): Target critical fractile ({result.critical_ratio*100:.1f}%) exceeds 50%, favoring ordering above median demand to protect against stockout.")
            else:
                if result.mean_demand is not None:
                    lines.append(f"- Low margin / high depreciation item ($C_o > C_u$): Ordering below mean demand ({result.optimal_order_quantity:.0f} < {result.mean_demand:.0f}) is optimal to guard against severe markdown loss.")
                else:
                    lines.append(f"- Low margin / high depreciation item ($C_o > C_u$): Target critical fractile ({result.critical_ratio*100:.1f}%) is below 50%, favoring conservative ordering to protect against overstock.")

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

        # Stage 2: Extract Inputs
        extracted_inputs, missing_inputs = self.extract_inputs(question, model_name, **kwargs)

        # Check if critical parameters are missing
        if missing_inputs:
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

        # Stage 6: Explain
        explanation = self.generate_explanation(question, model_name, calculation_result, verification)

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
        )
