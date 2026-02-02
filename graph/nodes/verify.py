import re
from typing import Dict, Any, List, Optional, Tuple
from graph.tools import clean_number, calculate
from graph.state import AgentState

EQUATION_PATTERN = re.compile(
    r"((?:\(?\$?\s*\d[\d,\.]*\s*%?\)?))"
    r"\s*([-+*/–−])\s*"         
    r"((?:\(?\$?\s*\d[\d,\.]*\s*%?\)?))" 
    r"\s*=\s*"
    r"((?:\(?\$?\s*\d[\d,\.]*\s*%?\)?))"
)

MAG_MAP = {
    "thousand": 1e3,
    "million": 1e6,
    "billion": 1e9,
    "k": 1e3,
    "m": 1e6,
    "bn": 1e9,
    "b": 1e9,
}

NUM_WITH_MAG = re.compile(
    r"(\d[\d,]*\.?\d*)\s*(thousand|million|billion|k|m|bn|b)\b",
    flags=re.IGNORECASE,
)

PCT = re.compile(r"(\d[\d,]*\.?\d*)\s*%")

def normalize_operator(op_symbol: str) -> str:
    """Normalize fancy unicode dashes to standard minus."""
    if op_symbol in ['–', '−']: 
        return '-'
    return op_symbol

def normalize_expr(expr: str) -> str:
    # remove common lead-in text
    expr = re.sub(r"(?i)^\s*the calculation is:\s*", "", expr.strip())

    # normalize unicode minus
    expr = expr.replace("–", "-").replace("−", "-")

    # expand magnitudes: "328.1 million" -> "328.1*1000000"
    def _mag(match: re.Match) -> str:
        num = match.group(1).replace(",", "")
        unit = match.group(2).lower()
        mult = MAG_MAP[unit]
        return f"({num}*{mult})"
    expr = NUM_WITH_MAG.sub(_mag, expr)

    # convert "10%" inside expressions to "(10/100)"
    expr = PCT.sub(lambda match: f"({match.group(1).replace(',', '')}/100)", expr)

    # remove currency + commas
    expr = expr.replace("$", "").replace(",", "")
    return expr

def parse_rhs(rhs: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (rhs_as_fraction, rhs_as_percent) if rhs contains '%'.
    Otherwise (rhs_value, None).
    """
    rhs = rhs.strip()
    frac = clean_number(rhs)  # existing behavior: "29.8%" -> 0.298
    if frac is None:
        return (None, None)
    if "%" in rhs:
        return (frac, frac * 100.0)
    return (frac, None)

def verify_general_equalities(reasoning: str, log: List[str], corrections: List[Dict[str, Any]]) -> bool:
    found_any = False

    for raw_line in reasoning.splitlines():
        if "=" not in raw_line:
            continue

        line = re.sub(r"^\s*\d+\.\s*", "", raw_line).strip()
        if "=" not in line:
            continue

        lhs, rhs = line.split("=", 1)
        lhs_expr = normalize_expr(lhs)

        calculated = calculate(lhs_expr)
        if calculated is None:
            continue

        rhs_frac, rhs_pct = parse_rhs(rhs)
        if rhs_frac is None and rhs_pct is None:
            continue

        # choose the closer interpretation if RHS had '%'
        candidates = [(rhs_frac, "fraction")]
        if rhs_pct is not None:
            candidates.append((rhs_pct, "percent"))

        best_val, best_kind = min(
            candidates, key=lambda t: abs(calculated - t[0])
        )

        found_any = True

        denom = max(1.0, abs(calculated))
        diff = abs(calculated - best_val)
        if (diff / denom) < 0.01:
            log.append(f"Verified: {line.strip()}  (rhs as {best_kind})")
        else:
            log.append(
                f"Math Error: '{line.strip()}'. Real: {calculated:,.4f}, Claimed: {best_val:,.4f}"
            )
            corrections.append({
                "original_text": line.strip(),
                "claimed_val": best_val,
                "calculated_val": calculated
            })

    return found_any

def verify_reasoning(reasoning: str) -> Dict[str, Any]:
    log = []
    corrections = []
    has_error = False
    
    # find all patterns that look like "A + B = C"
    matches = EQUATION_PATTERN.findall(reasoning)

    found_any = False
    if matches:
        found_any = True

    for match in matches:
        raw_left, raw_op, raw_right, raw_claimed = match
        
        # clean numbers (returns None if invalid)
        left_val = clean_number(raw_left)
        right_val = clean_number(raw_right)
        claimed_val = clean_number(raw_claimed)
        
        # skip if any part is not a valid number
        if left_val is None or right_val is None or claimed_val is None:
            continue
            
        # calculate true value
        op = normalize_operator(raw_op)
        expr_str = f"{left_val} {op} {right_val}"
        calculated_val = calculate(expr_str)
        
        if calculated_val is None:
            log.append(f"Skipping invalid calculation: {expr_str}")
            continue
            
        # stable tolerance check
        # use a denominator of at least 1.0 to avoid division by zero or massive % shifts on tiny numbers
        denom = max(1.0, abs(calculated_val))
        diff = abs(calculated_val - claimed_val)
        
        # check if difference is > 1%
        if (diff / denom) < 0.01:
            log.append(f"Verified: {raw_left} {raw_op} {raw_right} = {raw_claimed}")
        else:
            has_error = True
            correction_msg = (
                f"Math Error: '{raw_left} {raw_op} {raw_right} = {raw_claimed}'. "
                f"Real: {calculated_val:,.2f}. Diff: {diff:,.2f}"
            )
            log.append(correction_msg)
            corrections.append({
                "original_text": f"{raw_left} {raw_op} {raw_right} = {raw_claimed}",
                "claimed_val": claimed_val,
                "calculated_val": calculated_val
            })

    # fallback: handle "(A/B)*100 = C%" style lines
    found_any = verify_general_equalities(reasoning, log, corrections) or found_any

    if not found_any:
        log.append("No explicit math equations found to verify.")

    return {
        "arithmetic_errors_found": has_error,
        "verification_log": log,
        "corrections": corrections
    }

def verify_math_node(state: AgentState) -> Dict:
    reasoning = state.reasoning_logs if state.reasoning_logs else ""

    if not reasoning:
        return {
            "verification_log": ["No reasoning trace found."],
            "arithmetic_errors_found": False,
        }

    result = verify_reasoning(reasoning)

    return {
        "arithmetic_errors_found": result["arithmetic_errors_found"],
        "verification_log": result["verification_log"],
    }

if __name__ == "__main__":
    print("Running Verify Logic Tests...")
    
    text_correct = "We see that $1,000 + $200 = $1,200."
    res = verify_reasoning(text_correct)
    assert res["arithmetic_errors_found"] is False
    assert "Verified" in res["verification_log"][0]
    print("Test 1 (Correct): PASS")
    
    text_wrong = "The net change is 500 - 200 = 400."
    res = verify_reasoning(text_wrong)
    assert res["arithmetic_errors_found"] is True
    assert "Math Error" in res["verification_log"][0]
    print("Test 2 (Wrong): PASS")
    
    text_finance = "Operating Income: ($5,000) / 2 = ($2,500)"
    res = verify_reasoning(text_finance)
    assert res["arithmetic_errors_found"] is False
    print("Test 3 (Financial Syntax): PASS")
    print(verify_reasoning("The calculation is: ($328.1 million / $1.1 billion) * 100 = 29.8%"))