import re
from typing import Any, Dict, List, Optional, Tuple
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
    rhs = rhs.strip()
    frac = clean_number(rhs)  # existing behavior: "29.8%" -> 0.298
    if frac is None:
        return (None, None)
    if "%" in rhs:
        return (frac, frac * 100.0)
    return (frac, None)

def verify_general_equalities(
    reasoning: str, log: List[str], corrections: List[Dict[str, Any]]
) -> Tuple[bool, bool]:
    found_any = False
    has_error = False

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
            has_error = True
            log.append(
                f"Math Error: '{line.strip()}'. Real: {calculated:,.4f}, Claimed: {best_val:,.4f}"
            )
            corrections.append({
                "original_text": line.strip(),
                "claimed_val": best_val,
                "calculated_val": calculated
            })

    return found_any, has_error


def has_math_equations(reasoning: str) -> bool:
    if not reasoning:
        return False

    if EQUATION_PATTERN.search(reasoning):
        return True

    for raw_line in reasoning.splitlines():
        if "=" not in raw_line:
            continue

        left, right = raw_line.split("=", 1)
        if re.search(r"\d", left) and re.search(r"\d", right):
            return True

    return False

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
    stage_b_found_any, stage_b_has_error = verify_general_equalities(reasoning, log, corrections)
    found_any = stage_b_found_any or found_any
    has_error = has_error or stage_b_has_error

    if not found_any:
        log.append("No explicit math equations found to verify.")

    return {
        "arithmetic_errors_found": has_error,
        "verification_log": log,
        "corrections": corrections
    }

def verify_math_node(state: AgentState) -> Dict[str, Any]:
    reasoning = state.reasoning_logs or state.raw_output or ""

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

    text_stage_b_wrong = "The calculation is: (100/4)*100 = 10%"
    res = verify_reasoning(text_stage_b_wrong)
    assert res["arithmetic_errors_found"] is True
    assert any("Math Error" in line for line in res["verification_log"])
    print("Test 4 (Stage B Wrong): PASS")

    text_stage_b_correct = "The calculation is: (328.1 million / 1.1 billion) * 100 = 29.8%"
    res = verify_reasoning(text_stage_b_correct)
    assert res["arithmetic_errors_found"] is False
    assert any("Verified" in line for line in res["verification_log"])
    print("Test 5 (Stage B Correct): PASS")

    assert has_math_equations("We compute 100 + 200 = 300.") is True
    assert has_math_equations("The calculation is: (100/4)*100 = 10%") is True
    assert has_math_equations("No equations in this reasoning.") is False
    print("Test 6 (Math Detection): PASS")
