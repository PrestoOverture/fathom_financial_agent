import re
from typing import Dict, Any
from graph.tools import clean_number, calculate
from graph.state import AgentState

EQUATION_PATTERN = re.compile(
    r"((?:\(?\$?\s*\d[\d,\.]*\s*%?\)?))"    # Group 1: Left Operand
    r"\s*([-+*/–−])\s*"                      # Group 2: Operator (inc. em-dash)
    r"((?:\(?\$?\s*\d[\d,\.]*\s*%?\)?))"    # Group 3: Right Operand
    r"\s*=\s*"                               # Equals
    r"((?:\(?\$?\s*\d[\d,\.]*\s*%?\)?))"    # Group 4: Claimed Result
)

def normalize_operator(op_symbol: str) -> str:
    """Normalize fancy unicode dashes to standard minus."""
    if op_symbol in ['–', '−']: 
        return '-'
    return op_symbol

def verify_reasoning(reasoning: str) -> Dict[str, Any]:
    log = []
    corrections = []
    has_error = False
    
    # Find all patterns that look like "A + B = C"
    matches = EQUATION_PATTERN.findall(reasoning)
    
    if not matches:
        log.append("No explicit math equations found to verify.")
    
    for match in matches:
        raw_left, raw_op, raw_right, raw_claimed = match
        
        # 1. Clean numbers (returns None if invalid)
        left_val = clean_number(raw_left)
        right_val = clean_number(raw_right)
        claimed_val = clean_number(raw_claimed)
        
        # Skip if any part is not a valid number
        if left_val is None or right_val is None or claimed_val is None:
            continue
            
        # 2. Calculate True Value
        op = normalize_operator(raw_op)
        expr_str = f"{left_val} {op} {right_val}"
        calculated_val = calculate(expr_str)
        
        if calculated_val is None:
            log.append(f"Skipping invalid calculation: {expr_str}")
            continue
            
        # 3. Stable Tolerance Check
        # We use a denominator of at least 1.0 to avoid division by zero or massive % shifts on tiny numbers
        denom = max(1.0, abs(calculated_val))
        diff = abs(calculated_val - claimed_val)
        
        # Check if difference is > 1%
        if (diff / denom) < 0.01:
            log.append(f"✅ Verified: {raw_left} {raw_op} {raw_right} = {raw_claimed}")
        else:
            has_error = True
            correction_msg = (
                f"❌ Math Error: '{raw_left} {raw_op} {raw_right} = {raw_claimed}'. "
                f"Real: {calculated_val:,.2f}. Diff: {diff:,.2f}"
            )
            log.append(correction_msg)
            corrections.append({
                "original_text": f"{raw_left} {raw_op} {raw_right} = {raw_claimed}",
                "claimed_val": claimed_val,
                "calculated_val": calculated_val
            })

    return {
        "arithmetic_errors_found": has_error,
        "verification_log": log,
        "corrections": corrections
    }

def verify_math_node(state: AgentState) -> Dict:
    """
    LangGraph node entry point.
    """
    # reasoning = ""
    # if hasattr(state, "reasoning_logs"):
    #     reasoning = state.reasoning_logs or ""
    # elif isinstance(state, dict):
    #     reasoning = state.get("reasoning_logs", "") or state.get("reasoning", "")

    reasoning = state.reasoning_logs if state.reasoning_logs else "" 
    if not reasoning:
        return {
            "verification_log": ["No reasoning trace found."],
            "arithmetic_errors_found": False
        }
    
    result = verify_reasoning(reasoning)
    
    # Return updates to the state
    return {
        "arithmetic_errors_found": result["arithmetic_errors_found"],
        "verification_log": result["verification_log"]
        # In a more advanced version, we would overwrite 'answer' here if a correction was found.
    }

if __name__ == "__main__":
    print("Running Verify Logic Tests...")
    
    # Test 1: Correct Math
    text_correct = "We see that $1,000 + $200 = $1,200."
    res = verify_reasoning(text_correct)
    assert res["arithmetic_errors_found"] is False
    assert "Verified" in res["verification_log"][0]
    print("Test 1 (Correct): PASS")
    
    # Test 2: Incorrect Math
    text_wrong = "The net change is 500 - 200 = 400." # Should be 300
    res = verify_reasoning(text_wrong)
    assert res["arithmetic_errors_found"] is True
    assert "Math Error" in res["verification_log"][0]
    print("Test 2 (Wrong): PASS")
    
    # Test 3: Financial Format with formatting characters
    text_finance = "Operating Income: ($5,000) / 2 = ($2,500)"
    res = verify_reasoning(text_finance)
    assert res["arithmetic_errors_found"] is False
    print("Test 3 (Financial Syntax): PASS")