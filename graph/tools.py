import ast
import operator
from typing import Optional

# convert messy financial strings into usable numbers
def clean_number(num_str: str) -> Optional[float]:
    if not num_str:
        return None
    s = num_str.strip()

    # handle negative sign
    is_negative = False
    if s.startswith("(") and s.endswith(")"):
        is_negative = True
        s = s[1:-1]
    
    # remove currency symbols
    s = s.replace("$", "").replace(",", "").replace("€", "").replace("£", "")

    # replace percentage
    is_percent = False
    if s.endswith("%"):
        is_percent = True
        s = s.replace("%", "")
        
    try:
        val = float(s)
        if is_percent:
            val = val / 100.0
        
        if is_negative:
            val = -val
            
        return val
    except ValueError:
        return None


def calculate(expression: str) -> float:
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
    }

    def eval_node(node):
        # convert string to number if possible
        if isinstance(node, ast.Constant): 
            return node.value
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            else:
                raise TypeError(f"Non-numeric constant: {node.value}")
        
        # handle binary operations
        elif isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            if type(node.op) in operators:
                return operators[type(node.op)](left, right)
            
        # handle unary operations
        elif isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            if type(node.op) in operators:
                return operators[type(node.op)](operand)
                
        raise TypeError(f"Unsupported operation: {node}")

    try:
        tree = ast.parse(expression, mode='eval')
        return eval_node(tree.body)
    except (SyntaxError, TypeError, ZeroDivisionError):
        return None
    except Exception:
        return None

    

if __name__ == "__main__":
    print("Running tool tests...")
    assert clean_number("$1,234.50") == 1234.5
    assert clean_number("(500)") == -500.0
    assert clean_number("10%") == 0.1
    assert clean_number("(5.5%)") == -0.055
    assert clean_number("Not a number") is None
    print("clean_number: PASS")
    
    assert calculate("100 + 200") == 300.0
    assert calculate("100 - 50") == 50.0
    assert calculate("5 * 5") == 25.0
    assert calculate("10 / 2") == 5.0
    assert calculate("10 / 0") is None
    assert calculate("import os") is None
    print("calculate: PASS")