import unittest
from graph.nodes.verify import has_math_equations, verify_math_node, verify_reasoning
from graph.state import AgentState
from graph.workflow import route_after_reason

class VerifyNodeTests(unittest.TestCase):
    def test_stage_b_wrong_equation_sets_error_flag(self) -> None:
        result = verify_reasoning("The calculation is: (100/4)*100 = 10%")
        self.assertTrue(result["arithmetic_errors_found"])
        self.assertTrue(any("Math Error" in line for line in result["verification_log"]))

    def test_stage_b_correct_equation_keeps_error_flag_false(self) -> None:
        result = verify_reasoning(
            "The calculation is: (328.1 million / 1.1 billion) * 100 = 29.8%"
        )
        self.assertFalse(result["arithmetic_errors_found"])
        self.assertTrue(any("Verified" in line for line in result["verification_log"]))

    def test_has_math_equations_detector(self) -> None:
        self.assertTrue(has_math_equations("We compute 100 + 200 = 300."))
        self.assertTrue(has_math_equations("The ratio is (100/4)*100 = 2500%"))
        self.assertFalse(has_math_equations("No equations in this reasoning trace."))

    def test_verify_math_node_uses_raw_output_fallback(self) -> None:
        state = AgentState(
            question="q",
            reasoning_logs="",
            raw_output="Reasoning:\nThe calculation is: (100/4)*100 = 10%",
        )

        result = verify_math_node(state)
        self.assertTrue(result["arithmetic_errors_found"])

    def test_route_after_reason_uses_raw_output_fallback(self) -> None:
        state_with_math = AgentState(
            question="q",
            reasoning_logs="",
            raw_output="Reasoning:\nThe ratio is 100/4 = 25",
        )
        state_without_math = AgentState(
            question="q",
            reasoning_logs="No equations here.",
            raw_output="",
        )

        self.assertEqual(route_after_reason(state_with_math), "verify")
        self.assertEqual(route_after_reason(state_without_math), "end")

if __name__ == "__main__":
    unittest.main()
