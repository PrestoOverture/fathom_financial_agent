import asyncio
from graph.workflow import create_graph

async def main():
    graph = create_graph()
    input_state = {"question": "What percent of Ulta Beauty's total spend on stock repurchases for FY 2023 occurred in Q4 of FY2023?"}
    print(f"Question: {input_state['question']}")
    result = await graph.ainvoke(input_state)

    print("\n--- Output ---")
    print(f"Answer: {result.get('answer')}")
    print(f"\nReasoning:\n{result.get('reasoning_logs')}")
    print(f"\nArithmetic Errors Found: {result.get('arithmetic_errors_found')}")
    print(f"Verification Log: {result.get('verification_log')}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())